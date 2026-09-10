#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from PIL import Image
from tqdm import tqdm

from src.checkpoint import torch_load
from src.checkpoint_lite import load_trainable_state
from src.data import (
    CSIGSampleDataset,
    build_transform,
)
from src.dist_utils import (
    barrier,
    cleanup,
    init_distributed,
    is_main,
    make_autocast,
)
from src.memory import (
    CategoryMemory,
    MemoryConfig,
    extract_features,
)
from src.model import build_model
from src.postprocess import (
    maps_to_uint8,
    smooth_map,
    squash_score,
)
from src.submission import zip_submission


def parse_args():

    p = argparse.ArgumentParser()

    p.add_argument(
        "--test-root",
        required=True,
    )

    p.add_argument(
        "--ckpt",
        required=True,
    )

    p.add_argument(
        "--split",
        required=True,
        choices=["A", "B"],
    )

    p.add_argument(
        "--out-dir",
        required=True,
    )

    p.add_argument(
        "--zip",
        required=True,
    )

    # Image branch
    p.add_argument(
        "--image-sigma",
        type=float,
        default=7.0,
    )

    p.add_argument(
        "--image-topk",
        type=float,
        default=0.01,
    )

    # Pixel branch
    p.add_argument(
        "--pixel-sigma",
        type=float,
        default=2.0,
    )

    p.add_argument(
        "--mask-scale",
        type=float,
        default=1.0,
    )

    # Transductive memory
    p.add_argument(
        "--memory",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    p.add_argument(
        "--seen-memory-weight",
        type=float,
        default=0.0,
    )

    p.add_argument(
        "--unseen-memory-weight",
        type=float,
        default=0.35,
    )

    p.add_argument(
        "--memory-layer-start",
        type=int,
        default=0,
    )

    p.add_argument(
        "--memory-layer-end",
        type=int,
        default=4,
    )

    p.add_argument(
        "--consensus-ratio",
        type=float,
        default=0.70,
    )

    p.add_argument(
        "--consensus-neighbors",
        type=int,
        default=5,
    )

    p.add_argument(
        "--memory-samples",
        type=int,
        default=8,
    )

    p.add_argument(
        "--spatial-radius",
        type=int,
        default=3,
    )

    p.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    return p.parse_args()


# ============================================================
# Model
# ============================================================

def load_model(
    checkpoint_path,
    device,
):

    checkpoint = torch_load(
        checkpoint_path,
        map_location="cpu",
    )

    cfg = checkpoint["args"]

    model = build_model(
        encoder_name=cfg.get(
            "encoder",
            "dinov2reg_vit_large_14",
        ),
        inp_num=int(
            cfg.get(
                "inp_num",
                12,
            )
        ),
        decoder_depth=int(
            cfg.get(
                "decoder_depth",
                8,
            )
        ),
        bottleneck_drop=float(
            cfg.get(
                "bottleneck_drop",
                0.0,
            )
        ),
        residual_strength=float(
            cfg.get(
                "residual_strength",
                0.20,
            )
        ),
        encoder_source=cfg.get(
            "encoder_source",
            "auto",
        ),
    ).to(device)

    state = checkpoint[
        "model"
    ]

    # Old DDP compatibility.
    state = {
        (
            k[len("module."):]
            if k.startswith("module.")
            else k
        ): v
        for k, v in state.items()
    }

    load_trainable_state(
        model,
        state,
    )

    model.eval()

    trained_categories = set(
        checkpoint.get(
            "trained_categories",
            [],
        )
    )

    # Checkpoint may be very large.
    del state
    del checkpoint

    return (
        model,
        cfg,
        trained_categories,
    )


# ============================================================
# Helpers
# ============================================================

def group_categories(
    dataset,
):

    groups = defaultdict(list)

    for index, (
        category,
        _sid,
        _directory,
    ) in enumerate(
        dataset.samples
    ):

        groups[
            category
        ].append(index)

    return groups


def topk_score(
    maps,
    ratio,
):

    scores = []

    for amap in maps:

        flat = np.asarray(
            amap,
            dtype=np.float32,
        ).reshape(-1)

        k = max(
            1,
            int(
                len(flat)
                * ratio
            ),
        )

        k = min(
            k,
            len(flat),
        )

        scores.append(
            float(
                np.partition(
                    flat,
                    -k,
                )[-k:].mean()
            )
        )

    return float(
        max(scores)
    )


@torch.no_grad()
def model_predict(
    model,
    images,
    amp,
):

    with make_autocast(
        amp
    ):

        result = model(
            images,
            return_maps=True,
        )

    image_maps = (
        result[
            "image_map"
        ][:, 0]
        .float()
        .cpu()
        .numpy()
    )

    pixel_maps = (
        result[
            "pixel_map"
        ][:, 0]
        .float()
        .cpu()
        .numpy()
    )

    return (
        image_maps,
        pixel_maps,
    )


def calibrate_memory_scale(
    pixel_maps,
    memory_maps,
    selected,
    sigma,
    ratio,
):

    inp_scores = []
    mem_scores = []

    for index in selected:

        if index not in pixel_maps:
            continue

        inp = smooth_map(
            pixel_maps[index],
            sigma=sigma,
        )

        mem = smooth_map(
            memory_maps[index],
            sigma=sigma,
        )

        a = topk_score(
            inp,
            ratio,
        )

        b = topk_score(
            mem,
            ratio,
        )

        if (
            np.isfinite(a)
            and np.isfinite(b)
            and b > 1e-8
        ):

            inp_scores.append(a)
            mem_scores.append(b)

    if not mem_scores:

        return 1.0

    scale = (
        np.median(
            inp_scores
        )
        / max(
            np.median(
                mem_scores
            ),
            1e-8,
        )
    )

    return float(
        np.clip(
            scale,
            0.25,
            4.0,
        )
    )


# ============================================================
# Distributed helpers
# ============================================================

def gather_objects(
    obj,
    info,
):
    """
    Gather small Python objects such as rows and diagnostics
    to rank 0.

    Do NOT use this for feature maps or model tensors.
    """

    if info.world_size == 1:

        return [
            obj
        ]

    gathered = (
        [
            None
            for _ in range(
                info.world_size
            )
        ]
        if info.rank == 0
        else None
    )

    dist.gather_object(
        obj,
        gathered,
        dst=0,
    )

    return gathered


# ============================================================
# Main
# ============================================================

@torch.no_grad()
def main():

    args = parse_args()

    info = init_distributed()

    device = info.device

    # --------------------------------------------------------
    # Each rank owns one GPU and loads one copy of the model.
    #
    # No DDP wrapper is needed because inference is independent
    # at category granularity.
    # --------------------------------------------------------

    (
        model,
        cfg,
        trained_categories,
    ) = load_model(
        args.ckpt,
        device,
    )

    image_size = int(
        cfg.get(
            "image_size",
            448,
        )
    )

    dataset = CSIGSampleDataset(
        args.test_root,
        transform=build_transform(
            image_size,
            is_train=False,
        ),
        image_size=image_size,
    )

    categories = group_categories(
        dataset
    )

    category_names = sorted(
        categories
    )

    # --------------------------------------------------------
    # Old checkpoint fallback.
    # --------------------------------------------------------

    if not trained_categories:

        from src.data import (
            SEEN_CATEGORIES,
        )

        trained_categories = set(
            SEEN_CATEGORIES
        )

    output = Path(
        args.out_dir
    )

    # --------------------------------------------------------
    # Only rank 0 may reset/create the output directory.
    #
    # Without this barrier rank 1 can begin writing masks while
    # rank 0 simultaneously removes the directory.
    # --------------------------------------------------------

    if info.rank == 0:

        if output.exists():

            shutil.rmtree(
                output
            )

        (
            output
            / "predicted_masks"
        ).mkdir(
            parents=True,
            exist_ok=True,
        )

    barrier()

    memory_config = MemoryConfig(
        layer_start=(
            args.memory_layer_start
        ),
        layer_end=(
            args.memory_layer_end
        ),
        consensus_ratio=(
            args.consensus_ratio
        ),
        neighbors=(
            args.consensus_neighbors
        ),
        max_samples=(
            args.memory_samples
        ),
        radius=(
            args.spatial_radius
        ),
    )

    # ========================================================
    # Category-level distribution.
    #
    # Example for 2 GPUs:
    #
    # rank 0 -> category 0, 2, 4, ...
    # rank 1 -> category 1, 3, 5, ...
    #
    # A complete category always remains on one GPU.
    # ========================================================

    local_categories = (
        category_names[
            info.rank
            ::info.world_size
        ]
    )

    print(
        f"[rank {info.rank}] "
        f"device={device} "
        f"categories="
        f"{len(local_categories)}/"
        f"{len(category_names)}",
        flush=True,
    )

    local_rows = []

    local_diagnostics = {}

    iterator = tqdm(
        local_categories,
        desc=(
            f"GPU {info.rank}"
        ),
        ncols=100,
        position=info.rank,
        disable=False,
    )

    for category in iterator:

        indices = categories[
            category
        ]

        seen = (
            category
            in trained_categories
        )

        if (
            args.split == "A"
            and not seen
        ):

            print(
                f"[rank {info.rank}] "
                f"[warn] Test_A category "
                f"{category} is not in "
                "training categories.",
                flush=True,
            )

        memory_weight = (
            args.seen_memory_weight
            if seen
            else args.unseen_memory_weight
        )

        if not args.memory:

            memory_weight = 0.0

        cache = {}

        memory = (
            CategoryMemory(
                memory_config
            )
            if memory_weight > 0
            else None
        )

        # ====================================================
        # Pass 1
        # ====================================================

        for index in indices:

            item = dataset[
                index
            ]

            images = (
                item[
                    "images"
                ]
                .to(
                    device,
                    non_blocking=True,
                )
            )

            (
                image_maps,
                pixel_maps,
            ) = model_predict(
                model,
                images,
                args.amp,
            )

            cache[index] = {
                "group":
                    item[
                        "group_folder"
                    ],

                "image_maps":
                    image_maps,

                "pixel_maps":
                    pixel_maps,
            }

            if memory is not None:

                features = extract_features(
                    model.encoder,
                    images,
                    start=(
                        memory_config
                        .layer_start
                    ),
                    end=(
                        memory_config
                        .layer_end
                    ),
                )

                memory.add(
                    index,
                    features,
                )

            # We no longer need the input tensor explicitly.
            del images

        memory_maps = {}

        memory_scale = 1.0

        # ====================================================
        # Transductive memory
        # ====================================================

        if memory is not None:

            memory.finalize()

            patch = (
                model.encoder
                .patch_size
            )

            # Handle patch_size implementations which expose
            # either int or (h, w).
            if isinstance(
                patch,
                (
                    tuple,
                    list,
                ),
            ):

                patch_h = int(
                    patch[0]
                )

                patch_w = int(
                    patch[1]
                )

            else:

                patch_h = int(
                    patch
                )

                patch_w = int(
                    patch
                )

            grid_h = (
                image_size
                // patch_h
            )

            grid_w = (
                image_size
                // patch_w
            )

            for index in indices:

                maps = memory.maps(
                    index,
                    device,
                    grid_h,
                    grid_w,
                    (
                        image_size,
                        image_size,
                    ),
                )

                memory_maps[
                    index
                ] = (
                    maps
                    .float()
                    .cpu()
                    .numpy()
                )

            pixel_dict = {
                i:
                    cache[i][
                        "pixel_maps"
                    ]
                for i in indices
            }

            memory_scale = (
                calibrate_memory_scale(
                    pixel_dict,
                    memory_maps,
                    memory.selected,
                    args.pixel_sigma,
                    args.image_topk,
                )
            )

            del pixel_dict

        local_diagnostics[
            category
        ] = {
            "seen":
                seen,

            "samples":
                len(indices),

            "memory_weight":
                memory_weight,

            "memory_scale":
                memory_scale,

            "memory_selected":
                (
                    memory.selected
                    if memory is not None
                    else []
                ),
        }

        # ====================================================
        # Final prediction
        # ====================================================

        for index in indices:

            item = cache[
                index
            ]

            image_maps = smooth_map(
                item[
                    "image_maps"
                ],
                sigma=(
                    args.image_sigma
                ),
            )

            raw_image_score = topk_score(
                image_maps,
                args.image_topk,
            )

            image_score = squash_score(
                raw_image_score
            )

            pixel_maps = item[
                "pixel_maps"
            ]

            if (
                memory is not None
                and index
                in memory_maps
            ):

                mem = (
                    memory_maps[
                        index
                    ]
                    * memory_scale
                )

                pixel_maps = (
                    (
                        1.0
                        - memory_weight
                    )
                    * pixel_maps
                    + memory_weight
                    * mem
                )

            pixel_maps = smooth_map(
                pixel_maps,
                sigma=(
                    args.pixel_sigma
                ),
            )

            masks = maps_to_uint8(
                pixel_maps,
                scale=(
                    args.mask_scale
                ),
            )

            destination = (
                output
                / "predicted_masks"
                / item[
                    "group"
                ]
            )

            # Categories are disjoint across ranks, therefore
            # each group should be owned by exactly one rank.
            destination.mkdir(
                parents=True,
                exist_ok=True,
            )

            for view in range(
                masks.shape[0]
            ):

                Image.fromarray(
                    masks[
                        view
                    ],
                    mode="L",
                ).save(
                    destination
                    / (
                        f"{view}"
                        f"_mask.png"
                    )
                )

            local_rows.append(
                (
                    item[
                        "group"
                    ],
                    float(
                        image_score
                    ),
                )
            )

        # ====================================================
        # Category cleanup.
        # ====================================================

        del memory

        memory_maps.clear()

        cache.clear()

        if (
            torch.cuda
            .is_available()
        ):

            torch.cuda.empty_cache()

    # ========================================================
    # All mask files must exist before rank 0 builds the final
    # submission.
    # ========================================================

    barrier()

    # ========================================================
    # Gather small metadata objects only.
    # ========================================================

    gathered_rows = gather_objects(
        local_rows,
        info,
    )

    gathered_diagnostics = gather_objects(
        local_diagnostics,
        info,
    )

    # ========================================================
    # Only rank 0 creates CSV/meta/ZIP.
    # ========================================================

    if info.rank == 0:

        rows = []

        for rank_rows in gathered_rows:

            rows.extend(
                rank_rows
            )

        diagnostics = {}

        for rank_diagnostics in (
            gathered_diagnostics
        ):

            diagnostics.update(
                rank_diagnostics
            )

        # ----------------------------------------------------
        # Defensive checks.
        # ----------------------------------------------------

        expected_rows = sum(
            len(v)
            for v in categories.values()
        )

        if (
            len(rows)
            != expected_rows
        ):

            raise RuntimeError(
                "Distributed inference "
                "produced an unexpected "
                "number of rows: "
                f"expected="
                f"{expected_rows}, "
                f"actual={len(rows)}"
            )

        group_names = [
            row[0]
            for row in rows
        ]

        if (
            len(
                set(
                    group_names
                )
            )
            != len(
                group_names
            )
        ):

            raise RuntimeError(
                "Duplicate group_folder "
                "detected after distributed "
                "inference."
            )

        rows.sort(
            key=lambda x:
                x[0]
        )

        # ----------------------------------------------------
        # CSV
        # ----------------------------------------------------

        with open(
            output
            / "submission.csv",
            "w",
            encoding="utf-8-sig",
            newline="",
        ) as f:

            writer = csv.writer(
                f
            )

            writer.writerow([
                "group_folder",
                "anomaly_score",
            ])

            for group, score in rows:

                writer.writerow([
                    group,
                    f"{score:.8f}",
                ])

        # ----------------------------------------------------
        # Metadata
        # ----------------------------------------------------

        (
            output
            / "infer_meta.json"
        ).write_text(
            json.dumps(
                {
                    "split":
                        args.split,

                    "checkpoint":
                        args.ckpt,

                    "image_sigma":
                        args.image_sigma,

                    "pixel_sigma":
                        args.pixel_sigma,

                    "image_topk":
                        args.image_topk,

                    "mask_scale":
                        args.mask_scale,

                    "memory":
                        args.memory,

                    "seen_memory_weight":
                        args.seen_memory_weight,

                    "unseen_memory_weight":
                        args.unseen_memory_weight,

                    "world_size":
                        info.world_size,

                    "categories":
                        diagnostics,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        # ----------------------------------------------------
        # ZIP
        # ----------------------------------------------------

        zip_path = zip_submission(
            str(
                output
            ),
            args.zip,
        )

        print(
            f"[done] {zip_path}",
            flush=True,
        )

    # Do not allow non-zero ranks to tear down distributed
    # communication while rank 0 is still creating the zip.
    barrier()


if __name__ == "__main__":

    os.environ.setdefault(
        "NCCL_P2P_DISABLE",
        "1",
    )

    os.environ.setdefault(
        "NCCL_IB_DISABLE",
        "1",
    )

    try:

        main()

    finally:

        cleanup()
