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

from PIL import Image
from tqdm import tqdm

from src.checkpoint import torch_load
from src.checkpoint_lite import load_trainable_state
from src.data import (
    CSIGSampleDataset,
    build_transform,
)
from src.dist_utils import (
    cleanup,
    init_distributed,
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

    return (
        model,
        cfg,
        trained_categories,
    )


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
    with make_autocast(amp):
        result = model(
            images,
            return_maps=True,
        )

    image_maps = (
        result["image_map"][:, 0]
        .float()
        .cpu()
        .numpy()
    )

    pixel_maps = (
        result["pixel_map"][:, 0]
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
        np.median(inp_scores)
        / max(
            np.median(mem_scores),
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


@torch.no_grad()
def main():
    args = parse_args()

    info = init_distributed()

    if info.world_size != 1:
        raise RuntimeError(
            "Transductive inference should "
            "run on one GPU."
        )

    device = info.device

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

    # Old checkpoint fallback.
    if not trained_categories:
        from src.data import SEEN_CATEGORIES
        trained_categories = set(
            SEEN_CATEGORIES
        )

    output = Path(
        args.out_dir
    )

    if output.exists():
        shutil.rmtree(output)

    (
        output
        / "predicted_masks"
    ).mkdir(
        parents=True,
        exist_ok=True,
    )

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

    rows = []

    diagnostics = {}

    for category in tqdm(
        sorted(categories),
        desc="categories",
        ncols=100,
    ):
        indices = categories[
            category
        ]

        seen = (
            category
            in trained_categories
        )

        # Test_A should contain only seen categories.
        if (
            args.split == "A"
            and not seen
        ):
            print(
                f"[warn] Test_A category "
                f"{category} is not in "
                "training categories."
            )

        memory_weight = (
            args.seen_memory_weight
            if seen
            else args.unseen_memory_weight
        )

        # For Test_A, default behavior should be
        # deterministic baseline model inference.
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

        # -----------------------------------------
        # Pass 1
        # -----------------------------------------

        for index in indices:
            item = dataset[index]

            images = item[
                "images"
            ].to(
                device,
                non_blocking=True,
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

        memory_maps = {}
        memory_scale = 1.0

        # -----------------------------------------
        # Transductive memory
        # -----------------------------------------

        if memory is not None:
            memory.finalize()

            patch = (
                model.encoder.patch_size
            )

            grid_h = (
                image_size // patch
            )

            grid_w = (
                image_size // patch
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

                memory_maps[index] = (
                    maps.float()
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

        diagnostics[
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
                    if memory
                    is not None
                    else []
                ),
        }

        # -----------------------------------------
        # Final prediction
        # -----------------------------------------

        for index in indices:
            item = cache[index]

            image_maps = smooth_map(
                item["image_maps"],
                sigma=(
                    args.image_sigma
                ),
            )

            # Keep Image branch independent from
            # transductive memory.
            raw_image_score = (
                topk_score(
                    image_maps,
                    args.image_topk,
                )
            )

            image_score = (
                squash_score(
                    raw_image_score
                )
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
                    memory_maps[index]
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
                / item["group"]
            )

            destination.mkdir(
                parents=True,
                exist_ok=True,
            )

            for view in range(
                masks.shape[0]
            ):
                Image.fromarray(
                    masks[view],
                    mode="L",
                ).save(
                    destination
                    / f"{view}_mask.png"
                )

            rows.append(
                (
                    item["group"],
                    float(
                        image_score
                    ),
                )
            )

        # Memory must not remain across categories.
        del memory
        memory_maps.clear()
        cache.clear()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    rows.sort(
        key=lambda x:
        x[0]
    )

    with open(
        output / "submission.csv",
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        writer = csv.writer(f)

        writer.writerow([
            "group_folder",
            "anomaly_score",
        ])

        for group, score in rows:
            writer.writerow([
                group,
                f"{score:.8f}",
            ])

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

                "categories":
                    diagnostics,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    zip_path = zip_submission(
        str(output),
        args.zip,
    )

    print(
        f"[done] {zip_path}"
    )


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
