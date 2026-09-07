#!/usr/bin/env python3
"""
A/B Test B

ONLY experimental change:
    Pixel anomaly maps are converted using robust MAD normalization.

Image score pipeline is intentionally identical to baseline.

Robust pixel transform:

    z =
        (A - Median(A))
        /
        (1.4826 * MAD(A) + eps)

    pixel =
        sigmoid(
            (z - tau)
            / temperature
        )

IMPORTANT:
    Image raw_scores are computed BEFORE this transformation.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from src.checkpoint import load_inpformer, torch_load
from src.data import (
    CSIGImageDataset,
    CSIGSampleDataset,
    build_transform,
)
from src.dist_utils import (
    all_gather_object,
    barrier,
    cleanup,
    init_distributed,
    is_main,
    setup_seed,
)
from src.encoder import prefetch_encoder_weights
from src.model import anomaly_map_from_features
from src.postprocess import (
    calibrate_scale,
    maps_to_uint8,
    smooth_map,
    squash_score,
)
from src.refine import (
    NormalStats,
    apply_view_gate,
    apply_view_refine,
)
from src.submission import zip_submission


def parse_args():
    p = argparse.ArgumentParser(
        description="INP-Former A/B Test B: robust pixel MAD"
    )

    p.add_argument(
        "--test-root",
        required=True,
    )

    p.add_argument(
        "--ckpt",
        required=True,
    )

    p.add_argument(
        "--out-dir",
        default="outputs/robust_pixel",
    )

    p.add_argument(
        "--zip",
        default="outputs/robust_pixel.zip",
    )

    p.add_argument(
        "--encoder",
        default="",
    )

    p.add_argument(
        "--encoder-source",
        default="auto",
    )

    p.add_argument(
        "--image-size",
        type=int,
        default=448,
    )

    p.add_argument(
        "--inp-num",
        type=int,
        default=6,
    )

    p.add_argument(
        "--decoder-depth",
        type=int,
        default=8,
    )

    p.add_argument(
        "--bottleneck-drop",
        type=float,
        default=0.0,
    )

    p.add_argument(
        "--residual",
        action="store_true",
    )

    p.add_argument(
        "--samples-per-batch",
        type=int,
        default=2,
    )

    p.add_argument(
        "--num-workers",
        type=int,
        default=2,
    )

    p.add_argument(
        "--sigma",
        type=float,
        default=2.5,
    )

    p.add_argument(
        "--max-ratio",
        type=float,
        default=0.01,
    )

    p.add_argument(
        "--reduce",
        default="max",
        choices=[
            "max",
            "mean",
            "lse",
        ],
    )

    p.add_argument(
        "--mask-scale",
        type=float,
        default=0.0,
    )

    p.add_argument(
        "--train-root",
        default="",
    )

    p.add_argument(
        "--calibrate-n",
        type=int,
        default=256,
    )

    p.add_argument(
        "--tta-flip",
        action="store_true",
    )

    p.add_argument(
        "--amp",
        dest="amp",
        action="store_true",
        default=True,
    )

    p.add_argument(
        "--no-amp",
        dest="amp",
        action="store_false",
    )

    p.add_argument(
        "--refine",
        dest="refine",
        action="store_true",
        default=True,
    )

    p.add_argument(
        "--no-refine",
        dest="refine",
        action="store_false",
    )

    p.add_argument(
        "--gamma",
        type=float,
        default=1.0,
    )

    p.add_argument(
        "--border",
        type=int,
        default=16,
    )

    p.add_argument(
        "--fg-gate",
        dest="fg_gate",
        action="store_true",
        default=False,
    )

    p.add_argument(
        "--no-fg-gate",
        dest="fg_gate",
        action="store_false",
    )

    p.add_argument(
        "--stats-path",
        default="",
    )

    p.add_argument(
        "--stats-per-class",
        type=int,
        default=20,
    )

    p.add_argument(
        "--view-gate",
        dest="view_gate",
        action="store_true",
        default=False,
    )

    p.add_argument(
        "--no-view-gate",
        dest="view_gate",
        action="store_false",
    )

    p.add_argument(
        "--gate-k",
        type=float,
        default=1.25,
    )

    p.add_argument(
        "--gate-temp",
        type=float,
        default=0.35,
    )

    p.add_argument(
        "--gate-floor",
        type=float,
        default=0.0,
    )

    p.add_argument(
        "--gate-hard",
        dest="gate_hard",
        action="store_true",
        default=True,
    )

    p.add_argument(
        "--gate-soft",
        dest="gate_hard",
        action="store_false",
    )

    p.add_argument(
        "--gate-mode",
        default="max",
        choices=[
            "max",
            "z",
        ],
    )

    p.add_argument(
        "--gate-margin",
        type=float,
        default=1.12,
    )

    # ========================================================
    # Only B-specific parameters
    # ========================================================

    p.add_argument(
        "--robust-tau",
        type=float,
        default=3.0,
    )

    p.add_argument(
        "--robust-temperature",
        type=float,
        default=1.0,
    )

    p.add_argument(
        "--mad-eps",
        type=float,
        default=1e-6,
    )

    return p.parse_args()


@torch.no_grad()
def predict_views(
    model,
    images,
    image_size,
    tta_flip=False,
    use_amp=True,
):
    from src.dist_utils import make_autocast

    with make_autocast(
        use_amp
    ):
        en, de, _ = model(
            images
        )

        amap = anomaly_map_from_features(
            en,
            de,
            out_size=image_size,
        )

        if tta_flip:
            flipped = torch.flip(
                images,
                dims=[-1],
            )

            en_f, de_f, _ = model(
                flipped
            )

            amap_f = anomaly_map_from_features(
                en_f,
                de_f,
                out_size=image_size,
            )

            amap_f = torch.flip(
                amap_f,
                dims=[-1],
            )

            amap = (
                0.5
                * (
                    amap
                    + amap_f
                )
            )

    return (
        amap[:, 0]
        .float()
        .cpu()
        .numpy()
    )


def baseline_topk_score(
    amap,
    max_ratio,
):
    flat = np.asarray(
        amap
    ).reshape(-1)

    k = max(
        1,
        int(
            flat.size
            * max_ratio
        ),
    )

    k = min(
        k,
        flat.size,
    )

    return float(
        np.partition(
            flat,
            -k,
        )[-k:].mean()
    )


def robust_mad_pixel_map(
    maps: np.ndarray,
    tau: float = 3.0,
    temperature: float = 1.0,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    maps:
        [V,H,W]

    Robust normalization is PER VIEW.

    This is the ONLY experimental pixel transform.
    """

    maps = np.asarray(
        maps,
        dtype=np.float32,
    )

    output = np.empty_like(
        maps,
        dtype=np.float32,
    )

    temperature = max(
        float(
            temperature
        ),
        1e-6,
    )

    for v in range(
        maps.shape[0]
    ):
        amap = maps[
            v
        ]

        flat = amap.reshape(
            -1
        )

        median = float(
            np.median(
                flat
            )
        )

        mad = float(
            np.median(
                np.abs(
                    flat
                    - median
                )
            )
        )

        scale = max(
            1.4826 * mad,
            eps,
        )

        z = (
            amap
            - median
        ) / scale

        logits = (
            z
            - tau
        ) / temperature

        # Numerical stability.
        logits = np.clip(
            logits,
            -30.0,
            30.0,
        )

        output[
            v
        ] = (
            1.0
            / (
                1.0
                + np.exp(
                    -logits
                )
            )
        )

    return output


def auto_scale(
    model,
    args,
    device,
):
    if not args.train_root:
        print(
            "[scale] no --train-root, "
            "fallback scale=2.0"
        )
        return 2.0

    ds = CSIGImageDataset(
        args.train_root,
        transform=build_transform(
            args.image_size,
            is_train=False,
        ),
        image_size=args.image_size,
    )

    n = min(
        args.calibrate_n,
        len(ds),
    )

    idx = np.linspace(
        0,
        len(ds) - 1,
        n,
    ).astype(int)

    maps = []
    batch = []

    for i in tqdm(
        idx,
        desc="calibrate",
        ncols=80,
    ):
        img, _ = ds[
            int(i)
        ]

        batch.append(
            img
        )

        if (
            len(batch)
            == max(
                1,
                args.samples_per_batch
                * 5,
            )
        ):
            x = torch.stack(
                batch,
                0,
            ).to(device)

            maps.append(
                predict_views(
                    model,
                    x,
                    args.image_size,
                    False,
                    args.amp,
                )
            )

            batch = []

    if batch:
        x = torch.stack(
            batch,
            0,
        ).to(device)

        maps.append(
            predict_views(
                model,
                x,
                args.image_size,
                False,
                args.amp,
            )
        )

    scale = calibrate_scale(
        maps,
        target_q=0.995,
        target_value=0.55,
    )

    print(
        f"[scale] auto mask_scale="
        f"{scale:.4f}"
    )

    return scale


def collect_normal_stats(
    model,
    args,
    device,
):
    from collections import defaultdict

    ds = CSIGSampleDataset(
        args.train_root,
        transform=build_transform(
            args.image_size,
            False,
        ),
        image_size=args.image_size,
    )

    by_cat = defaultdict(
        list
    )

    for i, (
        cat,
        _sid,
        _p,
    ) in enumerate(
        ds.samples
    ):
        by_cat[
            cat
        ].append(
            i
        )

    stats = NormalStats(
        hw=64
    )

    for cat, idxs in tqdm(
        sorted(
            by_cat.items()
        ),
        desc="normal-stats",
        ncols=80,
    ):
        step = max(
            1,
            len(idxs)
            // max(
                1,
                args.stats_per_class,
            ),
        )

        for i in idxs[
            ::step
        ][
            :args.stats_per_class
        ]:
            item = ds[
                i
            ]

            x = item[
                "images"
            ].to(
                device
            )

            maps = predict_views(
                model,
                x,
                args.image_size,
                False,
                args.amp,
            )

            maps = smooth_map(
                maps,
                sigma=args.sigma,
            )

            for v in range(
                maps.shape[0]
            ):
                sc = baseline_topk_score(
                    maps[v],
                    args.max_ratio,
                )

                stats.update(
                    cat,
                    v,
                    maps[v],
                    view_score=sc,
                )

    return stats


def main():
    args = parse_args()

    os.environ.setdefault(
        "NCCL_P2P_DISABLE",
        "1",
    )

    os.environ.setdefault(
        "NCCL_IB_DISABLE",
        "1",
    )

    enc_name = (
        args.encoder
        or "dinov2reg_vit_base_14"
    )

    if Path(
        args.ckpt
    ).is_file():
        peek = torch_load(
            args.ckpt,
            map_location="cpu",
        )

        peek_args = (
            peek.get(
                "args",
                {},
            )
            if isinstance(
                peek,
                dict,
            )
            else {}
        )

        enc_name = (
            args.encoder
            or peek_args.get(
                "encoder",
                enc_name,
            )
        )

        del peek

    prefetch_encoder_weights(
        enc_name,
        args.encoder_source,
        stamp_dir=str(
            Path(
                args.ckpt
            ).resolve().parent
            / "_prefetch"
        ),
    )

    info = init_distributed()

    setup_seed(
        0,
        rank=info.rank,
    )

    device = (
        info.device
    )

    model, load_info = (
        load_inpformer(
            args.ckpt,
            device,
            encoder=(
                args.encoder
                or None
            ),
            encoder_source=(
                args.encoder_source
            ),
            inp_num=args.inp_num,
            decoder_depth=(
                args.decoder_depth
            ),
            bottleneck_drop=(
                args.bottleneck_drop
            ),
            residual=(
                args.residual
            ),
        )
    )

    image_size = int(
        load_info[
            "args"
        ].get(
            "image_size"
        )
        or args.image_size
    )

    args.image_size = (
        image_size
    )

    if is_main(info):
        print(
            f"[ckpt] loaded "
            f"{args.ckpt}"
        )

        print(
            "[experiment] B: "
            "Image=baseline, "
            "Pixel=robust MAD"
        )

    # ========================================================
    # Baseline calibration unchanged
    # ========================================================

    need_stats = bool(
        args.train_root
    ) and (
        args.refine
        or args.view_gate
    )

    stats = None

    ckpt_tag = Path(
        args.ckpt
    ).stem

    default_stats = (
        Path(
            args.out_dir
        ).parent
        / (
            f"normal_stats_"
            f"{ckpt_tag}_"
            f"s{args.sigma}.npz"
        )
    )

    stats_path = (
        Path(
            args.stats_path
        )
        if args.stats_path
        else default_stats
    )

    scale = (
        float(
            args.mask_scale
        )
        if args.mask_scale > 0
        else 0.0
    )

    ok = torch.tensor(
        [1],
        device=device,
        dtype=torch.int32,
    )

    if is_main(info):
        try:
            if need_stats:
                reuse = False

                if stats_path.is_file():
                    stats = NormalStats.load(
                        str(
                            stats_path
                        )
                    )

                    reuse = (
                        not args.view_gate
                    ) or bool(
                        stats.view_scores
                    )

                if reuse:
                    print(
                        f"[stats] loaded "
                        f"{stats_path}"
                    )
                else:
                    stats = collect_normal_stats(
                        model,
                        args,
                        device,
                    )

                    stats.save(
                        str(
                            stats_path
                        )
                    )

            # Keep baseline scale logic.
            if scale <= 0:
                if args.refine:
                    scale = (
                        0.55
                        / max(
                            3.0
                            ** float(
                                args.gamma
                            ),
                            1e-3,
                        )
                    )
                else:
                    scale = auto_scale(
                        model,
                        args,
                        device,
                    )

        except Exception as exc:
            ok[0] = 0
            print(
                f"[calib] failed: {exc}"
            )

    if info.distributed:
        torch.distributed.all_reduce(
            ok,
            op=torch.distributed.ReduceOp.MIN,
        )

        if int(
            ok.item()
        ) == 0:
            raise RuntimeError(
                "Calibration failed"
            )

        buf = torch.tensor(
            [scale],
            device=device,
            dtype=torch.float32,
        )

        torch.distributed.broadcast(
            buf,
            src=0,
        )

        scale = float(
            buf.item()
        )

        barrier()

    if (
        need_stats
        and stats is None
        and stats_path.is_file()
    ):
        stats = NormalStats.load(
            str(
                stats_path
            )
        )

    ds = CSIGSampleDataset(
        args.test_root,
        transform=build_transform(
            image_size,
            False,
        ),
        image_size=image_size,
    )

    sampler = (
        DistributedSampler(
            ds,
            num_replicas=(
                info.world_size
            ),
            rank=(
                info.rank
            ),
            shuffle=False,
            drop_last=False,
        )
        if info.distributed
        else None
    )

    loader = DataLoader(
        ds,
        batch_size=(
            args.samples_per_batch
        ),
        sampler=sampler,
        shuffle=False,
        num_workers=(
            args.num_workers
        ),
        pin_memory=(
            device.type
            == "cuda"
        ),
    )

    out = Path(
        args.out_dir
    )

    if is_main(info):
        if out.exists():
            shutil.rmtree(
                out
            )

        (
            out
            / "predicted_masks"
        ).mkdir(
            parents=True,
            exist_ok=True,
        )

    barrier()

    (
        out
        / "predicted_masks"
    ).mkdir(
        parents=True,
        exist_ok=True,
    )

    local_rows = []

    n_views_total = 0
    n_views_zeroed = 0

    model.eval()

    iterator = (
        tqdm(
            loader,
            ncols=100,
            desc=f"infer-B-r{info.rank}",
        )
        if is_main(info)
        else loader
    )

    for batch in iterator:
        images = batch[
            "images"
        ].to(
            device,
            non_blocking=True,
        )

        b, v = (
            images.shape[:2]
        )

        stacked = images.reshape(
            b * v,
            *images.shape[2:],
        )

        maps = predict_views(
            model,
            stacked,
            args.image_size,
            tta_flip=(
                args.tta_flip
            ),
            use_amp=(
                args.amp
            ),
        )

        maps = maps.reshape(
            b,
            v,
            maps.shape[-2],
            maps.shape[-1],
        )

        folders = batch[
            "group_folder"
        ]

        cats = batch[
            "category"
        ]

        for i in range(
            b
        ):
            cat = cats[
                i
            ]

            # =================================================
            # BASELINE IMAGE SCORE - UNCHANGED
            # =================================================

            raw_s = smooth_map(
                maps[
                    i
                ],
                sigma=args.sigma,
            )

            raw_scores = [
                baseline_topk_score(
                    raw_s[
                        vv
                    ],
                    args.max_ratio,
                )
                for vv in range(
                    v
                )
            ]

            img_score = squash_score(
                float(
                    np.max(
                        raw_scores
                    )
                )
            )

            # =================================================
            # THE ONLY EXPERIMENT:
            # Pixel map -> robust MAD
            # =================================================

            robust_maps = (
                robust_mad_pixel_map(
                    raw_s,
                    tau=(
                        args.robust_tau
                    ),
                    temperature=(
                        args.robust_temperature
                    ),
                    eps=(
                        args.mad_eps
                    ),
                )
            )

            # -------------------------------------------------
            # IMPORTANT
            #
            # Since robust_maps are already the pixel maps,
            # applying old apply_view_refine() would constitute
            # a second experimental change and it expects raw
            # anomaly-map statistics.
            #
            # Therefore B replaces exactly the baseline pixel
            # representation AFTER smooth_map and BEFORE output.
            # -------------------------------------------------

            view_maps = robust_maps

            # Keep optional view gate behavior if explicitly enabled.
            if (
                args.view_gate
                and stats is not None
            ):
                (
                    view_maps,
                    gates,
                    _,
                ) = apply_view_gate(
                    view_maps,
                    cat,
                    stats,
                    max_ratio=args.max_ratio,
                    k=args.gate_k,
                    temp=args.gate_temp,
                    floor=args.gate_floor,
                    scores=raw_scores,
                    hard=args.gate_hard,
                    mode=args.gate_mode,
                    margin=args.gate_margin,
                )

                n_views_total += len(
                    gates
                )

                n_views_zeroed += sum(
                    1
                    for g in gates
                    if g <= 0
                )

            # Robust maps are already [0,1].
            # Additional baseline mask_scale would change their
            # designed calibration. Use scale=1.
            masks_u8 = maps_to_uint8(
                view_maps,
                scale=1.0,
            )

            gf = folders[
                i
            ]

            dest = (
                out
                / "predicted_masks"
                / gf
            )

            dest.mkdir(
                parents=True,
                exist_ok=True,
            )

            for vi in range(
                v
            ):
                Image.fromarray(
                    masks_u8[
                        vi
                    ],
                    mode="L",
                ).save(
                    dest
                    / f"{vi}_mask.png"
                )

            local_rows.append(
                (
                    gf,
                    float(
                        img_score
                    ),
                )
            )

    gathered = (
        all_gather_object(
            local_rows
        )
    )

    gathered_z = (
        all_gather_object(
            (
                n_views_zeroed,
                n_views_total,
            )
        )
    )

    if is_main(info):
        rows = []
        seen = set()

        for part in gathered:
            for gf, score in part:
                if gf in seen:
                    continue

                seen.add(
                    gf
                )

                rows.append(
                    (
                        gf,
                        score,
                    )
                )

        rows.sort(
            key=lambda x: x[0]
        )

        csv_path = (
            out
            / "submission.csv"
        )

        vals = []

        with open(
            csv_path,
            "w",
            encoding="utf-8-sig",
            newline="",
        ) as fcsv:
            writer = csv.writer(
                fcsv
            )

            writer.writerow([
                "group_folder",
                "anomaly_score",
            ])

            for gf, score in rows:
                # Preserve baseline second squash.
                s = squash_score(
                    score
                )

                vals.append(
                    s
                )

                writer.writerow([
                    gf,
                    f"{s:.8f}",
                ])

        if vals:
            print(
                "[csv] anomaly_score "
                f"min={min(vals):.6f} "
                f"max={max(vals):.6f}"
            )

        meta = {
            "experiment":
                "B_robust_pixel_only",

            "image_score":
                "baseline",

            "pixel_map":
                "sigmoid((A-median)/(1.4826*MAD)-tau)",

            "robust_tau":
                args.robust_tau,

            "robust_temperature":
                args.robust_temperature,

            "test_root":
                args.test_root,

            "ckpt":
                args.ckpt,

            "sigma":
                args.sigma,

            "max_ratio":
                args.max_ratio,

            "tta_flip":
                args.tta_flip,

            "n_samples":
                len(rows),

            "world_size":
                info.world_size,

            "arch":
                load_info[
                    "arch"
                ],
        }

        (
            out
            / "infer_meta.json"
        ).write_text(
            json.dumps(
                meta,
                indent=2,
            ),
            encoding="utf-8",
        )

        zip_path = zip_submission(
            str(out),
            args.zip,
        )

        print(
            f"[done] Test B: "
            f"{zip_path}"
        )

    barrier()


if __name__ == "__main__":
    try:
        main()
    finally:
        cleanup()
