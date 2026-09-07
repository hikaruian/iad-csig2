from __future__ import annotations

import argparse
import json
import os
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from torch.nn.parallel import (
    DistributedDataParallel as DDP,
)

from torch.utils.data import (
    DataLoader,
    DistributedSampler,
)

from tqdm import tqdm

from src.data import CSIGImageDataset

from src.dist_utils import (
    barrier,
    cleanup,
    init_distributed,
    is_main,
    make_autocast,
    make_scaler,
    reduce_mean,
    setup_seed,
    unwrap,
)

from src.encoder import (
    prefetch_encoder_weights,
)

from src.losses import (
    pixel_loss,
    total_loss,
)

from src.model import build_model

from src.optim import (
    StableAdamW,
    WarmCosineScheduler,
)


# ============================================================
# Arguments
# ============================================================

def parse_args():

    p = argparse.ArgumentParser()

    p.add_argument(
        "--train-root",
        required=True,
    )

    p.add_argument(
        "--save-dir",
        required=True,
    )

    # Training categories for a CV fold.
    # Empty = use all categories.
    p.add_argument(
        "--category-file",
        default="",
    )

    # Holdout categories used only for online proxy evaluation.
    p.add_argument(
        "--holdout-category-file",
        default="",
    )

    # Directory into which proxy_XXXX.json is written.
    # Empty = disable proxy evaluation.
    p.add_argument(
        "--proxy-output-dir",
        default="",
    )

    # Evaluate every N epochs.
    p.add_argument(
        "--proxy-eval-every",
        type=int,
        default=10,
    )

    p.add_argument(
        "--encoder",
        default="dinov2reg_vit_large_14",
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
        default=12,
    )

    p.add_argument(
        "--decoder-depth",
        type=int,
        default=8,
    )

    p.add_argument(
        "--residual-strength",
        type=float,
        default=0.20,
    )

    p.add_argument(
        "--epochs",
        type=int,
        default=100,
    )

    # Kept for compatibility with the original train.py.
    p.add_argument(
        "--save-every",
        type=int,
        default=10,
    )

    # Disabled by default to avoid consuming the 18 GB disk.
    p.add_argument(
        "--keep-epoch-checkpoints",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    p.add_argument(
        "--batch-size",
        type=int,
        default=1,
    )

    p.add_argument(
        "--grad-accum",
        type=int,
        default=8,
    )

    p.add_argument(
        "--lr",
        type=float,
        default=5e-4,
    )

    p.add_argument(
        "--min-lr",
        type=float,
        default=1e-6,
    )

    p.add_argument(
        "--weight-decay",
        type=float,
        default=1e-5,
    )

    p.add_argument(
        "--gather-weight",
        type=float,
        default=0.1,
    )

    p.add_argument(
        "--soft-y",
        type=float,
        default=2.0,
    )

    p.add_argument(
        "--synthetic-prob",
        type=float,
        default=0.8,
    )

    p.add_argument(
        "--pixel-warmup",
        type=int,
        default=10,
    )

    p.add_argument(
        "--num-workers",
        type=int,
        default=2,
    )

    p.add_argument(
        "--seed",
        type=int,
        default=1,
    )

    p.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    # Stop cleanly before the platform SIGKILL.
    # 0 = unlimited.
    p.add_argument(
        "--job-seconds",
        type=int,
        default=0,
    )

    return p.parse_args()


# ============================================================
# File helpers
# ============================================================

def atomic_save(
    state,
    path,
):

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = Path(
        str(path) + ".tmp"
    )

    torch.save(
        state,
        tmp,
    )

    os.replace(
        tmp,
        path,
    )


def atomic_write_json(
    obj,
    path,
):

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = Path(
        str(path) + ".tmp"
    )

    tmp.write_text(
        json.dumps(
            obj,
            indent=2,
        ),
        encoding="utf-8",
    )

    os.replace(
        tmp,
        path,
    )


def load_category_file(
    path,
):

    if not path:
        return None

    path = Path(path)

    data = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(
        data,
        list,
    ):

        raise ValueError(
            f"Category file must contain "
            f"a JSON list: {path}"
        )

    return set(data)


# ============================================================
# Proxy evaluation
#
# This is the logic previously contained in proxy_eval.py.
# ============================================================

def score_batch(
    amap,
):

    flat = amap.flatten(1)

    k = max(
        1,
        int(
            flat.shape[1]
            * 0.01
        ),
    )

    return torch.topk(
        flat,
        k,
        dim=1,
    ).values.mean(1)


@torch.no_grad()
def collect_clean(
    model,
    loader,
    device,
    amp,
):

    scores = []

    for images, _ in loader:

        images = images.to(
            device,
            non_blocking=True,
        )

        with make_autocast(
            amp
        ):

            output = model(
                images,
                return_maps=True,
            )

        amap = (
            output[
                "pixel_map"
            ]
            .float()
        )

        scores.extend(
            score_batch(
                amap
            )
            .cpu()
            .tolist()
        )

    return np.asarray(
        scores,
        dtype=np.float64,
    )


@torch.no_grad()
def collect_synthetic(
    model,
    loader,
    device,
    amp,
):

    image_scores = []
    f1_scores = []

    thresholds = torch.linspace(
        0.05,
        0.95,
        19,
        device=device,
        dtype=torch.float32,
    )

    for batch in loader:

        images = (
            batch[
                "synthetic"
            ]
            .to(
                device,
                non_blocking=True,
            )
        )

        target = (
            batch[
                "mask"
            ]
            .to(
                device,
                non_blocking=True,
            )
            .float()
        )

        with make_autocast(
            amp
        ):

            output = model(
                images,
                return_maps=True,
            )

        # Metric computation is deliberately
        # performed in FP32.
        amap = (
            output[
                "pixel_map"
            ]
            .float()
        )

        image_scores.extend(
            score_batch(
                amap
            )
            .cpu()
            .tolist()
        )

        best = torch.zeros(
            amap.shape[0],
            device=device,
            dtype=torch.float32,
        )

        for threshold in thresholds:

            pred = (
                amap
                >= threshold
            ).float()

            tp = (
                pred
                * target
            ).flatten(1).sum(1)

            fp = (
                pred
                * (
                    1.0
                    - target
                )
            ).flatten(1).sum(1)

            fn = (
                (
                    1.0
                    - pred
                )
                * target
            ).flatten(1).sum(1)

            f1 = (
                2.0 * tp
                / (
                    2.0 * tp
                    + fp
                    + fn
                    + 1e-6
                )
            )

            best = torch.maximum(
                best,
                f1,
            )

        f1_scores.extend(
            best.cpu().tolist()
        )

    return (
        np.asarray(
            image_scores,
            dtype=np.float64,
        ),
        np.asarray(
            f1_scores,
            dtype=np.float64,
        ),
    )


def run_proxy_eval(
    model,
    root,
    image_size,
    train_categories,
    holdout_categories,
    batch_size,
    num_workers,
    device,
    amp,
    synthetic_seed=913751,
):

    if not train_categories:

        raise RuntimeError(
            "Proxy evaluation requires "
            "non-empty training categories."
        )

    if not holdout_categories:

        raise RuntimeError(
            "Proxy evaluation requires "
            "non-empty holdout categories."
        )

    overlap = (
        train_categories
        & holdout_categories
    )

    if overlap:

        raise RuntimeError(
            "Training and holdout "
            "categories overlap: "
            f"{sorted(overlap)}"
        )

    train_clean = CSIGImageDataset(
        root,
        image_size=image_size,
        synthetic_anomaly=False,
        include_categories=(
            train_categories
        ),
    )

    holdout_clean = CSIGImageDataset(
        root,
        image_size=image_size,
        synthetic_anomaly=False,
        include_categories=(
            holdout_categories
        ),
    )

    holdout_syn = CSIGImageDataset(
        root,
        image_size=image_size,
        synthetic_anomaly=True,
        synthetic_prob=1.0,
        include_categories=(
            holdout_categories
        ),
        deterministic_synthetic=True,
        synthetic_seed=(
            synthetic_seed
        ),
    )

    if len(train_clean) == 0:

        raise RuntimeError(
            "Proxy train-clean "
            "dataset is empty."
        )

    if len(holdout_clean) == 0:

        raise RuntimeError(
            "Proxy holdout-clean "
            "dataset is empty."
        )

    if len(holdout_syn) == 0:

        raise RuntimeError(
            "Proxy holdout-synthetic "
            "dataset is empty."
        )

    workers = max(
        0,
        num_workers,
    )

    loader_kwargs = {
        "batch_size":
            max(
                1,
                batch_size,
            ),

        "shuffle":
            False,

        "num_workers":
            workers,

        "pin_memory":
            (
                device.type
                == "cuda"
            ),

        "persistent_workers":
            (
                workers > 0
            ),
    }

    train_loader = DataLoader(
        train_clean,
        **loader_kwargs,
    )

    holdout_loader = DataLoader(
        holdout_clean,
        **loader_kwargs,
    )

    synthetic_loader = DataLoader(
        holdout_syn,
        **loader_kwargs,
    )

    was_training = (
        model.training
    )

    model.eval()

    try:

        train_scores = collect_clean(
            model,
            train_loader,
            device,
            amp,
        )

        holdout_scores = (
            collect_clean(
                model,
                holdout_loader,
                device,
                amp,
            )
        )

        (
            synthetic_scores,
            f1,
        ) = collect_synthetic(
            model,
            synthetic_loader,
            device,
            amp,
        )

    finally:

        if was_training:
            model.train()

    if train_scores.size == 0:

        raise RuntimeError(
            "No train-clean proxy "
            "scores were produced."
        )

    if holdout_scores.size == 0:

        raise RuntimeError(
            "No holdout-clean proxy "
            "scores were produced."
        )

    if synthetic_scores.size == 0:

        raise RuntimeError(
            "No synthetic proxy "
            "scores were produced."
        )

    if f1.size == 0:

        raise RuntimeError(
            "No pixel F1 proxy "
            "scores were produced."
        )

    train_normal = float(
        np.mean(
            train_scores
        )
    )

    holdout_normal = float(
        np.mean(
            holdout_scores
        )
    )

    synthetic_mean = float(
        np.mean(
            synthetic_scores
        )
    )

    normal_gap = (
        holdout_normal
        / max(
            train_normal,
            1e-8,
        )
    )

    sensitivity = (
        synthetic_mean
        / max(
            holdout_normal,
            1e-8,
        )
    )

    return {
        "train_normal":
            train_normal,

        "holdout_normal":
            holdout_normal,

        "normal_gap":
            normal_gap,

        "synthetic":
            synthetic_mean,

        # Keep exactly the same field name
        # and definition as proxy_eval.py.
        "separation":
            sensitivity,

        "pixel_f1":
            float(
                np.mean(
                    f1
                )
            ),
    }


# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()

    # --------------------------------------------------------
    # Validate arguments before allocating the model.
    # --------------------------------------------------------

    if args.epochs <= 0:

        raise ValueError(
            "--epochs must be > 0"
        )

    if args.batch_size <= 0:

        raise ValueError(
            "--batch-size must be > 0"
        )

    if args.grad_accum <= 0:

        raise ValueError(
            "--grad-accum must be > 0"
        )

    if args.num_workers < 0:

        raise ValueError(
            "--num-workers must be >= 0"
        )

    if (
        args.proxy_output_dir
        and args.proxy_eval_every
        <= 0
    ):

        raise ValueError(
            "--proxy-eval-every "
            "must be > 0"
        )

    if (
        args.keep_epoch_checkpoints
        and args.save_every <= 0
    ):

        raise ValueError(
            "--save-every must be > 0"
        )

    job_start = time.time()

    # --------------------------------------------------------
    # Encoder prefetch.
    # --------------------------------------------------------

    prefetch_encoder_weights(
        args.encoder,
        args.encoder_source,
        str(
            Path(
                args.save_dir
            )
            / "_prefetch"
        ),
    )

    # --------------------------------------------------------
    # Distributed setup.
    # --------------------------------------------------------

    info = init_distributed()

    setup_seed(
        args.seed,
        rank=info.rank,
    )

    device = info.device

    save_dir = Path(
        args.save_dir
    )

    if is_main(info):

        save_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    barrier()

    # --------------------------------------------------------
    # Categories.
    # --------------------------------------------------------

    include = load_category_file(
        args.category_file
    )

    holdout = load_category_file(
        args.holdout_category_file
    )

    proxy_enabled = bool(
        args.proxy_output_dir
    )

    if proxy_enabled:

        if not include:

            raise RuntimeError(
                "--proxy-output-dir "
                "requires a non-empty "
                "--category-file"
            )

        if not holdout:

            raise RuntimeError(
                "--proxy-output-dir "
                "requires a non-empty "
                "--holdout-category-file"
            )

        overlap = (
            include
            & holdout
        )

        if overlap:

            raise RuntimeError(
                "Training and holdout "
                "categories overlap: "
                f"{sorted(overlap)}"
            )

    # --------------------------------------------------------
    # Training dataset.
    # --------------------------------------------------------

    dataset = CSIGImageDataset(
        args.train_root,
        image_size=(
            args.image_size
        ),
        synthetic_anomaly=True,
        synthetic_prob=(
            args.synthetic_prob
        ),
        include_categories=(
            include
        ),
    )

    if len(dataset) == 0:

        raise RuntimeError(
            "Training dataset is empty."
        )

    sampler = (
        DistributedSampler(
            dataset,
            shuffle=True,
            drop_last=True,
        )
        if info.distributed
        else None
    )

    loader = DataLoader(
        dataset,
        batch_size=(
            args.batch_size
        ),
        sampler=sampler,
        shuffle=(
            sampler is None
        ),
        num_workers=(
            args.num_workers
        ),
        pin_memory=(
            device.type
            == "cuda"
        ),
        drop_last=True,
        persistent_workers=(
            args.num_workers
            > 0
        ),
    )

    if not len(loader):

        raise RuntimeError(
            "Empty DataLoader. "
            "Dataset may be smaller "
            "than batch size per rank."
        )

    # --------------------------------------------------------
    # Model.
    # --------------------------------------------------------

    model = build_model(
        encoder_name=(
            args.encoder
        ),
        inp_num=args.inp_num,
        decoder_depth=(
            args.decoder_depth
        ),
        residual_strength=(
            args.residual_strength
        ),
        encoder_source=(
            args.encoder_source
        ),
    ).to(device)

    raw = model

    parameters = list(
        raw.trainable_parameters()
    )

    if not parameters:

        raise RuntimeError(
            "Model has no trainable "
            "parameters."
        )

    # --------------------------------------------------------
    # Optimizer.
    # --------------------------------------------------------

    accumulation = max(
        1,
        args.grad_accum,
    )

    optimizer_steps = (
        len(loader)
        + accumulation
        - 1
    ) // accumulation

    optimizer = StableAdamW(
        [
            {
                "params":
                    parameters
            }
        ],
        lr=args.lr,
        betas=(
            0.9,
            0.999,
        ),
        weight_decay=(
            args.weight_decay
        ),
        amsgrad=True,
        eps=1e-10,
    )

    scheduler = WarmCosineScheduler(
        optimizer,
        base_value=(
            args.lr
        ),
        final_value=(
            args.min_lr
        ),
        total_iters=(
            args.epochs
            * optimizer_steps
        ),
        warmup_iters=min(
            100,
            max(
                10,
                optimizer_steps,
            ),
        ),
    )

    scaler = make_scaler(
        args.amp
    )

    # --------------------------------------------------------
    # Automatic resume.
    #
    # Must happen before wrapping with DDP.
    # --------------------------------------------------------

    start_epoch = 0

    last_path = (
        save_dir
        / "last.pth"
    )

    if last_path.is_file():

        checkpoint = torch.load(
            last_path,
            map_location="cpu",
        )

        if "model" not in checkpoint:

            raise RuntimeError(
                "Invalid checkpoint: "
                f"{last_path}"
            )

        raw.load_state_dict(
            checkpoint["model"],
            strict=True,
        )

        if "optimizer" in checkpoint:

            optimizer.load_state_dict(
                checkpoint[
                    "optimizer"
                ]
            )

        if "scheduler" in checkpoint:

            scheduler.load_state_dict(
                checkpoint[
                    "scheduler"
                ]
            )

        if "scaler" in checkpoint:

            scaler.load_state_dict(
                checkpoint[
                    "scaler"
                ]
            )

        start_epoch = int(
            checkpoint.get(
                "epoch",
                0,
            )
        )

        if start_epoch < 0:

            raise RuntimeError(
                "Invalid checkpoint epoch."
            )

        if is_main(info):

            print(
                f"[resume] epoch "
                f"{start_epoch}"
            )

        del checkpoint

    # --------------------------------------------------------
    # DDP.
    # --------------------------------------------------------

    if info.distributed:

        model = DDP(
            model,
            device_ids=[
                info.local_rank
            ],
            output_device=(
                info.local_rank
            ),
            broadcast_buffers=False,
            find_unused_parameters=True,
        )

    raw = unwrap(
        model
    )

    final_epoch = (
        start_epoch
    )

    # ========================================================
    # Training loop.
    # ========================================================

    for epoch in range(
        start_epoch,
        args.epochs,
    ):

        model.train()

        # Encoder stays frozen/eval as in the
        # original train.py.
        raw.encoder.eval()

        if sampler is not None:

            sampler.set_epoch(
                epoch
            )

        pixel_weight = min(
            1.0,
            (epoch + 1)
            / max(
                1,
                args.pixel_warmup,
            ),
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        running = []

        iterator = (
            tqdm(
                loader,
                desc=(
                    f"{epoch + 1}/"
                    f"{args.epochs}"
                ),
            )
            if is_main(info)
            else loader
        )

        for step, batch in enumerate(
            iterator
        ):

            clean = (
                batch["clean"]
                .to(
                    device,
                    non_blocking=True,
                )
            )

            synthetic = (
                batch[
                    "synthetic"
                ]
                .to(
                    device,
                    non_blocking=True,
                )
            )

            mask = (
                batch["mask"]
                .to(
                    device,
                    non_blocking=True,
                )
            )

            do_step = (
                (
                    (step + 1)
                    % accumulation
                    == 0
                )
                or (
                    step + 1
                    == len(loader)
                )
            )

            sync = (
                model.no_sync()
                if (
                    info.distributed
                    and not do_step
                )
                else nullcontext()
            )

            with sync:

                # --------------------------------------------
                # Clean branch.
                # --------------------------------------------

                with make_autocast(
                    args.amp
                ):

                    clean_output = model(
                        clean,
                        return_maps=True,
                    )

                    loss_clean = total_loss(
                        clean_output[
                            "en"
                        ],
                        clean_output[
                            "de"
                        ],
                        clean_output[
                            "g_loss"
                        ],
                        gather_weight=(
                            args.gather_weight
                        ),
                        y=args.soft_y,
                    )

                scaler.scale(
                    loss_clean
                    / accumulation
                ).backward()

                del clean_output

                # --------------------------------------------
                # Synthetic branch.
                # --------------------------------------------

                with make_autocast(
                    args.amp
                ):

                    syn_output = model(
                        synthetic,
                        return_maps=True,
                    )

                    ploss = pixel_loss(
                        syn_output[
                            "pixel_map"
                        ],
                        syn_output[
                            "pixel_delta"
                        ],
                        mask,
                    )

                    loss_pixel = (
                        pixel_weight
                        * ploss[
                            "total"
                        ]
                    )

                scaler.scale(
                    loss_pixel
                    / accumulation
                ).backward()

                del syn_output
                del ploss

            if do_step:

                scaler.unscale_(
                    optimizer
                )

                nn.utils.clip_grad_norm_(
                    parameters,
                    0.5,
                )

                scaler.step(
                    optimizer
                )

                scaler.update()

                optimizer.zero_grad(
                    set_to_none=True
                )

                scheduler.step()

            reduced = reduce_mean(
                loss_clean.detach()
                + loss_pixel.detach()
            )

            running.append(
                float(
                    reduced.cpu()
                )
            )

            del loss_clean
            del loss_pixel
            del reduced

        if not running:

            raise RuntimeError(
                "No training steps "
                "were executed."
            )

        mean_loss = float(
            np.mean(
                running
            )
        )

        final_epoch = (
            epoch + 1
        )

        # ====================================================
        # Save rolling checkpoint.
        # ====================================================

        if is_main(info):

            state = {
                "epoch":
                    final_epoch,

                "model":
                    raw.state_dict(),

                "optimizer":
                    optimizer.state_dict(),

                "scheduler":
                    scheduler.state_dict(),

                "scaler":
                    scaler.state_dict(),

                "args":
                    vars(args),

                "loss":
                    mean_loss,
            }

            # Always overwrite one resume checkpoint.
            atomic_save(
                state,
                last_path,
            )

            # Historical checkpoints are optional and
            # disabled by default.
            if (
                args.keep_epoch_checkpoints
                and (
                    (
                        final_epoch
                        % args.save_every
                        == 0
                    )
                    or (
                        final_epoch
                        == args.epochs
                    )
                )
            ):

                atomic_save(
                    state,
                    save_dir
                    / (
                        f"epoch_"
                        f"{final_epoch:04d}"
                        f".pth"
                    ),
                )

            del state

            atomic_write_json(
                {
                    "epoch":
                        final_epoch,

                    "target_epochs":
                        args.epochs,

                    "loss":
                        mean_loss,
                },
                save_dir
                / "progress.json",
            )

        # All ranks must finish training/checkpointing
        # before rank 0 evaluates raw independently.
        barrier()

        # ====================================================
        # Online proxy evaluation.
        #
        # Same evaluation points as the original:
        # eval_every, 2*eval_every, ...
        #
        # The final epoch is always evaluated even if
        # epochs % proxy_eval_every != 0.
        # ====================================================

        should_proxy_eval = (
            proxy_enabled
            and (
                (
                    final_epoch
                    % args.proxy_eval_every
                    == 0
                )
                or (
                    final_epoch
                    == args.epochs
                )
            )
        )

        if (
            should_proxy_eval
            and is_main(info)
        ):

            proxy_dir = Path(
                args.proxy_output_dir
            )

            proxy_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            proxy_file = (
                proxy_dir
                / (
                    f"proxy_"
                    f"{final_epoch:04d}"
                    f".json"
                )
            )

            if (
                not proxy_file.is_file()
            ):

                print(
                    "[proxy-eval] "
                    f"epoch="
                    f"{final_epoch}",
                    flush=True,
                )

                if (
                    device.type
                    == "cuda"
                ):

                    torch.cuda.empty_cache()

                result = run_proxy_eval(
                    model=raw,
                    root=(
                        args.train_root
                    ),
                    image_size=(
                        args.image_size
                    ),
                    train_categories=(
                        include
                    ),
                    holdout_categories=(
                        holdout
                    ),
                    # Keep proxy_eval.py's original
                    # effective default of 2 where
                    # possible. A fixed value also keeps
                    # proxy scores independent of the
                    # training batch size.
                    batch_size=2,
                    num_workers=(
                        args.num_workers
                    ),
                    device=device,
                    amp=args.amp,
                    synthetic_seed=(
                        913751
                    ),
                )

                result["epoch"] = (
                    final_epoch
                )

                atomic_write_json(
                    result,
                    proxy_file,
                )

                print(
                    "[proxy-eval] "
                    + json.dumps(
                        result,
                        indent=2,
                    ),
                    flush=True,
                )

            else:

                print(
                    "[proxy-eval] "
                    f"epoch="
                    f"{final_epoch} "
                    "already exists; "
                    "skipping",
                    flush=True,
                )

        # Rank 0 performs evaluation without the DDP wrapper.
        # Other ranks wait here.
        barrier()

        # ====================================================
        # Safe exit.
        #
        # Do this AFTER proxy evaluation so an evaluation
        # point is never lost when the job exits cleanly.
        # ====================================================

        if (
            args.job_seconds > 0
            and final_epoch
            < args.epochs
        ):

            elapsed = (
                time.time()
                - job_start
            )

            stop = torch.tensor(
                int(
                    elapsed
                    >= args.job_seconds
                ),
                device=device,
                dtype=torch.int32,
            )

            if info.distributed:

                torch.distributed.all_reduce(
                    stop,
                    op=(
                        torch.distributed
                        .ReduceOp.MAX
                    ),
                )

            if stop.item():

                if is_main(info):

                    print(
                        "[safe-exit] "
                        f"epoch="
                        f"{final_epoch}",
                        flush=True,
                    )

                break

    # ========================================================
    # Handle resume where training had already reached the
    # requested target before this invocation.
    #
    # Example:
    # last.pth = epoch 100, but proxy_0100.json was not
    # successfully written because the previous process died
    # immediately after checkpointing.
    # ========================================================

    barrier()

    training_complete = (
        final_epoch
        >= args.epochs
    )

    if (
        training_complete
        and proxy_enabled
        and is_main(info)
    ):

        proxy_dir = Path(
            args.proxy_output_dir
        )

        proxy_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        final_proxy = (
            proxy_dir
            / (
                f"proxy_"
                f"{args.epochs:04d}"
                f".json"
            )
        )

        if not final_proxy.is_file():

            print(
                "[proxy-eval] "
                "recovering missing "
                f"final epoch="
                f"{args.epochs}",
                flush=True,
            )

            if (
                device.type
                == "cuda"
            ):

                torch.cuda.empty_cache()

            result = run_proxy_eval(
                model=raw,
                root=args.train_root,
                image_size=(
                    args.image_size
                ),
                train_categories=(
                    include
                ),
                holdout_categories=(
                    holdout
                ),
                batch_size=2,
                num_workers=(
                    args.num_workers
                ),
                device=device,
                amp=args.amp,
                synthetic_seed=(
                    913751
                ),
            )

            result["epoch"] = (
                args.epochs
            )

            atomic_write_json(
                result,
                final_proxy,
            )

            print(
                "[proxy-eval] "
                + json.dumps(
                    result,
                    indent=2,
                ),
                flush=True,
            )

    barrier()

    # ========================================================
    # Completion marker.
    # ========================================================

    if (
        training_complete
        and is_main(info)
    ):

        atomic_write_json(
            {
                "epoch":
                    int(
                        final_epoch
                    ),

                "target_epochs":
                    int(
                        args.epochs
                    ),

                "complete":
                    True,
            },
            save_dir
            / "complete.json",
        )

        print(
            "[done] "
            f"epoch={final_epoch}",
            flush=True,
        )

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
