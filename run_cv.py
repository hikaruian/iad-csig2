from __future__ import annotations

import argparse
import json
import shutil
import statistics
import subprocess
import time

from pathlib import Path

import numpy as np


def parse_args():

    p = argparse.ArgumentParser()

    p.add_argument(
        "--train-root",
        required=True,
    )

    p.add_argument(
        "--save-root",
        default="runs/final_cv",
    )

    p.add_argument(
        "--folds",
        type=int,
        default=3,
    )

    p.add_argument(
        "--holdout",
        type=int,
        default=10,
    )

    p.add_argument(
        "--max-epochs",
        type=int,
        default=100,
    )

    p.add_argument(
        "--eval-every",
        type=int,
        default=10,
    )

    p.add_argument(
        "--encoder",
        default="dinov2reg_vit_large_14",
    )

    p.add_argument(
        "--nproc",
        type=int,
        default=2,
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
        "--seed",
        type=int,
        default=1,
    )

    # 11h30m by default.
    p.add_argument(
        "--job-seconds",
        type=int,
        default=41400,
    )

    p.add_argument(
        "--sensitivity-retain",
        type=float,
        default=0.95,
    )

    return p.parse_args()


def run(command):

    print(
        "\n[RUN] "
        + " ".join(command),
        flush=True,
    )

    subprocess.run(
        command,
        check=True,
    )


def categories(root):

    root = Path(root)

    if not root.is_dir():

        raise RuntimeError(
            f"Training root does not exist: "
            f"{root}"
        )

    output = sorted(
        p.name
        for p in root.iterdir()
        if p.is_dir()
    )

    if not output:

        raise RuntimeError(
            "No categories found under "
            f"{root}"
        )

    return output


def generate_folds(
    cats,
    folds,
    holdout,
    seed,
):

    if folds <= 0:

        raise ValueError(
            "--folds must be > 0"
        )

    if holdout <= 0:

        raise ValueError(
            "--holdout must be > 0"
        )

    rng = np.random.default_rng(
        seed
    )

    order = list(cats)

    rng.shuffle(order)

    if (
        folds * holdout
        > len(order)
    ):

        raise ValueError(
            "folds*holdout exceeds "
            "category count"
        )

    output = []

    for fold in range(folds):

        start = (
            fold * holdout
        )

        held = sorted(
            order[
                start:
                start + holdout
            ]
        )

        held_set = set(
            held
        )

        train = sorted(
            x
            for x in cats
            if x not in held_set
        )

        if not train:

            raise RuntimeError(
                f"Fold {fold} has no "
                "training categories."
            )

        if not held:

            raise RuntimeError(
                f"Fold {fold} has no "
                "holdout categories."
            )

        output.append(
            {
                "fold":
                    fold,

                "train":
                    train,

                "holdout":
                    held,
            }
        )

    return output


def read_last_epoch(
    checkpoint,
):

    checkpoint = Path(
        checkpoint
    )

    if not checkpoint.is_file():

        return 0

    # Avoid loading a potentially very large model checkpoint
    # into this orchestration process merely to check progress.
    #
    # train.py writes progress.json after each successfully
    # completed epoch. Therefore this function is only a
    # fallback marker check and is not normally used to
    # deserialize last.pth.
    return None


def read_progress(
    model_dir,
):

    progress_file = (
        Path(model_dir)
        / "progress.json"
    )

    if not progress_file.is_file():

        return 0

    try:

        data = json.loads(
            progress_file.read_text(
                encoding="utf-8"
            )
        )

        return int(
            data.get(
                "epoch",
                0,
            )
        )

    except (
        OSError,
        ValueError,
        TypeError,
        json.JSONDecodeError,
    ):

        return 0


def proxy_epoch(
    path,
):

    try:

        return int(
            path.stem.split(
                "_"
            )[-1]
        )

    except (
        ValueError,
        IndexError,
    ) as exc:

        raise RuntimeError(
            f"Invalid proxy filename: "
            f"{path}"
        ) from exc


def choose_checkpoint(
    records,
    retain,
):
    """
    Preserve the original CV selection rule.

    sensitivity =
        separation * pixel_f1

    First retain epochs with at least `retain`
    times the best synthetic sensitivity.

    Among those candidates minimize:

        |log(normal_gap)|

    because both unseen >> seen and unseen << seen
    indicate a distribution mismatch.
    """

    if not records:

        raise ValueError(
            "No proxy records."
        )

    for record in records:

        record["sensitivity"] = (
            record["separation"]
            * record["pixel_f1"]
        )

        record["gap_penalty"] = abs(
            np.log(
                max(
                    record[
                        "normal_gap"
                    ],
                    1e-8,
                )
            )
        )

    best_sensitivity = max(
        x["sensitivity"]
        for x in records
    )

    minimum = (
        retain
        * best_sensitivity
    )

    candidates = [
        x
        for x in records
        if (
            x["sensitivity"]
            >= minimum
        )
    ]

    return min(
        candidates,
        key=lambda x: (
            x["gap_penalty"],
            -x["sensitivity"],
        ),
    )


def expected_proxy_epochs(
    max_epochs,
    eval_every,
):

    epochs = list(
        range(
            eval_every,
            max_epochs + 1,
            eval_every,
        )
    )

    # train.py must also evaluate the final epoch even when
    # max_epochs is not divisible by eval_every.
    if (
        not epochs
        or epochs[-1]
        != max_epochs
    ):

        epochs.append(
            max_epochs
        )

    return epochs


def proxy_complete(
    fold_dir,
    max_epochs,
    eval_every,
):

    expected = expected_proxy_epochs(
        max_epochs,
        eval_every,
    )

    for epoch in expected:

        path = (
            Path(fold_dir)
            / (
                f"proxy_"
                f"{epoch:04d}.json"
            )
        )

        if not path.is_file():

            return False

    return True


def remaining_seconds(
    start,
    limit,
):

    if limit <= 0:

        # Effectively unlimited from run_cv's point of view.
        return None

    return (
        limit
        - (
            time.time()
            - start
        )
    )


def child_job_seconds(
    start,
    limit,
):

    remaining = remaining_seconds(
        start,
        limit,
    )

    if remaining is None:

        return 0

    return max(
        300,
        int(
            remaining
            - 300
        ),
    )


def insufficient_time(
    start,
    limit,
):

    remaining = remaining_seconds(
        start,
        limit,
    )

    return (
        remaining is not None
        and remaining < 600
    )


def main():

    args = parse_args()

    if args.max_epochs <= 0:

        raise ValueError(
            "--max-epochs must be > 0"
        )

    if args.eval_every <= 0:

        raise ValueError(
            "--eval-every must be > 0"
        )

    if args.nproc <= 0:

        raise ValueError(
            "--nproc must be > 0"
        )

    if args.batch_size <= 0:

        raise ValueError(
            "--batch-size must be > 0"
        )

    if args.grad_accum <= 0:

        raise ValueError(
            "--grad-accum must be > 0"
        )

    if not (
        0.0
        < args.sensitivity_retain
        <= 1.0
    ):

        raise ValueError(
            "--sensitivity-retain must "
            "be in (0, 1]"
        )

    start = time.time()

    root = Path(
        args.save_root
    )

    root.mkdir(
        parents=True,
        exist_ok=True,
    )

    cats = categories(
        args.train_root
    )

    folds = generate_folds(
        cats,
        args.folds,
        args.holdout,
        args.seed,
    )

    folds_file = (
        root
        / "folds.json"
    )

    if folds_file.exists():

        existing = json.loads(
            folds_file.read_text(
                encoding="utf-8"
            )
        )

        if existing != folds:

            raise RuntimeError(
                "Existing folds.json does "
                "not match current settings."
            )

    else:

        folds_file.write_text(
            json.dumps(
                folds,
                indent=2,
            ),
            encoding="utf-8",
        )

    # ========================================================
    # Cross validation
    # ========================================================

    for info in folds:

        fold = info[
            "fold"
        ]

        fold_dir = (
            root
            / f"fold_{fold}"
        )

        fold_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        train_file = (
            fold_dir
            / "train_categories.json"
        )

        holdout_file = (
            fold_dir
            / "holdout_categories.json"
        )

        train_file.write_text(
            json.dumps(
                info["train"],
                indent=2,
            ),
            encoding="utf-8",
        )

        holdout_file.write_text(
            json.dumps(
                info["holdout"],
                indent=2,
            ),
            encoding="utf-8",
        )

        model_dir = (
            fold_dir
            / "model"
        )

        last_checkpoint = (
            model_dir
            / "last.pth"
        )

        # ----------------------------------------------------
        # Completion is determined primarily from proxy JSON.
        #
        # The final proxy file is only generated after the
        # corresponding epoch completed, so no epoch_NNNN.pth
        # checkpoint is needed.
        # ----------------------------------------------------

        final_proxy = (
            fold_dir
            / (
                f"proxy_"
                f"{args.max_epochs:04d}"
                f".json"
            )
        )

        fold_complete = (
            final_proxy.is_file()
            and proxy_complete(
                fold_dir,
                args.max_epochs,
                args.eval_every,
            )
        )

        if fold_complete:

            print(
                f"[fold {fold}] "
                "already complete",
                flush=True,
            )

            continue

        # ----------------------------------------------------
        # Train/resume fold.
        #
        # Proxy evaluation now happens INSIDE train.py.
        # ----------------------------------------------------

        if insufficient_time(
            start,
            args.job_seconds,
        ):

            print(
                "[safe-exit] "
                "insufficient job time"
            )

            return

        run([
            "torchrun",
            "--standalone",
            (
                "--nproc_per_node="
                f"{args.nproc}"
            ),
            "train.py",

            "--train-root",
            args.train_root,

            "--save-dir",
            str(
                model_dir
            ),

            "--category-file",
            str(
                train_file
            ),

            "--holdout-category-file",
            str(
                holdout_file
            ),

            "--proxy-output-dir",
            str(
                fold_dir
            ),

            "--proxy-eval-every",
            str(
                args.eval_every
            ),

            "--encoder",
            args.encoder,

            "--epochs",
            str(
                args.max_epochs
            ),

            "--batch-size",
            str(
                args.batch_size
            ),

            "--grad-accum",
            str(
                args.grad_accum
            ),

            "--seed",
            str(
                args.seed
            ),

            "--job-seconds",
            str(
                child_job_seconds(
                    start,
                    args.job_seconds,
                )
            ),

            "--amp",

            # Important for the 18 GB limit.
            "--no-keep-epoch-checkpoints",
        ])

        # ----------------------------------------------------
        # train.py can return normally because job_seconds
        # caused a safe exit before max_epochs.
        # ----------------------------------------------------

        if not proxy_complete(
            fold_dir,
            args.max_epochs,
            args.eval_every,
        ):

            print(
                f"[fold {fold}] "
                "not yet complete; "
                "rerun identical command",
                flush=True,
            )

            return

        if not last_checkpoint.is_file():

            raise RuntimeError(
                f"Fold {fold} proxy evaluation "
                "completed but last.pth is missing."
            )

        print(
            f"[fold {fold}] complete",
            flush=True,
        )

    # ========================================================
    # Ensure all folds reached target epoch and contain all
    # expected proxy records.
    # ========================================================

    for info in folds:

        fold = info[
            "fold"
        ]

        fold_dir = (
            root
            / f"fold_{fold}"
        )

        if not proxy_complete(
            fold_dir,
            args.max_epochs,
            args.eval_every,
        ):

            print(
                "[cv] not all folds complete; "
                "rerun identical command"
            )

            return

    # ========================================================
    # Select epochs.
    #
    # Same algorithm as the original run_cv.py.
    # ========================================================

    best_epochs = []

    fold_results = []

    expected_epochs = set(
        expected_proxy_epochs(
            args.max_epochs,
            args.eval_every,
        )
    )

    for info in folds:

        fold = info[
            "fold"
        ]

        fold_dir = (
            root
            / f"fold_{fold}"
        )

        records = []

        for file in sorted(
            fold_dir.glob(
                "proxy_*.json"
            )
        ):

            epoch = proxy_epoch(
                file
            )

            # Ignore stale files from a previous configuration.
            if (
                epoch
                not in expected_epochs
            ):

                continue

            metrics = json.loads(
                file.read_text(
                    encoding="utf-8"
                )
            )

            required = {
                "train_normal",
                "holdout_normal",
                "normal_gap",
                "synthetic",
                "separation",
                "pixel_f1",
            }

            missing = (
                required
                - set(metrics)
            )

            if missing:

                raise RuntimeError(
                    f"Invalid proxy result "
                    f"{file}: missing "
                    f"{sorted(missing)}"
                )

            metrics[
                "epoch"
            ] = epoch

            records.append(
                metrics
            )

        if not records:

            raise RuntimeError(
                f"No proxy records for "
                f"fold {fold}"
            )

        observed_epochs = {
            x["epoch"]
            for x in records
        }

        missing_epochs = (
            expected_epochs
            - observed_epochs
        )

        if missing_epochs:

            raise RuntimeError(
                f"Fold {fold} is missing "
                f"proxy epochs: "
                f"{sorted(missing_epochs)}"
            )

        best = choose_checkpoint(
            records,
            args.sensitivity_retain,
        )

        best_epochs.append(
            best["epoch"]
        )

        fold_results.append(
            {
                "fold":
                    fold,

                "best":
                    best,

                "records":
                    records,
            }
        )

        print(
            f"[fold {fold}] "
            f"best epoch="
            f"{best['epoch']} "
            f"gap="
            f"{best['normal_gap']:.4f} "
            f"sensitivity="
            f"{best['sensitivity']:.4f}"
        )

    final_epoch = int(
        statistics.median(
            best_epochs
        )
    )

    summary = {
        "best_epochs":
            best_epochs,

        "final_epoch":
            final_epoch,

        "fold_results":
            fold_results,
    }

    summary_file = (
        root
        / "cv_summary.json"
    )

    summary_file.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "[cv] best epochs:",
        best_epochs,
    )

    print(
        "[cv] final epoch:",
        final_epoch,
    )

    # ========================================================
    # Final training on all categories.
    #
    # No proxy evaluation is needed here.
    # Only final/last.pth is retained.
    # ========================================================

    final_dir = (
        root
        / "final"
    )

    final_checkpoint = (
        final_dir
        / "last.pth"
    )

    final_marker = (
        final_dir
        / "complete.json"
    )

    # --------------------------------------------------------
    # If final training was previously completed, use the
    # completion marker rather than relying merely on the
    # existence of last.pth. last.pth may represent an
    # interrupted epoch sequence.
    # --------------------------------------------------------

    if final_marker.is_file():

        try:

            marker = json.loads(
                final_marker.read_text(
                    encoding="utf-8"
                )
            )

            if (
                int(
                    marker.get(
                        "epoch",
                        -1,
                    )
                )
                == final_epoch
                and final_checkpoint.is_file()
            ):

                print(
                    "\nFINAL MODEL:",
                    final_checkpoint,
                )

                return

        except (
            ValueError,
            TypeError,
            json.JSONDecodeError,
        ):

            pass

    if insufficient_time(
        start,
        args.job_seconds,
    ):

        print(
            "[safe-exit] CV complete; "
            "rerun to continue final "
            "training"
        )

        return

    run([
        "torchrun",
        "--standalone",
        (
            "--nproc_per_node="
            f"{args.nproc}"
        ),
        "train.py",

        "--train-root",
        args.train_root,

        "--save-dir",
        str(
            final_dir
        ),

        "--encoder",
        args.encoder,

        "--epochs",
        str(
            final_epoch
        ),

        "--batch-size",
        str(
            args.batch_size
        ),

        "--grad-accum",
        str(
            args.grad_accum
        ),

        "--seed",
        str(
            args.seed
        ),

        "--job-seconds",
        str(
            child_job_seconds(
                start,
                args.job_seconds,
            )
        ),

        "--amp",

        "--no-keep-epoch-checkpoints",
    ])

    # --------------------------------------------------------
    # train.py should write complete.json on successfully
    # reaching its target epoch.
    # --------------------------------------------------------

    if final_marker.is_file():

        marker = json.loads(
            final_marker.read_text(
                encoding="utf-8"
            )
        )

        if (
            int(
                marker.get(
                    "epoch",
                    -1,
                )
            )
            == final_epoch
            and final_checkpoint.is_file()
        ):

            print()

            print(
                "FINAL MODEL:",
                final_checkpoint,
            )

            return

    print(
        "[safe-exit] final training "
        "incomplete; rerun identical "
        "command"
    )


if __name__ == "__main__":

    main()
