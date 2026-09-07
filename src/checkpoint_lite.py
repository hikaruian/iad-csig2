from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

import torch


def trainable_state_dict(model) -> Dict[str, torch.Tensor]:
    """
    Store only non-DINO trainable state.

    Frozen DINO is always reconstructed from pretrained weights.
    """
    state = model.state_dict()

    return {
        k: v.detach().cpu()
        for k, v in state.items()
        if not k.startswith("encoder.model.")
    }


def load_trainable_state(
    model,
    state,
):
    missing, unexpected = model.load_state_dict(
        state,
        strict=False,
    )

    bad_missing = [
        k for k in missing
        if not k.startswith("encoder.model.")
    ]

    if bad_missing:
        raise RuntimeError(
            "Missing trainable model parameters:\n"
            + "\n".join(bad_missing[:30])
        )

    if unexpected:
        raise RuntimeError(
            "Unexpected checkpoint parameters:\n"
            + "\n".join(unexpected[:30])
        )


def atomic_torch_save(
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

    torch.save(
        obj,
        tmp,
    )

    os.replace(
        tmp,
        path,
    )


def save_resume(
    path,
    model,
    optimizer,
    scheduler,
    scaler,
    epoch,
    args,
    history,
    trained_categories,
):
    state = {
        "format": "inpformer-lite-v1",
        "epoch": int(epoch),
        "model": trainable_state_dict(model),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "args": dict(vars(args)),
        "history": history,
        "trained_categories": sorted(
            trained_categories
        ),
    }

    atomic_torch_save(
        state,
        path,
    )


def export_model(
    resume_path,
    output_path,
):
    ckpt = torch.load(
        resume_path,
        map_location="cpu",
    )

    compact = {
        "format": "inpformer-lite-v1",
        "epoch": ckpt["epoch"],
        "model": ckpt["model"],
        "args": ckpt["args"],
        "history": ckpt.get(
            "history",
            [],
        ),
        "trained_categories":
            ckpt.get(
                "trained_categories",
                [],
            ),
    }

    atomic_torch_save(
        compact,
        output_path,
    )
