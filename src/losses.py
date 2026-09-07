"""Generalized INP losses."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def reconstruction_loss(
    en,
    de,
    power=2.0,
):

    losses = []

    for e, d in zip(
        en,
        de,
    ):

        distance = (
            1.0
            - F.cosine_similarity(
                e.detach().float(),
                d.float(),
                dim=1,
            )
        )

        with torch.no_grad():

            mean = (
                distance.mean(
                    (-2, -1),
                    keepdim=True,
                )
                .clamp_min(1e-6)
            )

            weight = (
                distance.detach()
                / mean
            ).pow(
                power
            ).clamp(
                0.25,
                8.0,
            )

            weight = (
                weight
                / weight.mean(
                    (-2, -1),
                    keepdim=True,
                ).clamp_min(1e-6)
            )

        losses.append(
            (
                weight
                * distance
            ).mean()
        )

    return torch.stack(
        losses
    ).mean()


def total_loss(
    en,
    de,
    gather_loss,
    gather_weight=0.1,
    y=2.0,
):

    return (
        reconstruction_loss(
            en,
            de,
            y,
        )
        + gather_weight
        * gather_loss
    )


def pixel_loss(
    pixel_map,
    delta,
    mask,
):

    if mask.ndim == 3:
        mask = mask.unsqueeze(1)

    if (
        mask.shape[-2:]
        != pixel_map.shape[-2:]
    ):

        mask = F.interpolate(
            mask.float(),
            size=pixel_map.shape[-2:],
            mode="nearest",
        )

    target = mask.float().to(
        pixel_map.device
    )

    probability = (
        pixel_map.float()
        .clamp(
            1e-5,
            1 - 1e-5,
        )
    )

    logits = torch.logit(
        probability
    )

    bce = (
        F.binary_cross_entropy_with_logits(
            logits,
            target,
            reduction="none",
        )
    )

    pt = (
        probability * target
        + (
            1 - probability
        ) * (
            1 - target
        )
    )

    alpha = (
        0.25 * target
        + 0.75
        * (
            1 - target
        )
    )

    focal = (
        alpha
        * (
            1 - pt
        ).square()
        * bce
    ).mean()

    p = probability.flatten(1)
    y = target.flatten(1)

    positive = (
        y.sum(1) > 0
    )

    if positive.any():

        pp = p[positive]
        yy = y[positive]

        intersection = (
            pp * yy
        ).sum(1)

        dice = (
            1
            - (
                (
                    2 * intersection
                    + 1e-6
                )
                / (
                    pp.sum(1)
                    + yy.sum(1)
                    + 1e-6
                )
            ).mean()
        )

    else:

        dice = (
            probability.sum()
            * 0
        )

    normal = probability[
        target < 0.5
    ]

    if normal.numel():

        k = max(
            1,
            int(
                normal.numel()
                * 0.01
            ),
        )

        hard_negative = (
            torch.topk(
                normal,
                min(
                    k,
                    normal.numel(),
                ),
            )
            .values
            .square()
            .mean()
        )

    else:

        hard_negative = (
            probability.sum()
            * 0
        )

    delta_reg = (
        delta.float()
        .square()
        .mean()
    )

    total = (
        focal
        + 0.5 * dice
        + 0.05 * hard_negative
        + 0.01 * delta_reg
    )

    return {
        "focal": focal,
        "dice": dice,
        "hard_negative": hard_negative,
        "delta": delta_reg,
        "total": total,
    }
