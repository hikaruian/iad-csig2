"""Transductive category/view DINO spatial memory."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn.functional as F


def normalize(x):
    return F.normalize(
        x.float(),
        dim=-1,
        eps=1e-6,
    )


@torch.no_grad()
def extract_features(
    encoder,
    images,
    start=0,
    end=4,
):

    layers = encoder.target_layers

    start = max(
        0,
        int(start),
    )

    end = min(
        len(layers),
        int(end),
    )

    if end <= start:
        raise ValueError(
            "Invalid memory layers"
        )

    features = encoder.forward_features(
        images,
        layers[start:end],
    )

    features = [
        normalize(x)
        for x in features
    ]

    return normalize(
        torch.stack(
            features
        ).mean(0)
    )


def descriptor(features):

    view = normalize(
        features.mean(1)
    )

    return normalize(
        view.flatten().unsqueeze(0)
    )[0]


@dataclass
class MemoryConfig:

    layer_start: int = 0
    layer_end: int = 4
    consensus_ratio: float = 0.70
    neighbors: int = 5
    max_samples: int = 8
    radius: int = 3


class CategoryMemory:

    def __init__(self, config):

        self.config = config

        self.features: Dict[
            int,
            torch.Tensor
        ] = {}

        self.descriptors = {}

        self.selected = []

    def add(
        self,
        index,
        features,
    ):

        self.features[index] = (
            features.detach()
            .cpu()
            .half()
        )

        self.descriptors[index] = (
            descriptor(features)
            .detach()
            .cpu()
            .float()
        )

    def finalize(self):

        indices = list(
            self.descriptors
        )

        if len(indices) <= 2:

            self.selected = indices

            return

        matrix = normalize(
            torch.stack([
                self.descriptors[i]
                for i in indices
            ])
        )

        dist = (
            1
            - matrix @ matrix.T
        ).clamp_min(0)

        dist.fill_diagonal_(
            float("inf")
        )

        k = min(
            max(
                1,
                self.config.neighbors,
            ),
            len(indices) - 1,
        )

        score = torch.topk(
            dist,
            k,
            largest=False,
            dim=1,
        ).values.mean(1)

        order = torch.argsort(
            score
        ).tolist()

        keep = min(
            len(indices),
            self.config.max_samples,
            max(
                2,
                int(
                    math.ceil(
                        len(indices)
                        * self.config
                        .consensus_ratio
                    )
                ),
            ),
        )

        self.selected = [
            indices[i]
            for i in order[:keep]
        ]

    @torch.no_grad()
    def maps(
        self,
        index,
        device,
        h,
        w,
        output_size,
    ):

        query = normalize(
            self.features[index]
            .to(
                device,
                dtype=torch.float32,
            )
        )

        output = []

        for view in range(
            query.shape[0]
        ):

            refs = [
                self.features[r][view]
                for r in self.selected
                if r != index
            ]

            if not refs:

                output.append(
                    torch.zeros(
                        output_size,
                        device=device,
                    )
                )

                continue

            refs = normalize(
                torch.stack(refs)
                .to(
                    device,
                    dtype=torch.float32,
                )
            )

            q = query[view]

            channels = q.shape[-1]

            ref_grid = refs.reshape(
                refs.shape[0],
                h,
                w,
                channels,
            )

            best = torch.full(
                (
                    h,
                    w,
                ),
                -1.0,
                device=device,
            )

            radius = self.config.radius

            qgrid = q.reshape(
                h,
                w,
                channels,
            )

            # Vectorized per spatial offset.
            for dy in range(
                -radius,
                radius + 1,
            ):

                for dx in range(
                    -radius,
                    radius + 1,
                ):

                    qy0 = max(
                        0,
                        -dy,
                    )

                    qy1 = min(
                        h,
                        h - dy,
                    )

                    qx0 = max(
                        0,
                        -dx,
                    )

                    qx1 = min(
                        w,
                        w - dx,
                    )

                    if (
                        qy1 <= qy0
                        or qx1 <= qx0
                    ):
                        continue

                    ry0 = qy0 + dy
                    ry1 = qy1 + dy

                    rx0 = qx0 + dx
                    rx1 = qx1 + dx

                    qpart = qgrid[
                        qy0:qy1,
                        qx0:qx1
                    ]

                    rpart = ref_grid[
                        :,
                        ry0:ry1,
                        rx0:rx1,
                        :
                    ]

                    similarity = (
                        rpart
                        * qpart.unsqueeze(0)
                    ).sum(-1)

                    local = similarity.max(0).values

                    best[
                        qy0:qy1,
                        qx0:qx1
                    ] = torch.maximum(
                        best[
                            qy0:qy1,
                            qx0:qx1
                        ],
                        local,
                    )

            distance = (
                1 - best
            ).clamp_min(0)

            amap = F.interpolate(
                distance[
                    None,
                    None
                ],
                size=output_size,
                mode="bilinear",
                align_corners=False,
            )[0, 0]

            output.append(amap)

        return torch.stack(output)
