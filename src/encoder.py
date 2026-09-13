"""Frozen DINOv2-with-registers encoder."""

from __future__ import annotations

import time
from pathlib import Path
from typing import List, Sequence

import torch
import torch.nn as nn

from .dist_utils import barrier, env_world, is_main


ENCODER_PRESETS = {
    "dinov2reg_vit_small_14": {
        "hub": "dinov2_vits14_reg",
        "timm": "vit_small_patch14_reg4_dinov2.lvd142m",
        "embed_dim": 384,
        "num_heads": 6,
        "target_layers": [2, 3, 4, 5, 6, 7, 8, 9],
        "patch_size": 14,
    },
    "dinov2reg_vit_base_14": {
        "hub": "dinov2_vitb14_reg",
        "timm": "vit_base_patch14_reg4_dinov2.lvd142m",
        "embed_dim": 768,
        "num_heads": 12,
        "target_layers": [2, 3, 4, 5, 6, 7, 8, 9],
        "patch_size": 14,
    },
    "dinov2reg_vit_large_14": {
        "hub": "dinov2_vitl14_reg",
        "timm": "vit_large_patch14_reg4_dinov2.lvd142m",
        "embed_dim": 1024,
        "num_heads": 16,
        "target_layers": [4, 6, 8, 10, 12, 14, 16, 18],
        "patch_size": 14,
    },
}


def prefetch_encoder_weights(
    name: str,
    source: str = "auto",
    stamp_dir: str = "runs/_prefetch",
):
    rank, _, world = env_world()

    if world <= 1:
        return

    stamp = Path(stamp_dir) / f"{name}.{source}.ready"
    fail = Path(stamp_dir) / f"{name}.{source}.fail"

    stamp.parent.mkdir(parents=True, exist_ok=True)

    if rank == 0:
        try:
            if fail.exists():
                fail.unlink()

            model = DinoV2Encoder(name, source)
            del model

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            stamp.write_text("ok", encoding="utf-8")

        except Exception as exc:
            fail.write_text(repr(exc), encoding="utf-8")
            raise

        return

    deadline = time.time() + 3600

    while time.time() < deadline:
        if stamp.exists():
            return

        if fail.exists():
            raise RuntimeError(
                fail.read_text(encoding="utf-8")
            )

        time.sleep(1)

    raise TimeoutError(
        f"Timeout while waiting for {name}"
    )


class DinoV2Encoder(nn.Module):

    def __init__(
        self,
        name="dinov2reg_vit_large_14",
        source="auto",
    ):
        super().__init__()

        if name not in ENCODER_PRESETS:
            raise ValueError(
                f"Unknown encoder {name}"
            )

        self.cfg = ENCODER_PRESETS[name]
        self.name = name
        self.backend = None

        self.model = self._load(source)

        self.model.eval()

        for p in self.model.parameters():
            p.requires_grad_(False)

    @property
    def embed_dim(self):
        return int(self.cfg["embed_dim"])

    @property
    def num_heads(self):
        return int(self.cfg["num_heads"])

    @property
    def target_layers(self):
        return list(self.cfg["target_layers"])

    @property
    def patch_size(self):
        return int(self.cfg["patch_size"])

    def _load(self, source):

        if source not in ("auto", "hub", "timm"):
            raise ValueError(source)

        order = (
            ("hub", "timm")
            if source == "auto"
            else (source,)
        )

        errors = []

        for kind in order:
            try:
                if kind == "hub":

                    model = torch.hub.load(
                        "facebookresearch/dinov2",
                        self.cfg["hub"],
                        pretrained=True,
                        trust_repo=True,
                    )

                    self.backend = "hub"
                    return model

                """
                import timm

                model = timm.create_model(
                    self.cfg["timm"],
                    pretrained=True,
                    dynamic_img_size=True,
                    num_classes=0,
                )

                self.backend = "timm"
                return model
                """

            except Exception as exc:
                errors.append(
                    f"{kind}: {exc}"
                )

        raise RuntimeError(
            "Failed to load DINOv2\n"
            + "\n".join(errors)
        )

    def train(self, mode=True):
        super().train(False)
        self.model.eval()
        return self

    @torch.no_grad()
    def forward_features(
        self,
        x: torch.Tensor,
        layers: Sequence[int] | None = None,
    ) -> List[torch.Tensor]:

        layers = (
            list(layers)
            if layers is not None
            else self.target_layers
        )

        n_expected = (
            x.shape[-2] // self.patch_size
        ) * (
            x.shape[-1] // self.patch_size
        )

        if self.backend == "hub":

            features = self.model.get_intermediate_layers(
                x,
                n=list(layers),
                reshape=False,
                return_class_token=False,
                norm=False,
            )

        else:

            features = self.model.forward_intermediates(
                x,
                indices=list(layers),
                norm=False,
                stop_early=True,
                output_fmt="NLC",
                intermediates_only=True,
            )

        result = []

        prefix = int(
            getattr(
                self.model,
                "num_prefix_tokens",
                1 + int(
                    getattr(
                        self.model,
                        "num_reg_tokens",
                        getattr(
                            self.model,
                            "num_register_tokens",
                            0,
                        ),
                    )
                ),
            )
        )

        for feature in features:

            if feature.ndim == 4:

                if feature.shape[1] == self.embed_dim:
                    feature = (
                        feature
                        .flatten(2)
                        .transpose(1, 2)
                    )

                else:
                    feature = feature.reshape(
                        feature.shape[0],
                        -1,
                        feature.shape[-1],
                    )

            if feature.ndim != 3:
                raise RuntimeError(
                    f"Unexpected DINO shape "
                    f"{tuple(feature.shape)}"
                )

            if feature.shape[1] != n_expected:

                if (
                    feature.shape[1]
                    >= n_expected + prefix
                ):

                    feature = feature[
                        :,
                        prefix:
                        prefix + n_expected,
                        :,
                    ]

                else:

                    raise RuntimeError(
                        f"DINO token count="
                        f"{feature.shape[1]}, "
                        f"expected={n_expected}"
                    )

            if feature.shape[1] != n_expected:
                raise RuntimeError(
                    "Failed to remove prefix tokens"
                )

            result.append(
                feature.detach().contiguous()
            )

        if len(result) != len(layers):
            raise RuntimeError(
                "Wrong number of DINO intermediate layers"
            )

        return result
