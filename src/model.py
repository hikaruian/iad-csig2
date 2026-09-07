from __future__ import annotations

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.init import trunc_normal_

from .blocks import (
    AggregationBlock,
    Mlp,
    PrototypeBlock,
)

from .encoder import DinoV2Encoder


def resize(x, size):
    return F.interpolate(
        x,
        size=size,
        mode="bilinear",
        align_corners=False,
    )


class ConvBlock(nn.Module):

    def __init__(self, cin, cout):
        super().__init__()

        groups = min(8, cout)

        while groups > 1 and cout % groups:
            groups -= 1

        self.net = nn.Sequential(
            nn.Conv2d(
                cin,
                cout,
                3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(
                groups,
                cout,
            ),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class PixelRefiner(nn.Module):

    def __init__(
        self,
        dim,
        levels=4,
        hidden=96,
    ):
        super().__init__()

        self.proj = nn.ModuleList([
            nn.Conv2d(
                dim,
                hidden,
                1,
            )
            for _ in range(levels)
        ])

        # semantic +
        # anomaly levels +
        # fixed RGB-gradient texture
        channels = (
            hidden * levels
            + levels
            + 4
        )

        self.body = nn.Sequential(
            ConvBlock(
                channels,
                hidden * 2,
            ),
            ConvBlock(
                hidden * 2,
                hidden,
            ),
            ConvBlock(
                hidden,
                hidden // 2,
            ),
        )

        self.head = nn.Conv2d(
            hidden // 2,
            1,
            1,
        )

    def forward(
        self,
        semantic,
        maps,
        texture,
        output_size,
    ):

        target_size = (
            texture.shape[-2:]
        )

        values = []

        for projection, feature in zip(
            self.proj,
            semantic,
        ):
            values.append(
                resize(
                    projection(
                        feature.detach()
                    ),
                    target_size,
                )
            )

        for amap in maps:
            values.append(
                resize(
                    amap.detach(),
                    target_size,
                )
            )

        values.append(
            texture
        )

        x = torch.cat(
            values,
            dim=1,
        )

        delta = torch.tanh(
            self.head(
                self.body(x)
            )
        )

        return resize(
            delta,
            output_size,
        )


class INPFormer(nn.Module):

    def __init__(
        self,
        encoder,
        inp_num=12,
        decoder_depth=8,
        bottleneck_drop=0.0,
        residual_strength=0.20,
    ):
        super().__init__()

        if len(encoder.target_layers) != 8:
            raise ValueError(
                "Expected exactly eight DINO layers"
            )

        if decoder_depth < 8:
            raise ValueError(
                "decoder_depth must be >= 8"
            )

        self.encoder = encoder

        self.embed_dim = encoder.embed_dim
        self.num_heads = encoder.num_heads
        self.target_layers = encoder.target_layers

        self.groups = (
            (0, 1),
            (2, 3),
            (4, 5),
            (6, 7),
        )

        dim = self.embed_dim

        self.feature_norm = nn.LayerNorm(
            dim,
            eps=1e-6,
            elementwise_affine=False,
        )

        # Reconstruction should not be dominated
        # by deepest category-semantic layers.
        self.input_logits = nn.Parameter(
            torch.tensor(
                [0.7, 1.2, 0.8, 0.2],
                dtype=torch.float32,
            )
        )

        # Image can use semantic evidence.
        self.image_logits = nn.Parameter(
            torch.tensor(
                [0.3, 0.8, 0.8, 0.3],
                dtype=torch.float32,
            ),
            requires_grad=False,
        )

        # Pixel strongly favors local levels.
        self.pixel_logits = nn.Parameter(
            torch.tensor(
                [1.6, 1.1, 0.2, -0.5],
                dtype=torch.float32,
            ),
            requires_grad=False,
        )

        self.prototype_token = nn.Parameter(
            torch.randn(
                inp_num,
                dim,
            )
        )

        norm = partial(
            nn.LayerNorm,
            eps=1e-6,
        )

        self.aggregation = nn.ModuleList([
            AggregationBlock(
                dim=dim,
                num_heads=self.num_heads,
                mlp_ratio=4.0,
                qkv_bias=True,
                norm_layer=norm,
            )
        ])

        self.bottleneck = nn.ModuleList([
            Mlp(
                dim,
                dim * 4,
                dim,
                drop=bottleneck_drop,
            )
        ])

        self.decoder = nn.ModuleList([
            PrototypeBlock(
                dim=dim,
                num_heads=self.num_heads,
                mlp_ratio=4.0,
                qkv_bias=True,
                norm_layer=norm,
            )
            for _ in range(decoder_depth)
        ])

        self.pixel_refiner = PixelRefiner(
            dim,
            levels=4,
            hidden=96,
        )

        self.residual_strength = float(
            residual_strength
        )

        self._init_weights()

    def _init_weights(self):

        trunc_normal_(
            self.prototype_token,
            std=0.02,
        )

        for module in (
            list(self.aggregation.modules())
            + list(self.bottleneck.modules())
            + list(self.decoder.modules())
        ):

            if isinstance(module, nn.Linear):

                trunc_normal_(
                    module.weight,
                    std=0.01,
                )

                if module.bias is not None:
                    nn.init.zeros_(
                        module.bias
                    )

    def trainable_parameters(self):

        yield self.prototype_token
        yield self.input_logits

        yield from self.aggregation.parameters()
        yield from self.bottleneck.parameters()
        yield from self.decoder.parameters()
        yield from self.pixel_refiner.parameters()

    def trainable_state_dict(self):

        prefixes = (
            "aggregation.",
            "bottleneck.",
            "decoder.",
            "pixel_refiner.",
        )

        return {
            key: value
            for key, value
            in self.state_dict().items()
            if (
                key in (
                    "prototype_token",
                    "input_logits",
                )
                or key.startswith(prefixes)
            )
        }

    def grouped_features(
        self,
        raw,
    ):

        raw = [
            self.feature_norm(
                feature.float()
            )
            for feature in raw
        ]

        return [
            (
                raw[a]
                + raw[b]
            ) * 0.5
            for a, b in self.groups
        ]

    @staticmethod
    def weighted_sum(
        tensors,
        logits,
    ):

        weights = torch.softmax(
            logits,
            dim=0,
        )

        result = (
            tensors[0]
            * weights[0]
        )

        for index in range(
            1,
            len(tensors),
        ):

            result = (
                result
                + tensors[index]
                * weights[index]
            )

        return (
            result,
            weights,
        )

    @staticmethod
    def gather_loss(
        query,
        keys,
    ):

        distance = (
            1.0
            - F.cosine_similarity(
                query.unsqueeze(2),
                keys.unsqueeze(1),
                dim=-1,
            )
        )

        return (
            distance.min(
                dim=2
            ).values.mean()
        )

    def reconstruct(
        self,
        grouped,
    ):

        fused, input_weights = (
            self.weighted_sum(
                grouped,
                self.input_logits,
            )
        )

        batch = fused.shape[0]

        prototype = (
            self.prototype_token
            .unsqueeze(0)
            .expand(
                batch,
                -1,
                -1,
            )
        )

        for block in self.aggregation:
            prototype = block(
                prototype,
                fused,
            )

        gather = self.gather_loss(
            fused,
            prototype,
        )

        tokens = fused

        for block in self.bottleneck:
            tokens = block(tokens)

        decoded = []

        for block in self.decoder:
            tokens = block(
                tokens,
                prototype,
            )

            decoded.append(tokens)

        decoded.reverse()

        de = [
            (
                decoded[0]
                + decoded[1]
            ) * 0.5,

            (
                decoded[2]
                + decoded[3]
            ) * 0.5,

            (
                decoded[4]
                + decoded[5]
            ) * 0.5,

            (
                decoded[6]
                + decoded[7]
            ) * 0.5,
        ]

        return (
            grouped,
            de,
            gather,
            input_weights,
        )

    @staticmethod
    def spatial(
        features,
        h,
        w,
    ):

        result = []

        for feature in features:

            b, n, c = feature.shape

            if n != h * w:
                raise RuntimeError(
                    f"Token mismatch: "
                    f"{n} vs {h}x{w}"
                )

            result.append(
                feature.transpose(
                    1,
                    2,
                ).reshape(
                    b,
                    c,
                    h,
                    w,
                ).contiguous()
            )

        return result

    @staticmethod
    def texture(
        x,
    ):

        # x is ImageNet-normalized, but gradients
        # remain useful for local boundary evidence.
        gray = (
            0.2989 * x[:, 0:1]
            + 0.5870 * x[:, 1:2]
            + 0.1140 * x[:, 2:3]
        )

        dx = (
            gray[:, :, :, 1:]
            - gray[:, :, :, :-1]
        )

        dy = (
            gray[:, :, 1:, :]
            - gray[:, :, :-1, :]
        )

        dx = F.pad(
            dx,
            (0, 1, 0, 0),
        )

        dy = F.pad(
            dy,
            (0, 0, 0, 1),
        )

        magnitude = torch.sqrt(
            dx.square()
            + dy.square()
            + 1e-6
        )

        texture = torch.cat(
            [
                gray,
                dx,
                dy,
                magnitude,
            ],
            dim=1,
        )

        return F.avg_pool2d(
            texture,
            4,
            4,
        )

    def forward(
        self,
        x,
        return_maps=False,
    ):

        raw = self.encoder.forward_features(
            x,
            self.target_layers,
        )

        grouped = self.grouped_features(
            raw
        )

        en, de, gather, input_weights = (
            self.reconstruct(
                grouped
            )
        )

        patch = self.encoder.patch_size

        h = x.shape[-2] // patch
        w = x.shape[-1] // patch

        en = self.spatial(
            en,
            h,
            w,
        )

        de = self.spatial(
            de,
            h,
            w,
        )

        if not return_maps:

            return (
                en,
                de,
                gather,
            )

        output_size = (
            x.shape[-2],
            x.shape[-1],
        )

        maps = []

        for e, d in zip(
            en,
            de,
        ):

            amap = (
                1.0
                - F.cosine_similarity(
                    e.float(),
                    d.float(),
                    dim=1,
                )
            ).unsqueeze(1)

            maps.append(
                resize(
                    amap,
                    output_size,
                )
            )

        image_map, image_weights = (
            self.weighted_sum(
                maps,
                self.image_logits,
            )
        )

        coarse, pixel_weights = (
            self.weighted_sum(
                maps,
                self.pixel_logits,
            )
        )

        coarse = (
            coarse / 2.0
        ).clamp(
            0.0,
            1.0,
        )

        delta = self.pixel_refiner(
            en,
            maps,
            self.texture(
                x.float()
            ),
            output_size,
        )

        # Critical:
        # refinement is multiplicatively gated by INP.
        pixel_map = (
            coarse
            + self.residual_strength
            * coarse.detach()
            * delta
        ).clamp(
            0.0,
            1.0,
        )

        return {
            "en":
                en,

            "de":
                de,

            "g_loss":
                gather,

            "level_maps":
                maps,

            "image_map":
                image_map,

            "coarse_pixel_map":
                coarse,

            "pixel_delta":
                delta,

            "pixel_map":
                pixel_map,

            "input_weights":
                input_weights,

            "image_weights":
                image_weights,

            "pixel_weights":
                pixel_weights,
        }


def build_model(
    encoder_name="dinov2reg_vit_large_14",
    inp_num=12,
    decoder_depth=8,
    bottleneck_drop=0.0,
    residual_strength=0.20,
    encoder_source="auto",
    **kwargs,
):

    encoder = DinoV2Encoder(
        encoder_name,
        source=encoder_source,
    )

    return INPFormer(
        encoder=encoder,
        inp_num=inp_num,
        decoder_depth=decoder_depth,
        bottleneck_drop=bottleneck_drop,
        residual_strength=residual_strength,
    )
