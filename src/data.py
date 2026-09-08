"""CSIG dataset + deterministic synthetic anomalies."""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Callable, List, Optional, Set, Tuple

import numpy as np
import torch

from PIL import (
    Image,
    ImageDraw,
    ImageEnhance,
    ImageFilter,
)

from torch.utils.data import Dataset

from torchvision import transforms
from torchvision.transforms import functional as TF


IMAGENET_MEAN = (
    0.485,
    0.456,
    0.406,
)

IMAGENET_STD = (
    0.229,
    0.224,
    0.225,
)

VIEWS = (
    0,
    1,
    2,
    3,
    4,
)


SEEN_CATEGORIES = {
    "3_adapter",
    "DVD_switch",
    "D_sub_connector",
    "PLCC_socket",
    "VR_joystick",
    "accurate_detection_switch",
    "battery",
    "blade_switch",
    "boost_converter_module",
    "button_battery_holder",
    "circuit_breaker",
    "connector_housing_female",
    "crimp_st_cable_mount_box",
    "dc_jack",
    "dc_power_connector",
    "detection_switch",
    "effect_transistor",
    "electronic_watch_movement",
    "ffc_connector_plug",
    "ingot_buckle",
    "laser_diode",
    "lego_pin_connector_plate",
    "limit_switch",
    "lithium_battery_plug",
    "littel_fuse",
    "lock",
    "miniature_lifting_motor",
    "mobile_charging_connector",
    "motor_bracket",
    "motor_gear_reducer",
    "motor_plug",
    "pencil_sharpener",
    "pinboard_connector",
    "potentiometer",
    "power_jack",
    "power_strip_socket",
    "purple_clay_pot",
    "retaining_ring",
    "rheostat",
    "self_lock_switch",
    "silicon_cell_sensor",
    "single_switch",
    "smd_receiver_module",
    "suction_cup",
    "toy_tire",
    "travel_switch",
    "vacuum_switch",
    "vehicle_harness_conductor",
    "vibration_motor",
    "wireless_receiver_module",
}


def build_transform(
    image_size: int = 448,
    is_train: bool = False,
):

    return transforms.Compose([
        transforms.Resize(
            (
                image_size,
                image_size,
            ),
            interpolation=(
                transforms
                .InterpolationMode
                .BICUBIC
            ),
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            IMAGENET_MEAN,
            IMAGENET_STD,
        ),
    ])


def discover_samples(
    root: Path,
):

    root = Path(
        root
    )

    if not root.is_dir():
        raise FileNotFoundError(
            f"Dataset not found: {root}"
        )

    samples = []

    for category_dir in sorted(
        p
        for p in root.iterdir()
        if p.is_dir()
    ):

        for sample_dir in sorted(
            p
            for p in category_dir.iterdir()
            if p.is_dir()
        ):

            samples.append(
                (
                    category_dir.name,
                    sample_dir.name,
                    sample_dir,
                )
            )

    if not samples:
        raise RuntimeError(
            f"No samples under {root}"
        )

    return samples


def view_path(
    sample_dir: Path,
    view: int,
):

    for filename in (
        f"{view}.png",
        f"{view}.PNG",
        f"{view:02d}.png",
        f"{view}.jpg",
        f"{view}.jpeg",
    ):

        path = (
            sample_dir
            / filename
        )

        if path.is_file():
            return path

    raise FileNotFoundError(
        f"Missing view {view} "
        f"in {sample_dir}"
    )


def load_rgb(
    path: Path,
):

    with Image.open(
        path
    ) as image:

        result = image.convert(
            "RGB"
        )

        result.load()

    return result


def to_normalized_tensor(
    image: Image.Image,
):

    tensor = TF.to_tensor(
        image
    )

    return TF.normalize(
        tensor,
        IMAGENET_MEAN,
        IMAGENET_STD,
    )


def mask_tensor(
    mask: Image.Image,
):

    array = (
        np.asarray(
            mask,
            dtype=np.uint8,
        )
        > 127
    ).astype(
        np.float32
    )

    return torch.from_numpy(
        array
    ).unsqueeze(
        0
    )


# ============================================================
# Synthetic anomaly
# ============================================================

def make_mask(
    size: int,
    rng: random.Random,
):

    kind = rng.choices(
        [
            "scratch",
            "spot",
            "rectangle",
            "blob",
        ],
        weights=[
            0.40,
            0.20,
            0.15,
            0.25,
        ],
        k=1,
    )[0]

    mask = Image.new(
        "L",
        (
            size,
            size,
        ),
        0,
    )

    draw = ImageDraw.Draw(
        mask
    )

    if kind == "scratch":

        x = rng.randint(
            0,
            size - 1,
        )

        y = rng.randint(
            0,
            size - 1,
        )

        angle = rng.uniform(
            0,
            2 * math.pi,
        )

        length = rng.randint(
            max(
                10,
                size // 20,
            ),
            max(
                20,
                size // 3,
            ),
        )

        width = rng.randint(
            1,
            max(
                2,
                size // 120,
            ),
        )

        points = []

        xf = float(
            x
        )

        yf = float(
            y
        )

        curvature = rng.uniform(
            -0.005,
            0.005,
        )

        for _ in range(
            length
        ):

            if not (
                0 <= xf < size
                and 0 <= yf < size
            ):
                break

            points.append(
                (
                    int(xf),
                    int(yf),
                )
            )

            angle += curvature

            xf += math.cos(
                angle
            )

            yf += math.sin(
                angle
            )

        if len(points) >= 2:

            draw.line(
                points,
                fill=255,
                width=width,
            )

    elif kind == "spot":

        radius = rng.randint(
            max(
                2,
                size // 160,
            ),
            max(
                3,
                size // 30,
            ),
        )

        x = rng.randint(
            radius,
            max(
                radius,
                size - radius - 1,
            ),
        )

        y = rng.randint(
            radius,
            max(
                radius,
                size - radius - 1,
            ),
        )

        draw.ellipse(
            (
                x - radius,
                y - radius,
                x + radius,
                y + radius,
            ),
            fill=255,
        )

    elif kind == "rectangle":

        width = rng.randint(
            3,
            max(
                4,
                size // 8,
            ),
        )

        height = rng.randint(
            3,
            max(
                4,
                size // 8,
            ),
        )

        x = rng.randint(
            0,
            max(
                0,
                size - width,
            ),
        )

        y = rng.randint(
            0,
            max(
                0,
                size - height,
            ),
        )

        draw.rectangle(
            (
                x,
                y,
                x + width,
                y + height,
            ),
            fill=255,
        )

    else:

        low_size = rng.choice(
            [
                8,
                16,
                32,
            ]
        )

        np_rng = np.random.default_rng(
            rng.randint(
                0,
                2**31 - 1,
            )
        )

        noise = (
            np_rng.random(
                (
                    low_size,
                    low_size,
                )
            )
            * 255
        ).astype(
            np.uint8
        )

        blob = Image.fromarray(
            noise
        ).resize(
            (
                size,
                size,
            ),
            Image.Resampling.BICUBIC,
        )

        array = np.asarray(
            blob
        )

        threshold = np.quantile(
            array,
            0.82,
        )

        mask = Image.fromarray(
            (
                array
                > threshold
            ).astype(
                np.uint8
            )
            * 255
        )

    return mask


def corrupt_image(
    image: Image.Image,
    rng: random.Random,
):

    mode = rng.choice(
        [
            "brightness",
            "contrast",
            "color",
            "blur",
            "noise",
        ]
    )

    if mode == "brightness":

        return (
            ImageEnhance
            .Brightness(image)
            .enhance(
                rng.choice([
                    rng.uniform(
                        0.4,
                        0.75,
                    ),
                    rng.uniform(
                        1.25,
                        1.8,
                    ),
                ])
            )
        )

    if mode == "contrast":

        return (
            ImageEnhance
            .Contrast(image)
            .enhance(
                rng.uniform(
                    0.4,
                    2.0,
                )
            )
        )

    if mode == "color":

        return (
            ImageEnhance
            .Color(image)
            .enhance(
                rng.uniform(
                    0.3,
                    1.8,
                )
            )
        )

    if mode == "blur":

        return image.filter(
            ImageFilter.GaussianBlur(
                rng.uniform(
                    1.0,
                    4.0,
                )
            )
        )

    np_rng = np.random.default_rng(
        rng.randint(
            0,
            2**31 - 1,
        )
    )

    array = np.asarray(
        image,
        dtype=np.float32,
    )

    noise = np_rng.normal(
        0,
        rng.uniform(
            8,
            40,
        ),
        array.shape,
    )

    array = np.clip(
        array + noise,
        0,
        255,
    ).astype(
        np.uint8
    )

    return Image.fromarray(
        array
    )


# ============================================================
# Image dataset
# ============================================================

class CSIGImageDataset(Dataset):

    def __init__(
        self,
        root: str,
        transform: Optional[
            Callable
        ] = None,
        image_size: int = 448,
        synthetic_anomaly: bool = False,
        synthetic_prob: float = 0.8,
        include_categories: Optional[
            Set[str]
        ] = None,
        deterministic_synthetic: bool = False,
        synthetic_seed: int = 10000,
    ):

        self.root = Path(
            root
        )

        self.image_size = int(
            image_size
        )

        self.transform = (
            transform
            or build_transform(
                image_size,
                True,
            )
        )

        self.synthetic_anomaly = (
            synthetic_anomaly
        )

        self.synthetic_prob = float(
            synthetic_prob
        )

        self.include_categories = (
            include_categories
        )

        self.deterministic_synthetic = (
            deterministic_synthetic
        )

        self.synthetic_seed = int(
            synthetic_seed
        )

        self.items = []

        for (
            category,
            sample_id,
            sample_dir,
        ) in discover_samples(
            self.root
        ):

            if (
                include_categories
                is not None
                and category
                not in include_categories
            ):

                continue

            for view in VIEWS:

                try:

                    self.items.append(
                        (
                            category,
                            sample_id,
                            view,
                            view_path(
                                sample_dir,
                                view,
                            ),
                        )
                    )

                except FileNotFoundError:

                    continue

        if not self.items:

            raise RuntimeError(
                "Dataset contains no images."
            )

        self.classes = sorted({
            item[0]
            for item
            in self.items
        })

        self.class_to_idx = {
            category: index
            for index, category
            in enumerate(
                self.classes
            )
        }

    def __len__(
        self,
    ):

        return len(
            self.items
        )

    def __getitem__(
        self,
        index,
    ):

        (
            category,
            sample_id,
            view,
            path,
        ) = self.items[
            index
        ]

        image = load_rgb(
            path
        )

        if not self.synthetic_anomaly:

            return (
                self.transform(
                    image
                ),
                self.class_to_idx[
                    category
                ],
            )

        clean = image.resize(
            (
                self.image_size,
                self.image_size,
            ),
            Image.Resampling.BICUBIC,
        )

        if (
            self.deterministic_synthetic
        ):

            rng = random.Random(
                self.synthetic_seed
                + index * 1009
            )

        else:

            rng = random

        create_anomaly = (
            rng.random()
            < self.synthetic_prob
        )

        if create_anomaly:

            mask = make_mask(
                self.image_size,
                rng,
            )

            corrupted = corrupt_image(
                clean,
                rng,
            )

            alpha = rng.uniform(
                0.55,
                1.0,
            )

            anomaly = Image.blend(
                clean,
                corrupted,
                alpha,
            )

            synthetic = (
                Image.composite(
                    anomaly,
                    clean,
                    mask,
                )
            )

        else:

            mask = Image.new(
                "L",
                clean.size,
                0,
            )

            synthetic = (
                clean.copy()
            )

        return {

            "clean":
                to_normalized_tensor(
                    clean
                ),

            "synthetic":
                to_normalized_tensor(
                    synthetic
                ),

            "mask":
                mask_tensor(
                    mask
                ),

            "class_idx":
                self.class_to_idx[
                    category
                ],

            "category":
                category,

            "sample_id":
                sample_id,

            "view_id":
                view,
        }


# ============================================================
# Physical-sample dataset
# ============================================================

class CSIGSampleDataset(Dataset):

    def __init__(
        self,
        root: str,
        transform=None,
        image_size=448,
        include_categories: Optional[
            Set[str]
        ] = None,
    ):

        samples = discover_samples(
            Path(root)
        )

        if (
            include_categories
            is not None
        ):

            samples = [
                sample
                for sample
                in samples
                if sample[0]
                in include_categories
            ]

        self.samples = samples

        self.transform = (
            transform
            or build_transform(
                image_size,
                False,
            )
        )

    def __len__(
        self,
    ):

        return len(
            self.samples
        )

    def __getitem__(
        self,
        index,
    ):

        (
            category,
            sample_id,
            sample_dir,
        ) = self.samples[
            index
        ]

        images = []

        for view in VIEWS:

            image = load_rgb(
                view_path(
                    sample_dir,
                    view,
                )
            )

            images.append(
                self.transform(
                    image
                )
            )

        return {

            "images":
                torch.stack(
                    images,
                    dim=0,
                ),

            "group_folder":
                (
                    f"{category}/"
                    f"{sample_id}"
                ),

            "category":
                category,

            "sample_id":
                sample_id,
        }
