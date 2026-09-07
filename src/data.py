from __future__ import annotations

import math
import random
from pathlib import Path

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

# Keep your existing SEEN_CATEGORIES here.
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
    image_size=448,
    is_train=False,
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


def discover_samples(root):
    root = Path(root)

    result = []

    for category in sorted(
        p
        for p in root.iterdir()
        if p.is_dir()
    ):
        for sample in sorted(
            p
            for p in category.iterdir()
            if p.is_dir()
        ):
            result.append(
                (
                    category.name,
                    sample.name,
                    sample,
                )
            )

    return result


def view_path(
    directory,
    view,
):
    for name in (
        f"{view}.png",
        f"{view}.PNG",
        f"{view}.jpg",
        f"{view}.jpeg",
    ):
        path = directory / name

        if path.is_file():
            return path

    raise FileNotFoundError(
        directory
    )


def load_image(path):
    with Image.open(path) as image:
        result = image.convert("RGB")
        result.load()

    return result


def normalized_tensor(image):
    return TF.normalize(
        TF.to_tensor(image),
        IMAGENET_MEAN,
        IMAGENET_STD,
    )


def synthetic_mask(size):
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

    kind = random.choice(
        [
            "scratch",
            "spot",
            "blob",
            "rectangle",
        ]
    )

    if kind == "scratch":
        x = random.randint(
            0,
            size - 1,
        )

        y = random.randint(
            0,
            size - 1,
        )

        angle = random.random() * (
            2 * math.pi
        )

        length = random.randint(
            max(
                10,
                size // 20,
            ),
            max(
                20,
                size // 3,
            ),
        )

        width = random.randint(
            1,
            max(
                2,
                size // 100,
            ),
        )

        x2 = int(
            x
            + math.cos(
                angle
            )
            * length
        )

        y2 = int(
            y
            + math.sin(
                angle
            )
            * length
        )

        draw.line(
            (
                x,
                y,
                x2,
                y2,
            ),
            fill=255,
            width=width,
        )

    elif kind == "spot":
        radius = random.randint(
            2,
            max(
                3,
                size // 25,
            ),
        )

        x = random.randint(
            radius,
            size - radius,
        )

        y = random.randint(
            radius,
            size - radius,
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
        w = random.randint(
            3,
            max(
                4,
                size // 8,
            ),
        )

        h = random.randint(
            3,
            max(
                4,
                size // 8,
            ),
        )

        x = random.randint(
            0,
            size - w,
        )

        y = random.randint(
            0,
            size - h,
        )

        draw.rectangle(
            (
                x,
                y,
                x + w,
                y + h,
            ),
            fill=255,
        )

    else:
        low = np.random.rand(
            16,
            16,
        )

        low = (
            low * 255
        ).astype(
            np.uint8
        )

        blob = Image.fromarray(
            low
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

        mask = Image.fromarray(
            (
                array
                > np.quantile(
                    array,
                    0.8,
                )
            ).astype(
                np.uint8
            )
            * 255
        )

    return mask


def corrupt(
    image,
):
    choice = random.randint(
        0,
        3,
    )

    if choice == 0:
        return ImageEnhance.Brightness(
            image
        ).enhance(
            random.uniform(
                0.4,
                1.8,
            )
        )

    if choice == 1:
        return ImageEnhance.Contrast(
            image
        ).enhance(
            random.uniform(
                0.4,
                2.0,
            )
        )

    if choice == 2:
        return image.filter(
            ImageFilter.GaussianBlur(
                random.uniform(
                    1,
                    4,
                )
            )
        )

    array = np.asarray(
        image,
        dtype=np.float32,
    )

    array += np.random.normal(
        0,
        random.uniform(
            8,
            40,
        ),
        array.shape,
    )

    return Image.fromarray(
        np.clip(
            array,
            0,
            255,
        ).astype(
            np.uint8
        )
    )


class CSIGImageDataset(Dataset):
    def __init__(
        self,
        root,
        transform=None,
        image_size=448,
        synthetic_anomaly=False,
        synthetic_prob=0.8,
    ):
        self.root = Path(root)

        self.image_size = (
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

        self.synthetic_prob = (
            synthetic_prob
        )

        self.items = []

        for (
            category,
            sid,
            directory,
        ) in discover_samples(
            root
        ):
            for view in VIEWS:
                try:
                    self.items.append(
                        (
                            category,
                            sid,
                            view,
                            view_path(
                                directory,
                                view,
                            ),
                        )
                    )
                except FileNotFoundError:
                    pass

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

    def __len__(self):
        return len(
            self.items
        )

    def __getitem__(
        self,
        index,
    ):
        (
            category,
            sid,
            view,
            path,
        ) = self.items[
            index
        ]

        image = load_image(
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
            random.random()
            < self.synthetic_prob
        ):
            mask = synthetic_mask(
                self.image_size
            )

            anomaly = corrupt(
                clean
            )

            alpha = random.uniform(
                0.6,
                1.0,
            )

            anomaly = Image.blend(
                clean,
                anomaly,
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
            synthetic = (
                clean.copy()
            )

            mask = Image.new(
                "L",
                clean.size,
                0,
            )

        mask_array = (
            np.asarray(
                mask
            )
            > 127
        ).astype(
            np.float32
        )

        return {
            "clean":
                normalized_tensor(
                    clean
                ),

            "synthetic":
                normalized_tensor(
                    synthetic
                ),

            "mask":
                torch.from_numpy(
                    mask_array
                ).unsqueeze(
                    0
                ),

            "class_idx":
                self.class_to_idx[
                    category
                ],
        }


class CSIGSampleDataset(Dataset):
    def __init__(
        self,
        root,
        transform=None,
        image_size=448,
    ):
        self.samples = (
            discover_samples(
                root
            )
        )

        self.transform = (
            transform
            or build_transform(
                image_size,
                False,
            )
        )

    def __len__(self):
        return len(
            self.samples
        )

    def __getitem__(
        self,
        index,
    ):
        category, sid, directory = (
            self.samples[
                index
            ]
        )

        images = []

        for view in VIEWS:
            images.append(
                self.transform(
                    load_image(
                        view_path(
                            directory,
                            view,
                        )
                    )
                )
            )

        return {
            "images":
                torch.stack(
                    images
                ),

            "group_folder":
                f"{category}/{sid}",

            "category":
                category,

            "sample_id":
                sid,
        }
