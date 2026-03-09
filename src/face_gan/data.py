from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Normalize, Resize, ToTensor


def get_transforms(size: Tuple[int, int]) -> Callable:
    transforms = [
        Resize(size),
        ToTensor(),
        Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
    return Compose(transforms)


class DatasetDirectory(Dataset):
    """
    Loads images from a directory containing files like `000001.jpg`.
    """

    def __init__(
        self,
        directory: str,
        transforms: Callable | None = None,
        extension: str = ".jpg",
    ) -> None:
        self.img_dir = directory
        self.transform = transforms
        self.extension = extension

        filenames = [
            os.path.join(directory, f)
            for f in os.listdir(directory)
            if f.endswith(extension)
        ]
        # Sorting keeps file order stable across OS/filesystems.
        self.img_filenames = sorted(filenames)

    def __len__(self) -> int:
        return len(self.img_filenames)

    def __getitem__(self, index: int) -> torch.Tensor:
        img_path = self.img_filenames[index]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image


@dataclass(frozen=True)
class DataConfig:
    data_dir: str
    image_size: int = 64
    extension: str = ".jpg"

