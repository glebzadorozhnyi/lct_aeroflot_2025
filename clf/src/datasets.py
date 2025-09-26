from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from joblib import Parallel, delayed
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data.dataset import Dataset as _Dataset
from tqdm import tqdm


@dataclass
class Sample:
    """Representation of a single dataset sample.

    Attributes:
        path (str): Filesystem path to the image.
        label (str): Human-readable label for the sample.
        target (int): Integer class id.
        image (np.ndarray): Raw image bytes decoded into a numpy array (H, W, C).
    """

    path: str = ""
    label: str = ""
    target: int = -1
    image: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.uint8))

    def __str__(self) -> str:
        return f"Sample(\
path='{self.path}', \
label='{self.label}', \
target={self.target}, \
image=np.array(shape={self.image.shape}, dtype={self.image.dtype})\
)"


class Batch:
    """Container for a batch of Sample objects.

    This class converts a list of Sample into batched numpy/torch structures
    suitable for training/validation, and provides a pin_memory method to speed up
    transfers to CUDA.
    """

    def __init__(self, samples: list[Sample]):
        self.path = np.array([sample.path for sample in samples], dtype=object)
        self.label = np.array([sample.label for sample in samples], dtype=object)
        self.target = torch.asarray([sample.target for sample in samples], dtype=torch.long)
        self.image = torch.stack([torch.from_numpy(sample.image) for sample in samples], dim=0)

    def pin_memory(self) -> object:
        """Pin batch tensors to page-locked memory.

        Returns:
            Batch: self with pinned tensors (target and image).
        """
        self.target = self.target.pin_memory()
        self.image = self.image.pin_memory()
        return self


def collate_fn(samples: list[Sample]) -> Batch:
    """Collate function for DataLoader to convert samples list into Batch.

    Args:
        samples (list[Sample]): List of dataset samples returned by __getitem__.

    Returns:
        Batch: Batched representation combining the individual samples.
    """
    return Batch(samples)


class Dataset(_Dataset):
    """PyTorch Dataset for .

    The dataset reads `annotations.csv` from the provided root directory and supports:
    - Filtering by split ('train', 'valid', or other)
    - Subsampling by ratio (useful for debugging / quick experiments)
    - Preloading image bytes into memory for faster iteration
    - Applying albumentations transforms via transform_image helper

    Args:
        root (str): Path to dataset root (must contain annotations.csv and image files).
        split (str, optional): Which split to load ('train'|'valid'|other). Defaults to "train".
        ratio (float, optional): Fraction of dataset to keep (0.0 < ratio <= 1.0). Defaults to 1.0.
        preload (bool, optional): If True, loads all image bytes into memory. Defaults to False.
        transform (albumentations.Compose | None, optional): Transform pipeline to apply to images.
    """

    @staticmethod
    def read_bytes(path) -> np.ndarray:
        with open(path, "rb") as f:
            data = f.read()
            np_data = np.frombuffer(data, dtype=np.uint8)
        return np_data

    @staticmethod
    def read_image(path, box, image_size) -> Image.Image:
        image = Image.open(path).convert("RGB")
        image = image.crop(tuple(box))
        w, h = image.size
        ratio = image_size * 1.2 / max(h, w)
        image = image.resize((int(w * ratio), int(h * ratio)))
        return image

    @staticmethod
    def pad2square(pil_image: Image.Image, pad_value: int = 0) -> np.ndarray:
        """
        Simple version: pad image to square, center-aligned, single pad value.

        Args:
            pil_image (Image.Image): Input image.
            pad_value (int): Padding value for all channels.

        Returns:
            Image.Image: Square image.
        """
        image = np.array(pil_image)
        h, w = image.shape[:2]
        if h == w:
            return pil_image
        size = max(h, w)
        top = (size - h) // 2
        left = (size - w) // 2
        image = cv2.copyMakeBorder(
            image,
            top,
            size - h - top,
            left,
            size - w - left,
            cv2.BORDER_CONSTANT,
            value=pad_value,
        )
        return Image.fromarray(image)

    def __init__(
        self,
        root: str,
        split: str = "train",
        image_size: int = 112,
    ):
        super().__init__()
        self.root = root
        self.split = split
        self.image_size = image_size
        root = Path(root)
        df = pd.read_csv(root / "annotations.csv")
        df = df[~((df.x0 == 0) & (df.y0 == 0) & (df.x1 == 0) & (df.y1 == 0))].reset_index(drop=True)
        self.paths = df.path.values
        self.labels = df.label.values
        self.targets = df.target.values
        self.boxes = np.stack(
            [np.array([row.x0, row.y0, row.x1, row.y1]) for _, row in df.iterrows()],
            axis=0,
        )
        idxs = np.arange(len(self.paths))
        train_idxs, val_idxs = train_test_split(
            idxs, test_size=0.2, random_state=42, shuffle=True, stratify=self.targets
        )
        idxs = train_idxs if split == "train" else val_idxs
        self.paths = self.paths[idxs]
        self.labels = self.labels[idxs]
        self.targets = self.targets[idxs]
        self.boxes = self.boxes[idxs]
        with Parallel(n_jobs=8) as p:
            self.images = p(
                delayed(self.read_image)(root / label / path, box, image_size)
                for path, label, box in tqdm(zip(self.paths, self.labels, self.boxes), desc="Loading images")
            )
        if split == "train":
            self.transform = T.Compose(
                [
                    T.RandomResizedCrop((image_size, image_size), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
                    T.RandomHorizontalFlip(p=0.5),
                    T.RandomVerticalFlip(p=0.5),
                    T.RandAugment(interpolation=T.InterpolationMode.BILINEAR),
                    T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                ]
            )
        else:
            self.transform = T.Compose(
                [
                    T.Resize((image_size, image_size)),
                ]
            )

    def __len__(self) -> int:
        """Return the number of samples in the dataset.

        Returns:
            int: Dataset length.
        """
        return len(self.paths)

    def __getitem__(self, index: any) -> Sample:
        """Retrieve a dataset sample by index.

        This method will decode the image (from preloaded bytes or from disk),
        apply optional transforms, and return a Sample.

        Args:
            index (int | slice | any): Index identifying the sample.

        Returns:
            Sample: The dataset sample containing path, label, target, size and image array.
        """
        path = self.paths[index]
        label = self.labels[index]
        target = self.targets[index]
        image = self.images[index]
        image = self.pad2square(image, pad_value=0)
        image = self.transform(image)
        image = np.array(image)
        return Sample(path=path, label=label, target=target, image=image)
