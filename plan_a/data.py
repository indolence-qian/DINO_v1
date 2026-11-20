"""Use the existing industrial dataset layouts with consistent mask handling."""

from collections import defaultdict
from pathlib import Path
import random

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class InspectionData(Dataset):
    def __init__(self, name, size, root=None, category="ALL"):
        from Datasets import DATASET_REGISTRY

        loader, splits, default_root = DATASET_REGISTRY[name]
        source = Path(root or default_root)
        if not source.is_dir():
            raise FileNotFoundError(f"{name} dataset directory is missing: {source}")
        dataset = loader(source=str(source), split=splits.TEST, classname=category,
                         resize=size, imagesize=size)
        self.records = dataset.data_to_iterate
        if not self.records:
            raise ValueError(f"No samples found for {name}/{category} under {source}.")
        self.size = size
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        for _, state, image_path, mask_path in self.records:
            if self.label(state) and (not mask_path or not Path(mask_path).is_file()):
                raise FileNotFoundError(f"Missing anomaly mask for {image_path}: {mask_path}")

    @staticmethod
    def label(state):
        return int(state.casefold() not in {"good", "normal", "ok"})

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        _, state, image_path, mask_path = self.records[i]
        label = self.label(state)
        with Image.open(image_path) as image:
            image_tensor = self.transform(image.convert("RGB"))
        if label:
            with Image.open(mask_path) as mask:
                mask = mask.convert("L").resize((self.size, self.size), Image.Resampling.NEAREST)
                mask_tensor = torch.from_numpy(np.asarray(mask) > 0).unsqueeze(0)
        else:
            mask_tensor = torch.zeros((1, self.size, self.size), dtype=torch.bool)
        return image_tensor, mask_tensor, label, image_path


def source_split(dataset, fraction, seed):
    groups = defaultdict(list)
    for i, (category, state, _, _) in enumerate(dataset.records):
        groups[(category, dataset.label(state))].append(i)
    rng = random.Random(seed)
    train, validation = [], []
    for key in sorted(groups):
        indices = groups[key]
        rng.shuffle(indices)
        n_val = min(len(indices) - 1, max(1, round(len(indices) * fraction)))
        validation.extend(indices[:n_val])
        train.extend(indices[n_val:])
    for partition in (train, validation):
        if {dataset.label(dataset.records[i][1]) for i in partition} != {0, 1}:
            raise ValueError("The source split must contain normal and anomalous images in both partitions.")
    return train, validation
