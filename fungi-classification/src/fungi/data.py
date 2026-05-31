"""Dane: manifest -> podzial -> DataLoadery.

Najwazniejsze decyzje projektowe sa tutaj:

1. Podzial grupowo-stratyfikowany. DeFungi to patche wycinane z wiekszych
   preparatow. Losowy podzial po patchu rozsmarowuje jeden preparat po
   zbiorze treningowym i testowym (przeciek -> zawyzone wyniki). Dzielimy
   wiec po kolumnie `group` (id preparatu/obrazu zrodlowego), zachowujac
   proporcje klas. Grupa nigdy nie trafia jednoczesnie do dwoch zbiorow.

2. Trzy zbiory: train / val / test. `val` sluzy tylko do wyboru najlepszego
   modelu, `test` jest nietykany do finalnej ewaluacji.

3. Niezbalansowanie klas: wagi w funkcji straty albo WeightedRandomSampler.

4. Kolumna `source` (defungi/openfungi) jest przenoszona dalej -- pozwala
   pozniej sprawdzic, czy model nie rozpoznaje *zrodla* zamiast morfologii
   (zob. scripts/evaluate.py -> raport per source).
"""
from __future__ import annotations
from typing import Dict, Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image

from .seed import seed_worker, make_generator

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
HALF_MEAN = (0.5, 0.5, 0.5)
HALF_STD = (0.5, 0.5, 0.5)


def _norm_stats(normalize: str) -> Tuple[tuple, tuple]:
    return (HALF_MEAN, HALF_STD) if normalize == "half" else (IMAGENET_MEAN, IMAGENET_STD)


def build_transforms(img_size: int, train: bool, augment: bool, normalize: str):
    mean, std = _norm_stats(normalize)
    if train and augment:
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),       # mikroskopia nie ma "gory" -- odbicie pionowe jest sensowne
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


class ManifestDataset(Dataset):
    """Dataset oparty na DataFrame z kolumnami: path, label_idx."""

    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        img = Image.open(row["path"]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, int(row["label_idx"])


def grouped_stratified_split(df: pd.DataFrame, ratios=(0.7, 0.15, 0.15), seed=42,
                             group_col="group", label_col="unified_label") -> pd.DataFrame:
    """Dodaje kolumne `split` (train/val/test). Cala grupa trafia do jednego
    zbioru; podzial jest stratyfikowany wzgledem (dominujacej) klasy grupy."""
    assert abs(sum(ratios) - 1.0) < 1e-6, "ratios musza sumowac sie do 1"
    g2l = (df.groupby(group_col)[label_col]
             .agg(lambda x: x.value_counts().idxmax())
             .reset_index())
    rng = np.random.default_rng(seed)
    split_of_group: Dict[str, str] = {}
    for _, sub in g2l.groupby(label_col):
        groups = sub[group_col].tolist()
        rng.shuffle(groups)
        n = len(groups)
        n_tr = int(round(ratios[0] * n))
        n_va = int(round(ratios[1] * n))
        # gwarancja niepustego val/test, gdy klasa ma >= 3 grupy
        if n >= 3:
            n_tr = min(n_tr, n - 2)
            n_va = max(n_va, 1)
        for g in groups[:n_tr]:
            split_of_group[g] = "train"
        for g in groups[n_tr:n_tr + n_va]:
            split_of_group[g] = "val"
        for g in groups[n_tr + n_va:]:
            split_of_group[g] = "test"
    out = df.copy()
    out["split"] = out[group_col].map(split_of_group)
    return out


def compute_class_weights(df_train: pd.DataFrame, n_classes: int) -> torch.Tensor:
    """Wagi odwrotnie proporcjonalne do licznosci klas (do CrossEntropyLoss)."""
    counts = df_train["label_idx"].value_counts().to_dict()
    total = len(df_train)
    w = torch.zeros(n_classes, dtype=torch.float32)
    for idx in range(n_classes):
        c = counts.get(idx, 0)
        w[idx] = total / (n_classes * c) if c > 0 else 0.0
    return w


def _make_sampler(df_train: pd.DataFrame, n_classes: int, generator) -> WeightedRandomSampler:
    class_w = compute_class_weights(df_train, n_classes)
    sample_w = df_train["label_idx"].map(lambda i: float(class_w[i])).to_numpy()
    return WeightedRandomSampler(weights=torch.from_numpy(sample_w).double(),
                                 num_samples=len(df_train), replacement=True,
                                 generator=generator)


def load_manifest(cfg) -> Tuple[pd.DataFrame, list, dict]:
    df = pd.read_csv(cfg.manifest)
    required = {"path", "unified_label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Manifest nie zawiera kolumn: {missing}")
    if "group" not in df.columns:
        df["group"] = df["path"]            # brak grupowania -> kazdy obraz osobno
    if "source" not in df.columns:
        df["source"] = "unknown"
    class_names = sorted(df["unified_label"].unique().tolist())
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    df["label_idx"] = df["unified_label"].map(class_to_idx)
    return df, class_names, class_to_idx


def build_dataloaders(cfg) -> dict:
    df, class_names, class_to_idx = load_manifest(cfg)
    n_classes = len(class_names)

    group_col = "group" if cfg.group_aware_split else "path"
    df = grouped_stratified_split(df, ratios=tuple(cfg.split_ratios), seed=cfg.seed,
                                  group_col=group_col)

    dfs = {s: df[df["split"] == s].copy() for s in ("train", "val", "test")}
    generator = make_generator(cfg.seed)

    tf_train = build_transforms(cfg.img_size, True, cfg.augment, cfg.normalize)
    tf_eval = build_transforms(cfg.img_size, False, False, cfg.normalize)
    datasets = {
        "train": ManifestDataset(dfs["train"], tf_train),
        "val": ManifestDataset(dfs["val"], tf_eval),
        "test": ManifestDataset(dfs["test"], tf_eval),
    }

    use_sampler = cfg.imbalance == "weighted_sampler"
    sampler = _make_sampler(dfs["train"], n_classes, generator) if use_sampler else None

    loaders = {}
    for split, ds in datasets.items():
        is_train = split == "train"
        loaders[split] = DataLoader(
            ds, batch_size=cfg.batch_size,
            shuffle=(is_train and sampler is None),
            sampler=sampler if is_train else None,
            num_workers=cfg.num_workers, pin_memory=True,
            worker_init_fn=seed_worker, generator=generator,
            drop_last=False,
        )

    class_weights = (compute_class_weights(dfs["train"], n_classes)
                     if cfg.imbalance == "class_weights" else None)

    return {
        "loaders": loaders,
        "datasets": datasets,
        "dfs": dfs,
        "class_names": class_names,
        "class_to_idx": class_to_idx,
        "n_classes": n_classes,
        "class_weights": class_weights,
    }
