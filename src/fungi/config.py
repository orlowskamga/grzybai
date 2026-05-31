"""Konfiguracja eksperymentu wczytywana z pliku YAML.

Jeden plik konfiguracyjny = jeden eksperyment. Zmieniajac wylacznie config
(np. model_name z 'resnet50' na 'efficientnet_b0') uruchamiasz porownywalny
przebieg na identycznym pipeline danych i ewaluacji.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict, fields
from typing import Tuple
import yaml


@dataclass
class Config:
    # --- dane ---
    manifest: str = "data/manifest.csv"
    img_size: int = 224
    batch_size: int = 32
    num_workers: int = 4
    split_ratios: Tuple[float, float, float] = (0.70, 0.15, 0.15)
    group_aware_split: bool = True

    # --- model ---
    model_name: str = "resnet50"
    pretrained: bool = True

    # --- trening ---
    epochs: int = 25
    lr: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: str = "adam"
    momentum: float = 0.9
    scheduler: str = "cosine"
    step_size: int = 7
    gamma: float = 0.1
    early_stop_patience: int = 8

    # --- transfer learning (dwufazowy) ---
    freeze_backbone: bool = True
    finetune_epochs: int = 10
    finetune_lr: float = 1e-4

    # --- niezbalansowanie klas ---
    imbalance: str = "class_weights"

    # --- augmentacja / normalizacja ---
    augment: bool = True
    normalize: str = "imagenet"

    # --- rozne ---
    seed: int = 42
    deterministic: bool = True
    device: str = "cuda"
    output_dir: str = "outputs"
    experiment_name: str = "exp"

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        valid = {f.name for f in fields(cls)}
        unknown = set(data) - valid
        if unknown:
            print(f"[config] Pomijam nieznane klucze: {sorted(unknown)}")
        known = {k: v for k, v in data.items() if k in valid}
        if "split_ratios" in known and isinstance(known["split_ratios"], list):
            known["split_ratios"] = tuple(known["split_ratios"])
        return cls(**known)

    def to_yaml(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(asdict(self), f, allow_unicode=True, sort_keys=False)

    def to_dict(self) -> dict:
        return asdict(self)
