"""Modele: wlasny CNN (od zera) oraz backbone'y z timm (transfer learning).

`build_model` zwraca model gotowy do uzycia. Dla wszystkich modeli dziala
`model.get_classifier()` -> ostatnia warstwa liniowa (glowica). Uzywaja tego:
  * dwufazowy fine-tuning (zamrazanie/odmrazanie backbone'u),
  * ekstrakcja embeddingow do t-SNE (hook na wejscie glowicy).
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class CustomFungiCNN(nn.Module):
    """Wlasna siec konwolucyjna trenowana od zera.

    Rozwiniecie wersji z notatnika: dodany BatchNorm (stabilizuje i przyspiesza
    uczenie, czesto przelamuje plateau ~70%) oraz czwarty blok konwolucyjny.
    """

    def __init__(self, num_classes: int, dropout: float = 0.5):
        super().__init__()

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
            )

        self.features = nn.Sequential(
            block(3, 32), block(32, 64), block(64, 128), block(128, 256),
        )
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.flatten = nn.Flatten()
        self.pre_head = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(512, 128), nn.ReLU(inplace=True), nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward_features(self, x):
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = self.flatten(x)
        return self.pre_head(x)         # embedding (wejscie glowicy) -> uzywane w t-SNE

    def forward(self, x):
        return self.classifier(self.forward_features(x))

    def get_classifier(self) -> nn.Module:
        return self.classifier


def build_model(cfg, num_classes: int) -> nn.Module:
    if cfg.model_name == "custom_cnn":
        return CustomFungiCNN(num_classes)
    # dowolny model z timm; timm sam podmienia glowice na rozmiar num_classes
    return timm.create_model(cfg.model_name, pretrained=cfg.pretrained,
                             num_classes=num_classes)


def is_from_scratch(cfg) -> bool:
    return cfg.model_name == "custom_cnn" or not cfg.pretrained


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    """Zamraza/odmraza caly model poza glowica (get_classifier)."""
    for p in model.parameters():
        p.requires_grad = trainable
    clf = model.get_classifier() if hasattr(model, "get_classifier") else None
    if clf is not None:
        for p in clf.parameters():
            p.requires_grad = True


def trainable_parameters(model: nn.Module):
    return [p for p in model.parameters() if p.requires_grad]


def count_parameters(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def find_last_conv(model: nn.Module) -> nn.Module:
    """Ostatnia warstwa Conv2d -- domyslna warstwa docelowa dla Grad-CAM."""
    last = None
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            last = m
    if last is None:
        raise ValueError("Nie znaleziono warstwy Conv2d w modelu.")
    return last
