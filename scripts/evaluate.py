"""Ewaluacja zapisanego modelu na zbiorze testowym (nietykanym podczas treningu).

    python scripts/evaluate.py --config configs/resnet50.yaml \
        --checkpoint outputs/resnet50/best.pth --tsne
"""
import _bootstrap  # noqa: F401
import argparse
import os
import torch

from fungi import Config, set_seed
from fungi.data import build_dataloaders
from fungi.models import build_model
from fungi.metrics import (predict, full_report, report_by_source,
                           extract_embeddings, plot_tsne)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="test", choices=["val", "test", "train"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--tsne", action="store_true")
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)
    if args.device:
        cfg.device = args.device
    if cfg.device == "cuda" and not torch.cuda.is_available():
        cfg.device = "cpu"
    set_seed(cfg.seed, cfg.deterministic)  # ten sam seed -> ten sam podzial

    ckpt = torch.load(args.checkpoint, map_location=cfg.device, weights_only=False)
    class_names = ckpt["class_names"]

    data = build_dataloaders(cfg)
    assert data["class_names"] == class_names, "Niezgodnosc klas miedzy modelem a danymi!"

    model = build_model(cfg, len(class_names))
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(cfg.device).eval()

    out_dir = os.path.join(cfg.output_dir, cfg.experiment_name, "eval")
    os.makedirs(out_dir, exist_ok=True)

    loader = data["loaders"][args.split]
    y_true, y_pred, y_score = predict(model, loader, cfg.device)
    full_report(y_true, y_pred, y_score, class_names, out_dir, prefix=args.split)
    report_by_source(y_true, y_pred, data["dfs"][args.split]["source"].to_numpy(),
                     out_dir, prefix=args.split)

    if args.tsne:
        feats, labels = extract_embeddings(model, loader, cfg.device)
        plot_tsne(feats, labels, class_names,
                  os.path.join(out_dir, f"{args.split}_tsne.png"), seed=cfg.seed)
    print(f"Raport zapisany -> {out_dir}")


if __name__ == "__main__":
    main()
