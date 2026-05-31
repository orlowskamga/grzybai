"""Trenuje model wg konfiguracji.

    python scripts/train.py --config configs/resnet50.yaml
    python scripts/train.py --config configs/customcnn.yaml --device cpu
"""
import _bootstrap  # noqa: F401
import argparse
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from fungi import Config, set_seed
from fungi.data import build_dataloaders
from fungi.models import build_model
from fungi.engine import fit


def plot_history(history, out_path):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 5))
    a1.plot(history["train_loss"], label="train")
    a1.plot(history["val_loss"], label="val")
    a1.set_title("Strata"); a1.set_xlabel("epoka"); a1.legend()
    a2.plot(history["train_acc"], label="train acc")
    a2.plot(history["val_acc"], label="val acc")
    a2.plot(history["val_macro_f1"], label="val macro-F1")
    a2.set_title("Dokladnosc / F1"); a2.set_xlabel("epoka"); a2.legend()
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()


def resolve_device(requested):
    if requested == "cuda" and not torch.cuda.is_available():
        print("[device] CUDA niedostepna -- przechodze na CPU")
        return "cpu"
    return requested


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default=None, help="nadpisuje device z configu")
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)
    if args.device:
        cfg.device = args.device
    cfg.device = resolve_device(cfg.device)

    set_seed(cfg.seed, cfg.deterministic)
    out_dir = os.path.join(cfg.output_dir, cfg.experiment_name)
    os.makedirs(out_dir, exist_ok=True)
    cfg.to_yaml(os.path.join(out_dir, "config.yaml"))

    data = build_dataloaders(cfg)
    print(f"Klasy ({data['n_classes']}): {data['class_names']}")
    for s in ("train", "val", "test"):
        print(f"  {s}: {len(data['dfs'][s])} obrazow")

    model = build_model(cfg, data["n_classes"])
    model, history, best = fit(model, data["loaders"], cfg,
                               class_weights=data["class_weights"], device=cfg.device)

    torch.save({
        "state_dict": model.state_dict(),
        "class_names": data["class_names"],
        "cfg": cfg.to_dict(),
        "best": {"macro_f1": best["macro_f1"], "acc": best["acc"]},
    }, os.path.join(out_dir, "best.pth"))
    with open(os.path.join(out_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    plot_history(history, os.path.join(out_dir, "training_curves.png"))
    print(f"Zapisano model i wyniki -> {out_dir}")


if __name__ == "__main__":
    main()
