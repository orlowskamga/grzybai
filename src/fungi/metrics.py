"""Ewaluacja i wizualizacje.

Komplet metryk, ktore obiecuje dokumentacja: accuracy, precision, recall,
F1 (per klasa + makro/wazone), macierz pomylek, krzywe ROC i AUC, oraz
t-SNE cech. Dodatkowo `report_by_source` -- zabezpieczenie przy wspolnym
klasyfikatorze (czy model nie rozpoznaje zrodla zamiast morfologii).
"""
from __future__ import annotations
import json
import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import cycle
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_curve, auc, f1_score)
from sklearn.preprocessing import label_binarize


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    y_true, y_pred, y_score = [], [], []
    for x, y in loader:
        x = x.to(device)
        out = model(x)
        score = F.softmax(out, dim=1)
        y_true.append(y.numpy())
        y_pred.append(out.argmax(1).cpu().numpy())
        y_score.append(score.cpu().numpy())
    return (np.concatenate(y_true), np.concatenate(y_pred), np.concatenate(y_score))


def plot_confusion_matrix(y_true, y_pred, class_names, out_path, normalize=False):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    fmt = "d"
    if normalize:
        cm = cm.astype(float) / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
        fmt = ".2f"
    plt.figure(figsize=(max(8, len(class_names)), max(7, len(class_names) - 1)))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Przewidziana klasa")
    plt.ylabel("Rzeczywista klasa")
    plt.title("Macierz pomylek" + (" (znormalizowana)" if normalize else ""))
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_roc(y_true, y_score, class_names, out_path):
    n = len(class_names)
    y_bin = label_binarize(y_true, classes=range(n))
    if n == 2:
        y_bin = np.hstack([1 - y_bin, y_bin])
    plt.figure(figsize=(10, 8))
    colors = cycle(["aqua", "darkorange", "cornflowerblue", "green", "red",
                    "purple", "brown", "magenta", "gray", "olive", "teal"])
    aucs = {}
    for i, color in zip(range(n), colors):
        if y_bin[:, i].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        aucs[class_names[i]] = float(roc_auc)
        plt.plot(fpr, tpr, color=color, lw=2,
                 label=f"{class_names[i]} (AUC={roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("Krzywe ROC (one-vs-rest)")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return aucs


def full_report(y_true, y_pred, y_score, class_names, out_dir, prefix="test"):
    os.makedirs(out_dir, exist_ok=True)
    rep = classification_report(y_true, y_pred, labels=range(len(class_names)),
                                target_names=class_names, output_dict=True, zero_division=0)
    rep_txt = classification_report(y_true, y_pred, labels=range(len(class_names)),
                                    target_names=class_names, zero_division=0)
    print(rep_txt)

    plot_confusion_matrix(y_true, y_pred, class_names,
                          os.path.join(out_dir, f"{prefix}_confusion_matrix.png"))
    plot_confusion_matrix(y_true, y_pred, class_names,
                          os.path.join(out_dir, f"{prefix}_confusion_matrix_norm.png"),
                          normalize=True)
    aucs = plot_roc(y_true, y_score, class_names,
                    os.path.join(out_dir, f"{prefix}_roc.png"))

    summary = {
        "accuracy": float((y_true == y_pred).mean()),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "roc_auc_per_class": aucs,
        "roc_auc_macro": float(np.mean(list(aucs.values()))) if aucs else None,
        "classification_report": rep,
    }
    with open(os.path.join(out_dir, f"{prefix}_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(os.path.join(out_dir, f"{prefix}_report.txt"), "w", encoding="utf-8") as f:
        f.write(rep_txt)
    return summary


def report_by_source(y_true, y_pred, sources, out_dir, prefix="test"):
    """Accuracy/macro-F1 w rozbiciu na zrodlo (defungi/openfungi).

    Duza dysproporcja jakosci miedzy zrodlami to sygnal, ze model moze
    korzystac z cech specyficznych dla zbioru, a nie z morfologii.
    """
    sources = np.asarray(sources)
    out = {}
    for src in np.unique(sources):
        m = sources == src
        if m.sum() == 0:
            continue
        out[str(src)] = {
            "n": int(m.sum()),
            "accuracy": float((y_true[m] == y_pred[m]).mean()),
            "macro_f1": float(f1_score(y_true[m], y_pred[m], average="macro", zero_division=0)),
        }
    with open(os.path.join(out_dir, f"{prefix}_by_source.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("Wyniki per zrodlo:", json.dumps(out, indent=2, ensure_ascii=False))
    return out


def extract_embeddings(model, loader, device):
    """Embeddingi z wejscia glowicy (hook na get_classifier) -> do t-SNE."""
    clf = model.get_classifier() if hasattr(model, "get_classifier") else getattr(model, "fc", None)
    captured = {}

    def pre_hook(module, args):
        captured["x"] = args[0].detach()

    h = clf.register_forward_pre_hook(pre_hook)
    feats, labels = [], []
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            model(x)
            f = captured["x"]
            if f.dim() > 2:
                f = torch.flatten(f, 1)
            feats.append(f.cpu().numpy())
            labels.append(y.numpy())
    h.remove()
    return np.concatenate(feats), np.concatenate(labels)


def plot_tsne(features, labels, class_names, out_path, seed=42):
    from sklearn.manifold import TSNE
    perplexity = min(30, max(5, len(features) // 4))
    tsne = TSNE(n_components=2, random_state=seed, perplexity=perplexity, init="pca")
    emb = tsne.fit_transform(features)
    plt.figure(figsize=(11, 9))
    sns.scatterplot(x=emb[:, 0], y=emb[:, 1],
                    hue=[class_names[l] for l in labels],
                    palette=sns.color_palette("hsv", len(class_names)),
                    legend="full", alpha=0.8, s=18)
    plt.title("t-SNE cech (wejscie glowicy)")
    plt.xlabel("t-SNE 1"); plt.ylabel("t-SNE 2")
    plt.legend(title="Klasa", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
