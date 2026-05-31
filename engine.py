"""Silnik treningu: petla ucząca z dwufazowym fine-tuningiem i wyborem
najlepszego modelu wg macro-F1 na zbiorze walidacyjnym.

Dlaczego macro-F1, a nie accuracy? Przy niezbalansowanych klasach (i przy
laczeniu latwego OpenFungi z trudniejszym DeFungi) accuracy bywa zawyzana
przez klasy liczne/latwe. Macro-F1 traktuje kazda klase rownorzednie.
"""
from __future__ import annotations
import copy
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score

from .models import set_backbone_trainable, trainable_parameters, is_from_scratch, count_parameters


def build_optimizer(params, cfg, lr: float):
    if cfg.optimizer == "sgd":
        return optim.SGD(params, lr=lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    if cfg.optimizer == "adamw":
        return optim.AdamW(params, lr=lr, weight_decay=cfg.weight_decay)
    return optim.Adam(params, lr=lr, weight_decay=cfg.weight_decay)


def build_scheduler(optimizer, cfg, epochs: int):
    if cfg.scheduler == "step":
        return optim.lr_scheduler.StepLR(optimizer, step_size=cfg.step_size, gamma=cfg.gamma)
    if cfg.scheduler == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    return None


@torch.no_grad()
def _evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum, n = 0.0, 0
    y_true, y_pred = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = criterion(out, y)
        loss_sum += loss.item() * x.size(0)
        n += x.size(0)
        y_true.append(y.cpu().numpy())
        y_pred.append(out.argmax(1).cpu().numpy())
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    acc = float((y_true == y_pred).mean())
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    return loss_sum / max(n, 1), acc, macro_f1


def _train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    loss_sum, n, correct = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * x.size(0)
        n += x.size(0)
        correct += (out.argmax(1) == y).sum().item()
    return loss_sum / max(n, 1), correct / max(n, 1)


def _run_phase(model, loaders, criterion, optimizer, scheduler, device,
               epochs, history, best, patience, phase_name):
    epochs_no_improve = 0
    for epoch in range(epochs):
        tr_loss, tr_acc = _train_one_epoch(model, loaders["train"], criterion, optimizer, device)
        va_loss, va_acc, va_f1 = _evaluate(model, loaders["val"], criterion, device)
        if scheduler is not None:
            scheduler.step()

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)
        history["val_macro_f1"].append(va_f1)
        history["phase"].append(phase_name)

        improved = va_f1 > best["macro_f1"] + 1e-5
        flag = ""
        if improved:
            best["macro_f1"] = va_f1
            best["acc"] = va_acc
            best["weights"] = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            flag = "  <-- best"
        else:
            epochs_no_improve += 1

        print(f"[{phase_name}] epoka {epoch + 1}/{epochs}  "
              f"train_loss={tr_loss:.4f} acc={tr_acc:.4f} | "
              f"val_loss={va_loss:.4f} acc={va_acc:.4f} macroF1={va_f1:.4f}{flag}")

        if patience and epochs_no_improve >= patience:
            print(f"[{phase_name}] wczesne zatrzymanie (brak poprawy przez {patience} epok)")
            break


def fit(model, loaders, cfg, class_weights=None, device=None):
    device = device or cfg.device
    model = model.to(device)

    weight = class_weights.to(device) if class_weights is not None else None
    criterion = nn.CrossEntropyLoss(weight=weight)

    history = {k: [] for k in
               ["train_loss", "train_acc", "val_loss", "val_acc", "val_macro_f1", "phase"]}
    best = {"macro_f1": -1.0, "acc": 0.0, "weights": copy.deepcopy(model.state_dict())}

    since = time.time()

    if is_from_scratch(cfg):
        # jedna faza: trenujemy caly model
        print(f"Trening od zera ({cfg.model_name}); parametry: {count_parameters(model)}")
        opt = build_optimizer(trainable_parameters(model), cfg, cfg.lr)
        sch = build_scheduler(opt, cfg, cfg.epochs)
        _run_phase(model, loaders, criterion, opt, sch, device,
                   cfg.epochs, history, best, cfg.early_stop_patience, "scratch")
    else:
        # FAZA 1: glowica na zamrozonym backbonie
        if cfg.freeze_backbone:
            set_backbone_trainable(model, trainable=False)
            print(f"Faza 1 (glowica). Parametry: {count_parameters(model)}")
            opt = build_optimizer(trainable_parameters(model), cfg, cfg.lr)
            sch = build_scheduler(opt, cfg, cfg.epochs)
            _run_phase(model, loaders, criterion, opt, sch, device,
                       cfg.epochs, history, best, cfg.early_stop_patience, "head")
        # FAZA 2: odmrozenie calego modelu, maly LR
        if cfg.finetune_epochs > 0:
            set_backbone_trainable(model, trainable=True)
            if best["weights"] is not None:
                model.load_state_dict(best["weights"])  # rusz z najlepszej glowicy
            print(f"Faza 2 (fine-tuning calosci). Parametry: {count_parameters(model)}")
            opt = build_optimizer(trainable_parameters(model), cfg, cfg.finetune_lr)
            sch = build_scheduler(opt, cfg, cfg.finetune_epochs)
            _run_phase(model, loaders, criterion, opt, sch, device,
                       cfg.finetune_epochs, history, best, cfg.early_stop_patience, "finetune")

    elapsed = time.time() - since
    print(f"Trening zakonczony w {elapsed // 60:.0f}m {elapsed % 60:.0f}s | "
          f"najlepszy val macro-F1 = {best['macro_f1']:.4f} (acc={best['acc']:.4f})")

    model.load_state_dict(best["weights"])
    return model, history, best
