"""Grad-CAM -- mapy aktywacji pokazujace, na ktore fragmenty obrazu patrzy
model. Implementacja recznie (hooki forward/backward), bez zewnetrznych
bibliotek -- jest odporna i przenosna (rozwiazuje problem z notatnika, gdzie
biblioteka Grad-CAM "nie chciala dzialac").

Wartosc mykologiczna: pozwala sprawdzic, czy model patrzy na struktury
diagnostyczne (zarodniki, strzepki), czy na tlo/artefakty preparatu.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F

from .data import _norm_stats
from .models import find_last_conv


class GradCAM:
    def __init__(self, model, target_layer=None):
        self.model = model.eval()
        self.target = target_layer or find_last_conv(model)
        self.activations = None
        self.gradients = None
        self._fwd = self.target.register_forward_hook(self._save_act)
        self._bwd = self.target.register_full_backward_hook(self._save_grad)

    def _save_act(self, module, inp, out):
        self.activations = out.detach()

    def _save_grad(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def __call__(self, x, class_idx=None):
        out = self.model(x)
        if class_idx is None:
            class_idx = out.argmax(dim=1)
        elif isinstance(class_idx, int):
            class_idx = torch.full((x.size(0),), class_idx, device=x.device, dtype=torch.long)
        score = out.gather(1, class_idx.view(-1, 1)).sum()
        self.model.zero_grad()
        score.backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1))
        cam = F.interpolate(cam.unsqueeze(1), size=x.shape[-2:],
                            mode="bilinear", align_corners=False).squeeze(1)
        cam = cam - cam.amin(dim=(1, 2), keepdim=True)
        cam = cam / (cam.amax(dim=(1, 2), keepdim=True) + 1e-8)
        return cam.detach().cpu().numpy(), class_idx.detach().cpu().numpy()

    def remove(self):
        self._fwd.remove()
        self._bwd.remove()


def denormalize(x: torch.Tensor, normalize: str = "imagenet") -> np.ndarray:
    """Tensor (C,H,W) -> obraz RGB [0,1] (H,W,C) do nalozenia mapy."""
    mean, std = _norm_stats(normalize)
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    img = (x.detach().cpu() * std + mean).clamp(0, 1)
    return img.permute(1, 2, 0).numpy()


def overlay_cam(rgb01: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    import matplotlib.cm as cm
    heat = cm.jet(cam)[..., :3]
    return np.clip((1 - alpha) * rgb01 + alpha * heat, 0, 1)
