"""Prosty interfejs testowy (Gradio): wczytaj obraz mikroskopowy -> predykcja
+ mapa Grad-CAM. Spelnia wymaganie dokumentacji ("prosty interfejs testowy").

    python app.py --checkpoint outputs/resnet50/best.pth --config configs/resnet50.yaml
"""
import os
import sys
import argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
import torch
import gradio as gr
from PIL import Image

from fungi import Config
from fungi.data import build_transforms
from fungi.models import build_model
from fungi.explain import GradCAM, denormalize, overlay_cam


def load(checkpoint, config, device):
    cfg = Config.from_yaml(config)
    cfg.device = device
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    class_names = ckpt["class_names"]
    model = build_model(cfg, len(class_names))
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device).eval()
    tf = build_transforms(cfg.img_size, train=False, augment=False, normalize=cfg.normalize)
    return model, class_names, tf, cfg


def make_predict_fn(model, class_names, tf, cfg, device):
    def predict(image: Image.Image):
        x = tf(image.convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(x), dim=1)[0].cpu().numpy()
        topk = {class_names[i]: float(probs[i]) for i in probs.argsort()[::-1][:5]}

        cam_engine = GradCAM(model)
        cam, _ = cam_engine(x)
        cam_engine.remove()
        rgb = denormalize(x[0], cfg.normalize)
        overlay = (overlay_cam(rgb, cam[0]) * 255).astype(np.uint8)
        return topk, Image.fromarray(overlay)
    return predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()

    model, class_names, tf, cfg = load(args.checkpoint, args.config, args.device)
    fn = make_predict_fn(model, class_names, tf, cfg, args.device)

    demo = gr.Interface(
        fn=fn,
        inputs=gr.Image(type="pil", label="Obraz mikroskopowy"),
        outputs=[gr.Label(num_top_classes=5, label="Predykcja"),
                 gr.Image(label="Grad-CAM (na co patrzy model)")],
        title="Klasyfikacja mikroskopowych obrazow grzybow",
        description="Wczytaj obraz preparatu. Model zwraca 5 najbardziej prawdopodobnych klas oraz mape Grad-CAM.",
    )
    demo.launch(share=args.share)


if __name__ == "__main__":
    main()
