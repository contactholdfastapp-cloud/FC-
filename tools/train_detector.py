"""Train the point detector on labelled frames, export ONNX, register the model.

    python tools/train_detector.py --data data/datasets/fc27 --epochs 60 --device cuda
    python tools/train_detector.py --data data/datasets/synth --epochs 5 --device cpu   (pipeline check)

Splits come from fctac.training.det_dataset (segment-level, no leakage).
The exported model is evaluated on the val split with the same metrics as the
baseline detector and registered as det_vNNN; it is only marked deployed if
it beats the currently deployed detector (fctac.training.registry).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from fctac.training.det_dataset import load_label, make_splits  # noqa: E402
from fctac.training.det_model import N_HM, TinyCenterNet, focal_loss  # noqa: E402

STRIDE = 4
MARKER_ABOVE = 1.15     # controlled-player marker centre, in player heights above the feet


def letterbox_params(w, h, in_w, in_h):
    s = in_w / w
    return s, int(round(h * s))


class DetDataset(Dataset):
    def __init__(self, paths, in_w=640, in_h=384, augment=False):
        self.paths = paths
        self.in_w, self.in_h = in_w, in_h
        self.aug = augment
        self.rng = np.random.default_rng(0)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        p = self.paths[i]
        img = cv2.imread(p)
        lab = load_label(p)
        h, w = img.shape[:2]
        s, rh = letterbox_params(w, h, self.in_w, self.in_h)
        x = np.zeros((self.in_h, self.in_w, 3), np.uint8)
        x[:min(rh, self.in_h)] = cv2.resize(img, (self.in_w, rh), interpolation=cv2.INTER_AREA)[:self.in_h]
        objs = lab["objects"]
        flip = self.aug and self.rng.random() < 0.5
        if flip:
            x = x[:, ::-1].copy()
        if self.aug:
            a = self.rng.uniform(0.75, 1.25)
            b = self.rng.uniform(-20, 20)
            x = np.clip(x.astype(np.float32) * a + b, 0, 255).astype(np.uint8)
        Ho, Wo = self.in_h // STRIDE, self.in_w // STRIDE
        hm = np.zeros((N_HM, Ho, Wo), np.float32)
        off = np.zeros((2, Ho, Wo), np.float32)
        hgt = np.zeros((1, Ho, Wo), np.float32)
        mask = np.zeros((1, Ho, Wo), np.float32)
        yy, xx = np.mgrid[0:Ho, 0:Wo]
        for o in objs:
            cx = o["x"] * s
            if flip:
                cx = self.in_w - 1 - cx
            cy = o["y"] * s
            ox, oy = cx / STRIDE, cy / STRIDE
            ix, iy = int(ox), int(oy)
            if not (0 <= ix < Wo and 0 <= iy < Ho):
                continue
            ph = max(o.get("h", 0.0) * s, 1.0)
            if o["cls"] == "ball":
                chans, sig = [1], 0.8
            else:
                chans, sig = [0], max(0.8, 0.12 * ph / STRIDE)
            g = np.exp(-((xx - ix) ** 2 + (yy - iy) ** 2) / (2 * sig * sig))
            for c in chans:
                hm[c] = np.maximum(hm[c], g)
                hm[c, iy, ix] = 1.0
            if o["cls"] != "ball" and o.get("controlled"):
                # the controlled cue is the marker above the head: target it directly
                mx, my = int(ox), int((cy - MARKER_ABOVE * ph) / STRIDE)
                if 0 <= mx < Wo and 0 <= my < Ho:
                    gm = np.exp(-((xx - mx) ** 2 + (yy - my) ** 2) / (2 * 1.0))
                    hm[2] = np.maximum(hm[2], gm)
                    hm[2, my, mx] = 1.0
            off[:, iy, ix] = (ox - ix, oy - iy)
            if o["cls"] != "ball":
                hgt[0, iy, ix] = np.log(ph)
            mask[0, iy, ix] = 1.0
        t = torch.from_numpy(x.transpose(2, 0, 1).astype(np.float32) / 255.0)
        return t, torch.from_numpy(hm), torch.from_numpy(off), torch.from_numpy(hgt), torch.from_numpy(mask)


def export(model, args, out_dir, val_focal=None):
    model.eval().cpu()
    onnx_path = os.path.join(out_dir, "detector.onnx")
    torch.onnx.export(model, torch.zeros(1, 3, args.height, args.width), onnx_path, input_names=["image"],
                      output_names=["hm", "off", "hgt"], opset_version=17, dynamo=False)
    meta = {"input": [args.height, args.width], "stride": STRIDE, "channels": ["player", "ball", "controlled_marker"],
            "marker_above": MARKER_ABOVE,
            "color": "bgr", "scale": 1 / 255.0, "data": args.data, "epochs": args.epochs, "val_focal": val_focal,
            "model_width": args.model_width}
    with open(onnx_path + ".json", "w") as f:
        json.dump(meta, f, indent=1)
    print("exported", onnx_path)
    return onnx_path


def train(args):
    splits = make_splits(args.data)
    out_dir = os.path.join("runs", "detector", args.name)
    if args.export_only:
        model = TinyCenterNet(args.model_width)
        model.load_state_dict(torch.load(os.path.join(out_dir, "best.pt")))
        return export(model, args, out_dir), splits
    tr = DetDataset(splits["train"], args.width, args.height, augment=True)
    va = DetDataset(splits["val"], args.width, args.height, augment=False)
    dl = DataLoader(tr, batch_size=args.batch, shuffle=True, num_workers=args.workers, drop_last=True)
    vl = DataLoader(va, batch_size=args.batch, shuffle=False, num_workers=args.workers)
    dev = torch.device(args.device)
    model = TinyCenterNet(args.model_width).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=max(1, args.epochs * len(dl)))
    os.makedirs(out_dir, exist_ok=True)
    best = (np.inf, None)
    for ep in range(args.epochs):
        model.train()
        t0 = time.time()
        tl = 0.0
        for x, hm, off, hgt, m in dl:
            x, hm, off, hgt, m = x.to(dev), hm.to(dev), off.to(dev), hgt.to(dev), m.to(dev)
            phm, poff, phgt = model(x)
            n = m.sum().clamp(min=1)
            loss = focal_loss(phm, hm) + (torch.abs(poff - off) * m).sum() / n \
                + 0.2 * (torch.abs(phgt - hgt) * (hgt > 0)).sum() / (hgt > 0).sum().clamp(min=1)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tl += float(loss)
        model.eval()
        vloss = 0.0
        with torch.no_grad():
            for x, hm, off, hgt, m in vl:
                phm, poff, phgt = model(x.to(dev))
                vloss += float(focal_loss(phm, hm.to(dev)))
        vloss /= max(len(vl), 1)
        print(f"epoch {ep + 1}/{args.epochs} train {tl / max(len(dl), 1):.3f} val {vloss:.3f} ({time.time() - t0:.0f}s)")
        if vloss < best[0]:
            best = (vloss, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
            torch.save(best[1], os.path.join(out_dir, "best.pt"))
    model.load_state_dict(best[1])
    return export(model, args, out_dir, best[0]), splits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--name", default="exp")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=4e-3)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=384)
    ap.add_argument("--model-width", type=float, default=1.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--no-register", action="store_true")
    ap.add_argument("--export-only", action="store_true", help="export runs/detector/<name>/best.pt without training")
    a = ap.parse_args()
    onnx_path, splits = train(a)
    if a.no_register:
        return
    from fctac.training.det_eval import evaluate_onnx_on_labels
    from fctac.training.registry import Registry
    m = evaluate_onnx_on_labels(onnx_path, splits["val"])
    print("val metrics", json.dumps(m))
    reg = Registry()
    entry = reg.register("detector", onnx_path, m, primary="det_score", data=a.data)
    print("registered", entry["version"], "deployed" if entry["deployed"] else "(not deployed: not better than current)")


if __name__ == "__main__":
    main()
