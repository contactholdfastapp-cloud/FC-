"""TinyRadarNet: heatmap CNN that reads the FC 27 radar (training + export).

    python tools/train_radar.py --synth data/radar/synth --real data/radar/real_pseudo --epochs 12

Input: canonical 320x192 RGB radar crop in [0, 1].  Output: 4 sigmoid
heatmaps at stride 2 (triangle team, circle team, ball, highlighted player).
Fully convolutional, so training uses random patches (much cheaper on CPU)
and inference the full crop.  ~0.15 GMAC per crop.
"""
from __future__ import annotations

import json
import os
import time
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fctac.vision.radar_fc27 import CH, CH_BALL, CH_CIR, CH_HL, CH_TRI, CW, STRIDE, decode

OUT_H, OUT_W = CH // STRIDE, CW // STRIDE


def cbr(cin, cout, k=3, s=1, d=1):
    return nn.Sequential(nn.Conv2d(cin, cout, k, s, padding=d * (k // 2), dilation=d, bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


def dwsep(c, cout=None):
    cout = cout or c
    return nn.Sequential(nn.Conv2d(c, c, 3, 1, 1, groups=c, bias=False), nn.BatchNorm2d(c), nn.ReLU(inplace=True),
                         nn.Conv2d(c, cout, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class TinyRadarNet(nn.Module):
    def __init__(self, c0=8, c1=16, c2=32, cm=24, out=4):
        super().__init__()
        self.stem0 = cbr(3, c0, 3, 1)
        self.stem1 = cbr(c0, c1, 3, 2)            # /2
        self.b1 = dwsep(c1)
        self.down = cbr(c1, c2, 3, 2)             # /4
        self.b2 = cbr(c2, c2, 3)
        self.b3 = cbr(c2, c2, 3, d=2)
        self.merge = cbr(c1 + c2, cm, 1)
        self.b4 = dwsep(cm)
        self.head = nn.Conv2d(cm, out, 1)
        nn.init.constant_(self.head.bias, -4.0)
        self.register_buffer("mean", torch.tensor([0.30, 0.40, 0.25]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.20, 0.20, 0.20]).view(1, 3, 1, 1))

    def forward(self, x):               # x: N,3,H,W RGB in [0,1] -> logits N,4,H/2,W/2
        x = (x - self.mean) / self.std
        a = self.b1(self.stem1(self.stem0(x)))
        b = self.b3(self.b2(self.down(a)))
        u = F.interpolate(b, size=a.shape[-2:], mode="nearest")
        return self.head(self.b4(self.merge(torch.cat([a, u], 1))))


class Exported(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        return torch.sigmoid(self.net(x))


# ----------------------------------------------------------------------------- data
def load_labels(root: str) -> list:
    rows = [json.loads(l) for l in open(os.path.join(root, "labels.jsonl"))]
    for r in rows:
        r["path"] = os.path.join(root, "img", r.get("file", f"{r['id']:06d}.jpg"))
    return rows


def gaussian_targets(players: list, ball, h: int, w: int, sigma: float = 1.0, x0: float = 0.0, y0: float = 0.0):
    """Heatmap targets (4, h, w) at stride 2 for a patch whose canonical origin is (x0, y0)."""
    t = np.zeros((4, h, w), np.float32)
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)

    def put(ch, u, v):
        cx = (u - x0 + 0.5) / STRIDE - 0.5
        cy = (v - y0 + 0.5) / STRIDE - 0.5
        if not (-2 <= cx < w + 2 and -2 <= cy < h + 2):
            return
        r = int(3 * sigma + 1)
        ix, iy = int(round(cx)), int(round(cy))
        xa, xb, ya, yb = max(0, ix - r), min(w, ix + r + 1), max(0, iy - r), min(h, iy + r + 1)
        if xa >= xb or ya >= yb:
            return
        g = np.exp(-((xs[ya:yb, xa:xb] - cx) ** 2 + (ys[ya:yb, xa:xb] - cy) ** 2) / (2 * sigma * sigma))
        np.maximum(t[ch, ya:yb, xa:xb], g, out=t[ch, ya:yb, xa:xb])
        if 0 <= ix < w and 0 <= iy < h:
            t[ch, iy, ix] = 1.0

    for u, v, s, hl in players:
        put(CH_TRI if s == 0 else CH_CIR, u, v)
        if hl:
            put(CH_HL, u, v)
    if ball is not None:
        put(CH_BALL, ball[0], ball[1])
    return t


class RadarDataset(torch.utils.data.Dataset):
    def __init__(self, rows: list, patch: Optional[tuple] = (128, 192), train: bool = True, seed: int = 0):
        self.rows = rows
        self.patch = patch
        self.train = train
        self.seed = seed

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        rng = np.random.default_rng((self.seed * 1000003 + i * 7919 + int(time.time() * 1000) % 100000)
                                    if self.train else i)
        img = cv2.imread(r["path"])
        if img.shape[:2] != (CH, CW):
            img = cv2.resize(img, (CW, CH), interpolation=cv2.INTER_AREA)
        players = [list(p) for p in r["players"]]
        ball = None if r["ball"] is None else list(r["ball"])
        x0 = y0 = 0
        if self.train:
            if rng.random() < 0.5:                       # horizontal flip (triangles stay upright)
                img = img[:, ::-1]
                players = [[CW - 1 - u, v, s, h] for u, v, s, h in players]
                ball = None if ball is None else [CW - 1 - ball[0], ball[1]]
            if rng.random() < 0.15:                      # colour-agnostic shapes
                img = img[:, :, rng.permutation(3)]
            if rng.random() < 0.3:
                hsv = cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_BGR2HSV).astype(np.int16)
                hsv[..., 0] = (hsv[..., 0] + int(rng.integers(-12, 13))) % 180
                hsv[..., 1] = np.clip(hsv[..., 1] * rng.uniform(0.7, 1.3), 0, 255)
                img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
            if self.patch is not None:
                ph, pw = self.patch
                y0 = int(rng.integers(0, (CH - ph) // STRIDE + 1)) * STRIDE
                x0 = int(rng.integers(0, (CW - pw) // STRIDE + 1)) * STRIDE
                img = img[y0:y0 + ph, x0:x0 + pw]
        img = np.ascontiguousarray(img[:, :, ::-1]).astype(np.float32) / 255.0
        h, w = img.shape[0] // STRIDE, img.shape[1] // STRIDE
        t = gaussian_targets(players, ball, h, w, x0=x0, y0=y0)
        mask = np.ones((1, h, w), np.float32)
        if "ignore" in r:                                # real pseudo-labels: unsure regions
            for u, v, rad in r["ignore"]:
                cx, cy = (u - x0) / STRIDE, (v - y0) / STRIDE
                rr = rad / STRIDE
                ya, yb = int(max(0, cy - rr)), int(min(h, cy + rr + 1))
                xa, xb = int(max(0, cx - rr)), int(min(w, cx + rr + 1))
                mask[:, ya:yb, xa:xb] = 0
        if r.get("no_hl"):                               # highlight channel not labelled
            mask = np.repeat(mask, 4, 0)
            mask[CH_HL] = 0
        return torch.from_numpy(img.transpose(2, 0, 1)), torch.from_numpy(t), torch.from_numpy(mask)


def focal_loss(logits, target, mask, alpha=2.0, beta=4.0):
    p = torch.sigmoid(logits).clamp(1e-4, 1 - 1e-4)
    pos = (target >= 0.999).float()
    neg = 1.0 - pos
    lpos = -torch.log(p) * (1 - p) ** alpha * pos
    lneg = -torch.log(1 - p) * p ** alpha * (1 - target) ** beta * neg
    mask = mask.expand_as(logits)
    n = (pos * mask).sum().clamp(min=1.0)
    return ((lpos + lneg) * mask).sum() / n


# ----------------------------------------------------------------------------- evaluation
def match_points(pred: np.ndarray, gt: np.ndarray, tol: float):
    """Greedy nearest matching -> list of (i_pred, j_gt, dist)."""
    if len(pred) == 0 or len(gt) == 0:
        return []
    d = np.hypot(pred[:, None, 0] - gt[None, :, 0], pred[:, None, 1] - gt[None, :, 1])
    out = []
    used_p, used_g = set(), set()
    for k in np.argsort(d, axis=None):
        i, j = divmod(int(k), d.shape[1])
        if d[i, j] > tol:
            break
        if i in used_p or j in used_g:
            continue
        used_p.add(i)
        used_g.add(j)
        out.append((i, j, float(d[i, j])))
    return out


def evaluate(predict, rows: list, tol: float = 3.0, ball_tol: float = 4.0, thr: float = 0.35) -> dict:
    """predict(img_bgr) -> heat (4, OUT_H, OUT_W).  Exact-label metrics on full crops."""
    tp = fp = fn = 0
    shape_ok = 0
    errs = []
    btp = bfp = bfn = 0
    berr = []
    htp = hfp = hfn = 0
    t_ms = []
    for r in rows:
        img = cv2.imread(r["path"])
        t0 = time.perf_counter()
        heat = predict(img)
        t_ms.append((time.perf_counter() - t0) * 1000)
        d = decode(heat, thr)
        gt = np.array([[p[0], p[1]] for p in r["players"]]).reshape(-1, 2)
        gs = np.array([p[2] for p in r["players"]], int)
        gh = np.array([p[3] for p in r["players"]], bool)
        # ignore GT symbols outside the canonical crop (cannot be predicted)
        inside = (gt[:, 0] >= 0) & (gt[:, 0] < CW) & (gt[:, 1] >= 0) & (gt[:, 1] < CH) if len(gt) else np.zeros(0, bool)
        m = match_points(d["uv"], gt, tol)
        mp = {i for i, _, _ in m}
        mg = {j for _, j, _ in m}
        tp += len(m)
        fp += len(d["uv"]) - len(mp)
        fn += int(sum(1 for j in range(len(gt)) if j not in mg and inside[j]))
        for i, j, e in m:
            errs.append(e)
            shape_ok += int(d["shape"][i] == gs[j])
            ph = d["highlight"][i] >= d["hl_thr"]
            htp += int(ph and gh[j])
            hfp += int(ph and not gh[j])
            hfn += int((not ph) and gh[j])
        gb = r["ball"]
        pb = d["ball_uv"]
        if gb is not None and pb is not None and np.hypot(*(np.array(gb) - pb)) <= ball_tol:
            btp += 1
            berr.append(float(np.hypot(*(np.array(gb) - pb))))
        else:
            bfn += int(gb is not None and 0 <= gb[0] < CW and 0 <= gb[1] < CH)
            bfp += int(pb is not None)
    P = tp / max(tp + fp, 1)
    R = tp / max(tp + fn, 1)
    px_m = 105.0 / 291.0
    return {"images": len(rows), "player_precision": round(P, 4), "player_recall": round(R, 4),
            "f1": round(2 * P * R / max(P + R, 1e-9), 4), "shape_acc": round(shape_ok / max(tp, 1), 4),
            "loc_err_px": round(float(np.mean(errs)) if errs else 0.0, 3),
            "loc_err_m": round((float(np.mean(errs)) if errs else 0.0) * px_m, 3),
            "ball_recall": round(btp / max(btp + bfn, 1), 4), "ball_precision": round(btp / max(btp + bfp, 1), 4),
            "ball_err_px": round(float(np.mean(berr)) if berr else 0.0, 3),
            "hl_precision": round(htp / max(htp + hfp, 1), 4), "hl_recall": round(htp / max(htp + hfn, 1), 4),
            "ms_mean": round(float(np.mean(t_ms)), 3)}


def torch_predictor(net):
    net.eval()
    dev = next(net.parameters()).device

    def f(img_bgr):
        x = torch.from_numpy(np.ascontiguousarray(img_bgr[:, :, ::-1]).transpose(2, 0, 1)[None].astype(np.float32) / 255.0)
        with torch.no_grad():
            return torch.sigmoid(net(x.to(dev)))[0].cpu().numpy()
    return f


def onnx_predictor(path: str, providers=None):
    from fctac.vision.learned import make_session
    sess = make_session(path, providers)
    name = sess.get_inputs()[0].name

    def f(img_bgr):
        x = np.ascontiguousarray(img_bgr[:, :, ::-1]).transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        return sess.run(None, {name: x})[0][0]
    return f


def export_onnx(net, path: str, meta: dict):
    net = net.cpu().eval()
    m = Exported(net)
    x = torch.zeros(1, 3, CH, CW)
    torch.onnx.export(m, x, path, input_names=["image"], output_names=["heat"], opset_version=17,
                      dynamo=False)
    with open(path + ".json", "w") as f:
        json.dump(meta, f, indent=1)
