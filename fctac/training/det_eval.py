"""Evaluate any detector on labelled images (same metrics for baseline and learned)."""
from __future__ import annotations

import time

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from fctac import types as T
from fctac.training.det_dataset import load_label


def evaluate_on_labels(detector, paths: list[str], gate_frac: float = 0.02) -> dict:
    tp = fp = fn = 0
    btp = bfp = bfn = 0
    ctrl_ok, ctrl_n = 0, 0
    err, ms = [], []
    for p in paths:
        img = cv2.imread(p)
        lab = load_label(p)
        w = lab["width"]
        gate = gate_frac * w
        t0 = time.perf_counter()
        dets = detector.detect(img, None)
        ms.append((time.perf_counter() - t0) * 1000)
        gp = [o for o in lab["objects"] if o["cls"] in ("player", "referee")]
        dp = [d for d in dets if d.cls == T.CLS_PLAYER]
        if gp and dp:
            D = np.array([[np.hypot(d.x - o["x"], d.y - o["y"]) for o in gp] for d in dp])
            a, b = linear_sum_assignment(D)
            ok = D[a, b] < gate
            tp += int(ok.sum())
            fp += len(dp) - int(ok.sum())
            fn += len(gp) - int(ok.sum())
            err += list(D[a, b][ok])
            gc = [j for j, o in enumerate(gp) if o.get("controlled")]
            if gc:
                ctrl_n += 1
                hit = [dp[i] for i, j in zip(a[ok], b[ok]) if j == gc[0]]
                ctrl_ok += int(bool(hit) and hit[0].controlled and sum(d.controlled for d in dp) == 1)
        else:
            fp += len(dp)
            fn += len(gp)
        gb = [o for o in lab["objects"] if o["cls"] == "ball"]
        db = [d for d in dets if d.cls == T.CLS_BALL]
        if gb:
            if db and np.hypot(db[0].x - gb[0]["x"], db[0].y - gb[0]["y"]) < gate:
                btp += 1
            else:
                bfn += 1
                bfp += int(bool(db))
        else:
            bfp += int(bool(db))
    pp = tp / max(tp + fp, 1)
    pr = tp / max(tp + fn, 1)
    bp = btp / max(btp + bfp, 1)
    br = btp / max(btp + bfn, 1)
    f1 = 2 * pp * pr / max(pp + pr, 1e-9)
    ctrl = ctrl_ok / max(ctrl_n, 1)
    return {
        # composite used for deployment: players matter most, but a detector that
        # cannot find the ball or the controlled player is not an upgrade
        "det_score": round(0.5 * f1 + 0.25 * br + 0.25 * ctrl, 4),
        "player_precision": round(pp, 4), "player_recall": round(pr, 4),
        "f1": round(f1, 4),
        "ball_precision": round(bp, 4), "ball_recall": round(br, 4),
        "controlled_acc": round(ctrl_ok / max(ctrl_n, 1), 4),
        "foot_err_px": round(float(np.mean(err)), 2) if err else None,
        "ms_mean": round(float(np.mean(ms)), 2) if ms else None, "images": len(paths),
    }


def evaluate_onnx_on_labels(onnx_path: str, paths: list[str], providers=None) -> dict:
    from fctac.vision.learned import OnnxDetector
    return evaluate_on_labels(OnnxDetector(onnx_path, providers=providers), paths)
