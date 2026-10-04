"""ONNX Runtime point detector (runtime side; no PyTorch needed).

Same interface as ColorDetector: ``detect(frame, ball_prior) -> [Detection]``.
Provider order comes from the config (TensorRT -> CUDA -> DirectML -> CPU,
whatever is installed); FP16 models are used as-is.
"""
from __future__ import annotations

import json
import os
from typing import Optional

import cv2
import numpy as np

from fctac import types as T


def make_session(path: str, providers: Optional[list] = None, threads: int = 0):
    import onnxruntime as ort
    avail = ort.get_available_providers()
    want = providers or ["TensorrtExecutionProvider", "CUDAExecutionProvider", "DmlExecutionProvider",
                         "CPUExecutionProvider"]
    use = []
    for p in want:
        if p not in avail:
            continue
        if p == "TensorrtExecutionProvider":
            cache = os.path.join(os.path.dirname(os.path.abspath(path)), "trt_cache")
            use.append((p, {"trt_fp16_enable": True, "trt_engine_cache_enable": True,
                            "trt_engine_cache_path": cache}))
        else:
            use.append(p)
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if threads:
        so.intra_op_num_threads = threads
    return ort.InferenceSession(path, so, providers=use or ["CPUExecutionProvider"])


def decode(hm, off, hgt, stride, thr, max_det=40):
    """Peaks of (C,H,W) heatmaps -> list of (channel, x, y, score, height) in input pixels."""
    out = []
    for c in range(hm.shape[0]):
        h = hm[c]
        peaks = (h >= cv2.dilate(h, np.ones((3, 3), np.uint8))) & (h > thr)
        ys, xs = np.nonzero(peaks)
        if len(xs) > max_det:
            k = np.argsort(-h[ys, xs])[:max_det]
            ys, xs = ys[k], xs[k]
        for y, x in zip(ys, xs):
            px = (x + off[0, y, x]) * stride
            py = (y + off[1, y, x]) * stride
            out.append((c, float(px), float(py), float(h[y, x]), float(np.exp(hgt[0, y, x]))))
    return out


class OnnxDetector:
    name = "onnx"

    def __init__(self, model_path: str, cfg=None, providers: Optional[list] = None, thr: float = 0.3):
        meta_path = model_path + ".json"
        meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {"input": [384, 640], "stride": 4}
        self.in_h, self.in_w = meta["input"]
        self.stride = meta.get("stride", 4)
        self.sess = make_session(model_path, providers)
        self.inp = self.sess.get_inputs()[0].name
        self.fp16 = "float16" in self.sess.get_inputs()[0].type
        self.thr = thr
        self.cfg = cfg
        self.buf = np.zeros((self.in_h, self.in_w, 3), np.uint8)
        chans = meta.get("channels", [])
        self.marker_mode = len(chans) > 2 and chans[2] == "controlled_marker"
        self.marker_above = meta.get("marker_above", 1.15)

    @property
    def providers(self) -> list:
        return self.sess.get_providers()

    def detect(self, frame: np.ndarray, ball_prior: Optional[np.ndarray] = None) -> list:
        H, W = frame.shape[:2]
        s = self.in_w / W
        rh = min(int(round(H * s)), self.in_h)
        small = cv2.resize(frame, (self.in_w, rh), interpolation=cv2.INTER_AREA)
        self.buf[:rh] = small[:rh]
        self.buf[rh:] = 0
        x = self.buf.transpose(2, 0, 1)[None].astype(np.float16 if self.fp16 else np.float32) * (1 / 255.0)
        hm, off, hgt = self.sess.run(None, {self.inp: x})
        peaks = decode(hm[0].astype(np.float32), off[0].astype(np.float32), hgt[0].astype(np.float32),
                       self.stride, self.thr)
        lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
        dets: list[T.Detection] = []
        ctrl = []
        balls = []
        for c, px, py, sc, ph in peaks:
            if c == 0:
                d = T.Detection(cls=T.CLS_PLAYER, x=px / s, y=py / s, w=0.35 * ph / s, h=ph / s, conf=sc)
                # kit colour from the torso region of the working image
                y0, y1 = int(py - 0.8 * ph), int(py - 0.5 * ph)
                x0, x1 = int(px - 0.12 * ph), int(px + 0.12 * ph) + 1
                patch = lab[max(0, y0):max(1, y1), max(0, x0):max(1, x1)]
                d.feature = patch.reshape(-1, 3).mean(0).astype(np.float32) if patch.size else np.zeros(3, np.float32)
                dets.append(d)
            elif c == 1:
                balls.append((sc, px / s, py / s))
            else:
                ctrl.append((sc, px / s, py / s))
        if ctrl and dets:
            sc, cx, cy = max(ctrl)
            if self.marker_mode:
                # marker above the head -> the player whose head is just below it
                def cost(d):
                    return abs(d.x - cx) / max(d.w, 1) + abs((d.y - self.marker_above * d.h) - cy) / max(d.h, 1)
                d = min(dets, key=cost)
                ok = cost(d) < 1.0
            else:
                d = min(dets, key=lambda d: np.hypot(d.x - cx, d.y - cy))
                ok = np.hypot(d.x - cx, d.y - cy) < max(d.h * 0.5, 8)
            if ok:
                d.controlled, d.controlled_conf = True, sc
        if balls:
            if ball_prior is not None:
                balls.sort(key=lambda b: -(b[0] * np.exp(-np.hypot(b[1] - ball_prior[0], b[2] - ball_prior[1]) / (0.1 * W))))
            else:
                balls.sort(key=lambda b: -b[0])
            sc, bx, by = balls[0]
            dets.append(T.Detection(cls=T.CLS_BALL, x=bx, y=by, w=8 / s, h=8 / s, conf=sc))
        return dets
