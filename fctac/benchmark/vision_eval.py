"""Vision evaluation against ground truth (synthetic GT or annotated FC 27 frames).

Metrics: player precision/recall, foot-point error, team accuracy,
controlled-player accuracy, ball precision/recall, radar recall/error,
latency.  Repeatable: same clip + config -> same numbers.
"""
from __future__ import annotations

import json
import time

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from fctac import types as T
from fctac.pitch.camera import CameraParams


def gt_screen(snap: dict, width: int, height: int):
    cam = CameraParams(**snap["camera"])
    if cam.width != width:
        cam = cam.scaled(width, height)
    P = np.array([[p["x"], p["y"], 0.0] for p in snap["players"]])
    uv, z = cam.project(P)
    b = snap["ball"]
    buv, bz = cam.project(np.array([[b["x"], b["y"], b["z"]]]))
    vis = (z > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
    # people hidden behind HUD (radar) are not expected from the main view
    x0, y0, x1, y1 = 0.415 * width, 0.795 * height, 0.585 * width, 0.985 * height
    in_hud = (uv[:, 0] > x0) & (uv[:, 0] < x1) & (uv[:, 1] > y0) & (uv[:, 1] < y1)
    vis &= ~in_hud
    bvis = bool(bz[0] > 0 and 0 <= buv[0, 0] < width and 0 <= buv[0, 1] < height)
    return uv, vis, buv[0], bvis, cam


class VisionEval:
    def __init__(self, gate_frac: float = 0.025):
        self.gate_frac = gate_frac
        self.tp = self.fp = self.fn = 0
        self.err = []
        self.team_ok = []
        self.ctrl = []
        self.ball_tp = self.ball_fp = self.ball_fn = 0
        self.ball_err = []
        self.ms = []

    def add(self, dets: list, snap: dict, width: int, height: int, ms: float | None = None, team_map=None):
        uv, vis, buv, bvis, _ = gt_screen(snap, width, height)
        gate = self.gate_frac * width
        players = [d for d in dets if d.cls == T.CLS_PLAYER]
        gi = np.where(vis)[0]
        if len(players) and len(gi):
            D = np.array([[np.hypot(d.x - uv[g, 0], d.y - uv[g, 1]) for g in gi] for d in players])
            a, b = linear_sum_assignment(D)
            ok = D[a, b] < gate
            self.tp += int(ok.sum())
            self.fp += len(players) - int(ok.sum())
            self.fn += len(gi) - int(ok.sum())
            self.err += list(D[a, b][ok])
            gt_team = np.array([p["team"] for p in snap["players"]])
            for ai, bi in zip(a[ok], b[ok]):
                d = players[ai]
                if d.team in (T.TEAM_US, T.TEAM_THEM):
                    self.team_ok.append(int(d.team == gt_team[gi[bi]]))
            gc = next(i for i, p in enumerate(snap["players"]) if p["controlled"])
            if vis[gc]:
                hit = [players[ai] for ai, bi in zip(a[ok], b[ok]) if gi[bi] == gc]
                flagged = [d for d in players if d.controlled]
                self.ctrl.append(int(len(hit) == 1 and hit[0].controlled and len(flagged) == 1))
        else:
            self.fp += len(players)
            self.fn += len(gi)
        balls = [d for d in dets if d.cls == T.CLS_BALL]
        if bvis:
            if balls and np.hypot(balls[0].x - buv[0], balls[0].y - buv[1]) < gate:
                self.ball_tp += 1
                self.ball_err.append(float(np.hypot(balls[0].x - buv[0], balls[0].y - buv[1])))
            else:
                self.ball_fn += 1
                if balls:
                    self.ball_fp += 1
        elif balls:
            self.ball_fp += 1
        if ms is not None:
            self.ms.append(ms)

    def summary(self) -> dict:
        p = self.tp / max(self.tp + self.fp, 1)
        r = self.tp / max(self.tp + self.fn, 1)
        return {
            "player_precision": round(p, 4), "player_recall": round(r, 4),
            "foot_err_px_mean": round(float(np.mean(self.err)), 2) if self.err else None,
            "team_acc": round(float(np.mean(self.team_ok)), 4) if self.team_ok else None,
            "controlled_acc": round(float(np.mean(self.ctrl)), 4) if self.ctrl else None,
            "ball_precision": round(self.ball_tp / max(self.ball_tp + self.ball_fp, 1), 4),
            "ball_recall": round(self.ball_tp / max(self.ball_tp + self.ball_fn, 1), 4),
            "ball_err_px": round(float(np.median(self.ball_err)), 2) if self.ball_err else None,
            "ms_mean": round(float(np.mean(self.ms)), 2) if self.ms else None,
            "ms_p95": round(float(np.percentile(self.ms, 95)), 2) if self.ms else None,
        }


def evaluate_detector(detector, video: str, gt: list, max_frames: int = 0, use_prior: bool = True) -> dict:
    cap = cv2.VideoCapture(video)
    ev = VisionEval()
    prior = None
    n = len(gt) if max_frames <= 0 else min(len(gt), max_frames)
    for i in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        t0 = time.perf_counter()
        dets = detector.detect(frame, prior if use_prior else None)
        ms = (time.perf_counter() - t0) * 1000
        ev.add(dets, gt[i], w, h, ms)
        b = [d for d in dets if d.cls == T.CLS_BALL]
        prior = np.array([b[0].x, b[0].y]) if b else None
    return ev.summary()


if __name__ == "__main__":
    import argparse
    from fctac.state.gt import load_gt
    from fctac.vision.detector import ColorDetector
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    ap.add_argument("--frames", type=int, default=0)
    a = ap.parse_args()
    print(json.dumps(evaluate_detector(ColorDetector(), a.clip + ".mp4", load_gt(a.clip + ".gt.jsonl"), a.frames), indent=1))
