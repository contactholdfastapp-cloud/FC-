"""Generate a synthetic FC-style clip with ground truth.

    python tools/make_synthetic.py --seconds 60 --out data/synthetic/clip01

Writes <out>.mp4, <out>.gt.jsonl (per-frame ground truth incl. camera) and
<out>.meta.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fctac.sim.match import MatchSim, SimConfig  # noqa: E402
from fctac.sim.render import CameraController, Renderer, RenderStyle, RADAR_RECT  # noqa: E402


def generate(out: str, seconds: float, seed: int, width: int, height: int, fps: int,
             kit_us: str, kit_them: str, attack_sign: int, warmup: float = 2.0) -> dict:
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    sim = MatchSim(SimConfig(fps=fps, seed=seed, us_attack_sign=attack_sign))
    style = RenderStyle(kit_us=kit_us, kit_them=kit_them)
    ren = Renderer(width, height, style, seed=seed)
    camc = CameraController(width, height, seed=seed)
    for _ in range(int(warmup * fps)):
        sim.step()
    vw = cv2.VideoWriter(out + ".mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not vw.isOpened():
        raise RuntimeError("cannot open video writer")
    n = int(seconds * fps)
    t0 = time.perf_counter()
    with open(out + ".gt.jsonl", "w") as f:
        for k in range(n):
            sim.step()
            snap = sim.snapshot()
            b = snap["ball"]
            cam = camc.update((b["x"], b["y"]), (b["vx"], b["vy"]), snap["t"], 1.0 / fps)
            img = ren.render(snap, cam)
            vw.write(img)
            snap["video_frame"] = k
            snap["camera"] = cam.to_dict()
            f.write(json.dumps(snap) + "\n")
    vw.release()
    meta = {"fps": fps, "width": width, "height": height, "frames": n, "seed": seed,
            "kit_us": kit_us, "kit_them": kit_them, "us_attack_sign": attack_sign,
            "radar_rect": RADAR_RECT, "synthetic": True,
            "render_s": round(time.perf_counter() - t0, 2)}
    with open(out + ".meta.json", "w") as f:
        json.dump(meta, f, indent=1)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic/clip01")
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--kit-us", default="red")
    ap.add_argument("--kit-them", default="blue")
    ap.add_argument("--attack-sign", type=int, default=1, choices=(1, -1))
    a = ap.parse_args()
    meta = generate(a.out, a.seconds, a.seed, a.width, a.height, a.fps, a.kit_us, a.kit_them, a.attack_sign)
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
