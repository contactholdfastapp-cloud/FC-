"""Extract diverse frames from gameplay videos and pre-annotate them.

    python tools/extract_frames.py data/recordings/session01.mp4 --out data/datasets/fc27 --every 15
    python tools/extract_frames.py data/synthetic/clip01.mp4 --out data/datasets/synth --gt   (labels from ground truth)

Pre-annotation runs the baseline pipeline (radar + colour detector +
radar-registered team labels) so annotating is mostly *correcting*
(tools/annotate.py).  Near-duplicate frames are skipped.
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fctac import types as T  # noqa: E402
from fctac.training.det_dataset import save_label  # noqa: E402


def gt_objects(snap: dict, w: int, h: int) -> list:
    from fctac.benchmark.vision_eval import gt_screen
    uv, vis, buv, bvis, cam = gt_screen(snap, w, h)
    P = np.array([[p["x"], p["y"], 1.85] for p in snap["players"]])
    head, _ = cam.project(P)
    objs = []
    for k, p in enumerate(snap["players"]):
        if vis[k]:
            objs.append({"cls": "player", "x": float(uv[k, 0]), "y": float(uv[k, 1]),
                         "h": float(uv[k, 1] - head[k, 1]), "team": int(p["team"]), "controlled": bool(p["controlled"])})
    r = snap.get("referee")
    if r:
        ruv, rz = cam.project(np.array([[r["x"], r["y"], 0.0], [r["x"], r["y"], 1.85]]))
        if rz[0] > 0 and 0 <= ruv[0, 0] < w and 0 <= ruv[0, 1] < h:
            objs.append({"cls": "referee", "x": float(ruv[0, 0]), "y": float(ruv[0, 1]),
                         "h": float(ruv[0, 1] - ruv[1, 1]), "team": 2, "controlled": False})
    if bvis:
        objs.append({"cls": "ball", "x": float(buv[0]), "y": float(buv[1]), "h": 0.0})
    return objs


def pipeline_objects(fa: T.FrameAnalysis) -> list:
    objs = []
    for d in fa.detections:
        if d.cls == T.CLS_PLAYER:
            objs.append({"cls": "player", "x": float(d.x), "y": float(d.y), "h": float(d.h),
                         "team": int(d.team) if d.team in (0, 1, 2) else -1, "controlled": bool(d.controlled)})
        elif d.cls == T.CLS_BALL:
            objs.append({"cls": "ball", "x": float(d.x), "y": float(d.y), "h": float(d.h)})
    return objs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--every", type=int, default=15, help="consider every Nth frame")
    ap.add_argument("--min-diff", type=float, default=6.0, help="skip frames this similar (mean abs diff, 0-255)")
    ap.add_argument("--max", type=int, default=0)
    ap.add_argument("--gt", action="store_true", help="labels from synthetic ground truth")
    ap.add_argument("--config", default="")
    a = ap.parse_args()
    total = 0
    for vid in a.videos:
        name = os.path.splitext(os.path.basename(vid))[0]
        out_dir = os.path.join(a.out, name)
        os.makedirs(out_dir, exist_ok=True)
        cap = cv2.VideoCapture(vid)
        gt = None
        an = None
        if a.gt:
            from fctac.state.gt import load_gt
            gt = load_gt(os.path.splitext(vid)[0] + ".gt.jsonl")
        else:
            from fctac.vision.analyzer import VisionAnalyzer
            an = VisionAnalyzer.from_files(a.config)
        prev_small = None
        i = -1
        n = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            i += 1
            fa = an.process(frame, i, i / 30.0) if an is not None else None   # keep trackers warm
            if i % a.every:
                continue
            small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA).astype(np.int16)
            if prev_small is not None and np.abs(small - prev_small).mean() < a.min_diff:
                continue
            prev_small = small
            h, w = frame.shape[:2]
            img_path = os.path.join(out_dir, f"{name}_{i:06d}.jpg")
            cv2.imwrite(img_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            objs = gt_objects(gt[i], w, h) if gt is not None else pipeline_objects(fa)
            save_label(img_path, {"image": os.path.basename(img_path), "width": w, "height": h,
                                  "source": os.path.basename(vid), "frame": i,
                                  "status": "verified" if gt is not None else "auto", "objects": objs})
            n += 1
            if a.max and n >= a.max:
                break
        total += n
        print(f"{vid}: {n} frames -> {out_dir}")
    print(f"total {total}")


if __name__ == "__main__":
    main()
