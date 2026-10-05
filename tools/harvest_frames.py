"""Harvest training material from FC 27 gameplay videos.

    python tools/harvest_frames.py data/recordings/*.mp4 --out data/real/harvest
    python tools/harvest_frames.py data/real/ts/*.ts --out data/real/harvest --groups "0-299:m1,300-469:m2,470-599:m3"

Decodes every frame sequentially (seeking in long-GOP streams is unreliable)
and keeps:
  frames/<clip>_<frame>.jpg      full frames every --frame-every frames
  radar/<clip>_<frame>.jpg       radar crops (panel + margin, native pixels)
                                 every --radar-every frames
  index.csv                      one row per kept sample: clip, frame, time,
                                 group (match id for leakage-safe splits),
                                 centre green fraction, radar line score

The radar panel position comes from ``RadarConfig.panel`` (normalised
screen rect of the radar pitch).  Nothing is filtered here; the gameplay
filter (radar visible, broadcast camera) is applied later from index.csv.
Several videos are processed in parallel (``--workers``).
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fctac.vision.radar import RadarConfig, panel_px, radar_crop_rect  # noqa: E402


def green_fraction(frame: np.ndarray) -> float:
    small = cv2.resize(frame, (480, 270), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    g = cv2.inRange(hsv, (33, 45, 35), (92, 255, 255))
    return float(g[40:200, 60:420].mean() / 255.0)


def radar_line_score(frame: np.ndarray, panel: tuple) -> float:
    """Contrast of the radar's halfway line against the panel next to it.

    With the radar on screen the 1-px halfway line is ~30-40 grey levels
    brighter than the dark panel 4-6 px either side; otherwise ~0.
    """
    x0, y0, x1, y1 = panel_px(panel, frame.shape[1], frame.shape[0])
    g = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = g.shape
    cx = w // 2
    s = max(1, int(round(w / 290.0)))
    rows = g[int(0.05 * h):int(0.95 * h)]
    line = rows[:, cx - 2 * s:cx + 2 * s + 1].max(1)
    side = np.maximum(rows[:, cx - 7 * s:cx - 4 * s].mean(1), rows[:, cx + 4 * s + 1:cx + 7 * s + 1].mean(1))
    return float(np.median(line - side))


def group_of(name: str, groups: list) -> str:
    m = re.search(r"(\d+)", os.path.basename(name))
    if not groups or not m:
        return os.path.splitext(os.path.basename(name))[0]
    k = int(m.group(1))
    for lo, hi, g in groups:
        if lo <= k <= hi:
            return g
    return "other"


def process(job):
    path, out, frame_every, radar_every, panel, margin, quality, groups, name = job
    clip = name or os.path.splitext(os.path.basename(path))[0]
    grp = name or group_of(path, groups)
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    rows = []
    i = 0
    while True:
        need_frame = i % frame_every == 0
        need_radar = i % radar_every == 0
        if not (need_frame or need_radar):
            if not cap.grab():
                break
            i += 1
            continue
        ok, f = cap.read()
        if not ok:
            break
        h, w = f.shape[:2]
        if need_frame:
            cv2.imwrite(os.path.join(out, "frames", f"{clip}_{i:05d}.jpg"), f, [cv2.IMWRITE_JPEG_QUALITY, quality])
        cx0, cy0, cx1, cy1 = radar_crop_rect(panel, w, h, margin)
        cv2.imwrite(os.path.join(out, "radar", f"{clip}_{i:05d}.jpg"), f[cy0:cy1, cx0:cx1],
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        rows.append({"clip": clip, "frame": i, "t": round(i / fps, 3), "group": grp, "full": int(need_frame),
                     "green": round(green_fraction(f), 4), "radar_line": round(radar_line_score(f, panel), 2),
                     "w": w, "h": h})
        i += 1
    return rows


def parse_groups(s: str) -> list:
    out = []
    for part in filter(None, s.split(",")):
        rng, g = part.split(":")
        lo, hi = rng.split("-")
        out.append((int(lo), int(hi), g))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--frame-every", type=int, default=30, help="keep a full frame every N frames (60 fps: 30 = 2/s)")
    ap.add_argument("--radar-every", type=int, default=12, help="keep a radar crop every N frames (60 fps: 12 = 5/s)")
    ap.add_argument("--margin", type=float, default=0.10, help="radar crop margin, fraction of the panel width")
    ap.add_argument("--quality", type=int, default=92)
    ap.add_argument("--groups", default="", help='"lo-hi:name,..." by the first number in the file name')
    ap.add_argument("--names", nargs="*", default=[], help="clip/group name per video (same order as videos)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2)))
    a = ap.parse_args()
    for d in ("frames", "radar"):
        os.makedirs(os.path.join(a.out, d), exist_ok=True)
    panel = RadarConfig().panel
    groups = parse_groups(a.groups)
    if a.names and len(a.names) != len(a.videos):
        raise SystemExit("--names needs one name per video")
    pairs = sorted(zip(a.videos, a.names or [""] * len(a.videos)))
    jobs = [(p, a.out, a.frame_every, a.radar_every, panel, a.margin, a.quality, groups, n) for p, n in pairs]
    rows = []
    with ProcessPoolExecutor(a.workers) as ex:
        for k, r in enumerate(ex.map(process, jobs)):
            rows += r
            if (k + 1) % 10 == 0 or k + 1 == len(jobs):
                print(f"{k + 1}/{len(jobs)} videos, {len(rows)} samples", flush=True)
    with open(os.path.join(a.out, "index.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    print("wrote", os.path.join(a.out, "index.csv"), len(rows), "rows")


if __name__ == "__main__":
    main()
