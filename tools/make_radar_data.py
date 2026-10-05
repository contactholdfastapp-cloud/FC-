"""Generate FC 27-style radar training crops with exact labels.

    python tools/make_radar_data.py --out data/radar/synth --n 40000 --workers 4

Each sample is a canonical 320x192 radar crop (``<out>/img/NNNNNN.jpg``);
labels go to ``<out>/labels.jsonl`` (canonical px: players with shape
0 = triangle / 1 = circle and highlight flag, ball or null, render mode).
Backgrounds are real main-view patches from ``data/real/harvest`` when
available (``--bg-groups`` restricts them, e.g. to the training matches).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fctac.sim.radar_fc27 import PositionPool, default_backgrounds, render  # noqa: E402


def work(job):
    lo, hi, out, seed, pos_path, bg_root, groups = job
    cv2.setNumThreads(1)
    pool = PositionPool.load(pos_path)
    bg = default_backgrounds(bg_root, groups, seed=seed, max_frames=150)
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(lo, hi):
        img, lab = render(rng, pool, bg)
        cv2.imwrite(os.path.join(out, "img", f"{k:06d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        rows.append({"id": k, "mode": lab["mode"],
                     "players": [[round(float(u), 2), round(float(v), 2), int(s), int(h)]
                                 for (u, v), s, h in zip(lab["uv"], lab["shape"], lab["highlight"])],
                     "ball": None if lab["ball"] is None else [round(float(lab["ball"][0]), 2),
                                                               round(float(lab["ball"][1]), 2)]})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/radar/synth")
    ap.add_argument("--n", type=int, default=40000)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--positions", default="data/radar/positions.npz")
    ap.add_argument("--bg-root", default="data/real/harvest")
    ap.add_argument("--bg-groups", default="", help="comma list of harvest groups for backgrounds (default all)")
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "img"), exist_ok=True)
    if not os.path.exists(a.positions):
        os.makedirs(os.path.dirname(a.positions) or ".", exist_ok=True)
        PositionPool(n_runs=24, steps=3000, every=12, seed=a.seed).save(a.positions)
    groups = tuple(g for g in a.bg_groups.split(",") if g) or None
    chunk = 500
    jobs = [(lo, min(lo + chunk, a.n), a.out, a.seed * 100003 + lo, a.positions, a.bg_root, groups)
            for lo in range(0, a.n, chunk)]
    rows = []
    with ProcessPoolExecutor(a.workers) as ex:
        for r in ex.map(work, jobs):
            rows += r
            print(f"{len(rows)}/{a.n}", flush=True)
    with open(os.path.join(a.out, "labels.jsonl"), "w") as f:
        for r in sorted(rows, key=lambda r: r["id"]):
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
