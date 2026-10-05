"""Auto-label main-view players on real FC 27 footage (radar + calibration).

    python tools/pseudo_label_view.py --video data/real/m1.mp4 --out data/datasets/real/m1 --source m1
    python tools/pseudo_label_view.py --video my_game.mp4 --out data/datasets/real/mine --source mine --webcams none

The pipeline runs on the video (FC 27 radar reader + colour detector +
radar<->view registration).  On frames where the registration is confident,
every colour detection that lands on a radar player becomes a labelled
player (foot point, height, controlled flag from the radar highlight).
Everything uncertain becomes an *ignore* region instead of a label:
radar players the detector missed (occluded / merged), detections on the
pitch without a radar partner (referee or error), a ball that is not
confirmed by both the radar and the image.  People off the pitch (stewards,
photographers, ad boards) get no label, so a model trained on these learns
to ignore them.  ``--webcams`` masks broadcast picture-in-picture boxes.

The labels are "auto" quality; check a sample with tools/annotate.py.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fctac import types as T  # noqa: E402
from fctac.config import load_config  # noqa: E402
from fctac.pitch.camera import apply_h  # noqa: E402
from fctac.training.det_dataset import save_label  # noqa: E402
from fctac.vision.analyzer import VisionAnalyzer  # noqa: E402
from tools.harvest_frames import radar_line_score  # noqa: E402

# EA livestream picture-in-picture webcams + name panels (normalised rects)
WEBCAMS = [(0.045, 0.68, 0.225, 0.97), (0.775, 0.68, 0.955, 0.97)]


def label_frame(an: VisionAnalyzer, frame: np.ndarray, gate_m: float = 1.5) -> tuple:
    """-> (objects, info) or (None, reason)"""
    h, w = frame.shape[:2]
    cal = an.calib
    rd = an.radar
    last = getattr(rd, "last", None)
    if cal.H is None or cal.conf < 0.6 or cal.last_rmse > 1.0 or len(cal.last_matches) < 10:
        return None, "calibration"
    if last is None or len(last["uv"]) < 18:
        return None, "radar"
    from fctac.vision.radar_fc27 import canon_to_pitch
    R = canon_to_pitch(last["uv"])
    if an.cfg.radar.flip:
        R = np.array([105.0, 68.0]) - R
    hl = last["highlight"] >= last["hl_thr"]
    H = cal.H
    try:
        Hinv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return None, "calibration"
    dets = [d for d in an.last_dets if d.cls == T.CLS_PLAYER]
    balls = [d for d in an.last_dets if d.cls == T.CLS_BALL]
    objs = []
    used_r = set()
    if dets:
        P = apply_h(H, np.array([[d.x, d.y] for d in dets]))
        D = np.hypot(P[:, None, 0] - R[None, :, 0], P[:, None, 1] - R[None, :, 1])
        from scipy.optimize import linear_sum_assignment
        a, b = linear_sum_assignment(D)
        pair = {int(i): int(j) for i, j in zip(a, b) if D[i, j] < gate_m}
        for i, d in enumerate(dets):
            if i in pair:
                j = pair[i]
                used_r.add(j)
                objs.append({"cls": "player", "x": round(float(d.x), 1), "y": round(float(d.y), 1),
                             "h": round(float(d.h), 1), "team": -1, "radar_shape": int(last["shape"][j]),
                             "controlled": bool(hl[j])})
            elif -1.0 < P[i, 0] < 106.0 and -1.0 < P[i, 1] < 69.0:
                objs.append({"cls": "ignore", "x": round(float(d.x), 1), "y": round(float(d.y), 1),
                             "r": round(float(max(12.0, 0.6 * d.h)), 1)})
    # radar players the detector missed: unsure -> ignore
    hs = [o["h"] for o in objs if o["cls"] == "player"]
    href = float(np.median(hs)) if hs else 0.05 * h
    q = apply_h(Hinv, R)
    for j in range(len(R)):
        if j in used_r:
            continue
        x, y = q[j]
        if 0 <= x < w and 0 <= y < h:
            objs.append({"cls": "ignore", "x": round(float(x), 1), "y": round(float(y), 1),
                         "r": round(float(max(14.0, 0.8 * href)), 1)})
    # ball: needs radar and image to agree
    if last["ball_uv"] is not None:
        B = canon_to_pitch(last["ball_uv"][None])[0]
        if an.cfg.radar.flip:
            B = np.array([105.0, 68.0]) - B
        ok = False
        for d in balls:
            if np.hypot(*(apply_h(H, np.array([[d.x, d.y]]))[0] - B)) < 2.0:
                objs.append({"cls": "ball", "x": round(float(d.x), 1), "y": round(float(d.y), 1),
                             "h": round(float(d.h), 1)})
                ok = True
                break
        if not ok:
            x, y = apply_h(Hinv, B[None])[0]
            if 0 <= x < w and 0 <= y < h:
                objs.append({"cls": "ignore", "x": round(float(x), 1), "y": round(float(y) - 0.3 * href, 1),
                             "r": round(float(max(20.0, 0.9 * href)), 1)})
    n_pl = sum(1 for o in objs if o["cls"] == "player")
    if n_pl < 6:
        return None, "few_players"
    return objs, {"calib_conf": round(cal.conf, 3), "rmse": round(cal.last_rmse, 3), "players": n_pl}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--source", default="")
    ap.add_argument("--config", default="configs/fc27_1440p.json")
    ap.add_argument("--every", type=int, default=30, help="label every N source frames")
    ap.add_argument("--step", type=int, default=2, help="analyse every N source frames (keeps tracking alive)")
    ap.add_argument("--max-labels", type=int, default=0)
    ap.add_argument("--webcams", default="livestream", choices=["livestream", "none"])
    ap.add_argument("--debug-every", type=int, default=0, help="also save an overlay image every N labels")
    ap.add_argument("--all-frames", action="store_true",
                    help="analyse frames without a visible radar panel too (slower; they are never labelled)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    cfg = load_config(a.config, {"runtime": {"detect_every": 1}, "radar": {"mode": "fc27"}})
    an = VisionAnalyzer(cfg)
    cap = cv2.VideoCapture(a.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    src = a.source or os.path.splitext(os.path.basename(a.video))[0]
    ignore_rects = [list(r) for r in WEBCAMS] if a.webcams == "livestream" else []
    for r in cfg.detector.hud_masks:
        ignore_rects.append(list(r))
    i = n = 0
    reasons: dict = {}
    while True:
        if i % a.step:
            if not cap.grab():
                break
            i += 1
            continue
        ok, frame = cap.read()
        if not ok:
            break
        if not a.all_frames and radar_line_score(frame, cfg.radar.panel) < 12:
            reasons["no_radar_panel"] = reasons.get("no_radar_panel", 0) + int(i % a.every == 0)
            i += 1
            continue
        an.process(frame, i // a.step, i / fps)
        if i % a.every == 0:
            objs, info = label_frame(an, frame)
            if objs is None:
                reasons[info] = reasons.get(info, 0) + 1
            else:
                name = f"{src}_{i:06d}.jpg"
                p = os.path.join(a.out, name)
                cv2.imwrite(p, frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
                save_label(p, {"image": name, "width": frame.shape[1], "height": frame.shape[0], "source": src,
                               "frame": i, "status": "auto", "objects": objs, "ignore_rects": ignore_rects,
                               "info": info})
                n += 1
                if a.debug_every and n % a.debug_every == 1:
                    dbg = frame.copy()
                    for o in objs:
                        c = (int(o["x"]), int(o["y"]))
                        if o["cls"] == "player":
                            col = (0, 255, 255) if o["controlled"] else ((255, 255, 0) if o["radar_shape"] == 0 else (255, 0, 255))
                            cv2.circle(dbg, c, 6, col, 2)
                            cv2.line(dbg, c, (c[0], int(o["y"] - o["h"])), col, 1)
                        elif o["cls"] == "ball":
                            cv2.drawMarker(dbg, c, (0, 165, 255), cv2.MARKER_TILTED_CROSS, 18, 2)
                        else:
                            cv2.circle(dbg, c, int(o["r"]), (128, 128, 128), 1)
                    cv2.imwrite(os.path.join(a.out, f"debug_{name}"), dbg, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if a.max_labels and n >= a.max_labels:
                    break
        i += 1
        if i % 3000 == 0:
            print(f"{src}: frame {i}, labelled {n}, skipped {reasons}", flush=True)
    print(json.dumps({"source": src, "frames": i, "labelled": n, "skipped": reasons}))


if __name__ == "__main__":
    main()
