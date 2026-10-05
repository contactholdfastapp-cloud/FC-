"""End-to-end check of the full pipeline on a real FC 27 video (no ground truth).

    python tools/real_check.py --video data/real/m3.mp4 --seconds 120 --out runs/real_check/m3

Runs the live analysis path (radar -> detector -> calibration -> tracking ->
state -> tactics) on every --step'th frame and reports what can be measured
without labels: how often the radar is read, how often the screen<->pitch
calibration is confident and its registration error (metres, radar vs main
view), possession / phase coverage, recommendation status mix, timings.
Saves an overlay image every --snap-every seconds for visual QA.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fctac import types as T  # noqa: E402
from fctac.config import load_config  # noqa: E402
from fctac.overlay.renderer import OverlayRenderer  # noqa: E402
from fctac.vision.analyzer import VisionAnalyzer  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--config", default="configs/fc27_1440p.json")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--seconds", type=float, default=120.0)
    ap.add_argument("--step", type=int, default=2)
    ap.add_argument("--snap-every", type=float, default=4.0)
    ap.add_argument("--out", default="runs/real_check")
    ap.add_argument("--set", nargs="*", default=[], help="config overrides section.key=value (JSON values)")
    a = ap.parse_args()
    ov: dict = {}
    for kv in a.set:
        k, v = kv.split("=", 1)
        sec, key = k.split(".", 1)
        try:
            v = json.loads(v)
        except json.JSONDecodeError:
            pass
        ov.setdefault(sec, {})[key] = v
    cfg = load_config(a.config, ov)
    an = VisionAnalyzer(cfg)
    os.makedirs(a.out, exist_ok=True)
    cap = cv2.VideoCapture(a.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    i0 = int(a.start * fps)
    for _ in range(i0):
        cap.grab()
    n_total = int(a.seconds * fps)
    stats = Counter()
    rmse, tms, cconf = [], [], []
    status = Counter()
    phases = Counter()
    renderer = None
    next_snap = 0.0
    k = 0
    for i in range(n_total):
        if i % a.step:
            if not cap.grab():
                break
            continue
        ok, frame = cap.read()
        if not ok:
            break
        t = (i0 + i) / fps
        fa = an.process(frame, k, t)
        k += 1
        stats["frames"] += 1
        rd = getattr(an.radar, "last", None)
        if rd is not None and len(rd["uv"]) >= 15:
            stats["radar_read"] += 1
        if an.calib.H is not None and an.calib.conf >= 0.5:
            stats["calibrated"] += 1
            cconf.append(an.calib.conf)
            if an.calib.fails == 0 and an.calib.last_matches:
                rmse.append(an.calib.last_rmse)
        st = fa.state
        if st is not None:
            phases[{T.TEAM_US: "us", T.TEAM_THEM: "them"}.get(st.possession, "unknown")] += 1
            if st.ball is not None:
                stats["ball"] += 1
            if st.controlled is not None:
                stats["controlled"] += 1
        status[fa.recommendation.status] += 1
        tms.append(sum(fa.timings_ms.values()))
        if t - a.start >= next_snap:
            next_snap += a.snap_every
            if renderer is None:
                renderer = OverlayRenderer(frame.shape[1], frame.shape[0], cfg.overlay)
            lines = [f"t={t:.1f}s calib={an.calib.conf:.2f} rmse={an.calib.last_rmse:.2f}m "
                     f"matches={len(an.calib.last_matches)} radar={0 if rd is None else len(rd['uv'])}",
                     f"rec {fa.recommendation.status} {fa.recommendation.action.text() if fa.recommendation.action else ''}"
                     f" {fa.recommendation.reason}"]
            canvas = renderer.render(fa, lines)
            img = canvas.composite_onto(frame.copy())
            if an.calib.H is not None:
                # projected radar players: where the calibration puts them on screen
                try:
                    from fctac.pitch.camera import apply_h
                    from fctac.vision.radar_fc27 import canon_to_pitch
                    if rd is not None and len(rd["uv"]):
                        q = apply_h(np.linalg.inv(an.calib.H), canon_to_pitch(rd["uv"]))
                        for (x, y), s in zip(q, rd["shape"]):
                            if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
                                cv2.circle(img, (int(x), int(y)), 9, (255, 255, 0) if s == 0 else (255, 0, 255), 2)
                except (np.linalg.LinAlgError, ImportError):
                    pass
            for d in fa.detections:
                if d.cls == T.CLS_PLAYER:
                    cv2.line(img, (int(d.x) - 5, int(d.y)), (int(d.x) + 5, int(d.y)), (0, 255, 0), 2)
            cv2.imwrite(os.path.join(a.out, f"snap_{t:07.1f}.jpg"),
                        cv2.resize(img, (1280, 720), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 85])
    n = max(stats["frames"], 1)
    rep = {"video": a.video, "start": a.start, "seconds": a.seconds, "frames": stats["frames"],
           "radar_read_rate": round(stats["radar_read"] / n, 3),
           "calibrated_rate": round(stats["calibrated"] / n, 3),
           "calib_conf_mean": round(float(np.mean(cconf)), 3) if cconf else None,
           "registration_rmse_m_median": round(float(np.median(rmse)), 3) if rmse else None,
           "ball_rate": round(stats["ball"] / n, 3), "controlled_rate": round(stats["controlled"] / n, 3),
           "recommendation_status": {k2: round(v / n, 3) for k2, v in status.items()},
           "possession": {str(k2): round(v / n, 3) for k2, v in phases.items()},
           "analysis_ms_mean": round(float(np.mean(tms)), 2), "analysis_ms_p95": round(float(np.percentile(tms, 95)), 2)}
    print(json.dumps(rep, indent=1))
    with open(os.path.join(a.out, "report.json"), "w") as f:
        json.dump(rep, f, indent=1)


if __name__ == "__main__":
    main()
