"""Render the assistant's overlay onto a recording -> shareable mp4.

    python tools/render_demo.py --video my_game.mp4 --start 30 --seconds 20 --out demo.mp4
    python tools/render_demo.py --video data/real/m3.mp4 --start 34 --seconds 26 --out demo_fc27.mp4 --debug

Runs exactly the live analysis path (radar -> detector -> calibration ->
tracking -> state -> tactics -> overlay) on every --step'th frame and writes
the composited frames (H.264 via ffmpeg when available, else OpenCV mp4v).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fctac.config import load_config  # noqa: E402
from fctac.overlay.renderer import OverlayRenderer  # noqa: E402
from fctac.vision.analyzer import VisionAnalyzer  # noqa: E402


class Writer:
    def __init__(self, path, w, h, fps):
        self.ff = None
        if shutil.which("ffmpeg"):
            self.ff = subprocess.Popen(
                ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(fps),
                 "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
                 "-movflags", "+faststart", path], stdin=subprocess.PIPE)
        else:
            self.cv = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    def write(self, img):
        if self.ff:
            self.ff.stdin.write(np.ascontiguousarray(img).tobytes())
        else:
            self.cv.write(img)

    def close(self):
        if self.ff:
            self.ff.stdin.close()
            self.ff.wait()
        else:
            self.cv.release()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--config", default="configs/fc27_1440p.json")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--step", type=int, default=2, help="process every Nth frame (60 fps source, 2 -> 30 fps video)")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--debug", action="store_true", help="show the debug panel (latency, calibration, radar)")
    ap.add_argument("--caption", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = load_config(a.config)
    cfg.overlay.show_debug = a.debug
    an = VisionAnalyzer(cfg)
    cap = cv2.VideoCapture(a.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    i0 = int(a.start * fps)
    for _ in range(i0):
        cap.grab()
    renderer = None
    writer = None
    k = 0
    counts = {}
    for i in range(int(a.seconds * fps)):
        if i % a.step:
            if not cap.grab():
                break
            continue
        ok, frame = cap.read()
        if not ok:
            break
        fa = an.process(frame, k, (i0 + i) / fps)
        k += 1
        if renderer is None:
            renderer = OverlayRenderer(frame.shape[1], frame.shape[0], cfg.overlay)
        lines = None
        if a.debug:
            st = fa.state
            lines = [f"analysis {sum(fa.timings_ms.values()):.0f} ms   calib {an.calib.conf:.2f}   "
                     f"radar {0 if getattr(an.radar, 'last', None) is None else len(an.radar.last['uv'])} symbols",
                     f"ball {'-' if st is None or st.ball is None else 'ok'}   "
                     f"controlled {'-' if st is None or st.controlled is None else st.controlled.label}",
                     f"rec {fa.recommendation.status} {fa.recommendation.action.text() if fa.recommendation.action else ''}"]
        img = renderer.render(fa, lines).composite_onto(frame.copy())
        s = a.width / img.shape[1]
        img = cv2.resize(img, (a.width, int(round(img.shape[0] * s / 2)) * 2), interpolation=cv2.INTER_AREA)
        if a.caption:
            cv2.putText(img, a.caption, (12, img.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, a.caption, (12, img.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                        cv2.LINE_AA)
        if writer is None:
            writer = Writer(a.out, img.shape[1], img.shape[0], round(fps / a.step))
        writer.write(img)
        st_ = fa.recommendation.status
        counts[st_] = counts.get(st_, 0) + 1
    if writer:
        writer.close()
    print(json.dumps({"out": a.out, "frames": k, "status": counts}))


if __name__ == "__main__":
    main()
