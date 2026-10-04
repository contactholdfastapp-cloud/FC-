"""Transparent overlay prototype demo.

Windows: plays a clip in a normal window and puts the real click-through
layered overlay (fctac.overlay.win32) on top of it, driven by the analysis -
exactly how the live assistant sits above FC 27.

Other OS: falls back to compositing the overlay into the preview window.

    python tools/overlay_demo.py --video data/synthetic/clip01.mp4 --oracle
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fctac.capture.video import VideoSource  # noqa: E402
from fctac.overlay.renderer import OverlayConfig, OverlayRenderer  # noqa: E402
from fctac.replay.viewer import build_analyzer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--oracle", action="store_true")
    ap.add_argument("--gt", default="")
    ap.add_argument("--events", default="")
    ap.add_argument("--config", default="")
    ap.add_argument("--calib", default="")
    ap.add_argument("--seconds", type=float, default=0)
    ap.add_argument("--ranker", default="")
    ap.add_argument("--predictor", default="")
    a = ap.parse_args()
    src = VideoSource(a.video)
    an, _ = build_analyzer(a, src)
    ov = OverlayRenderer(src.width, src.height, OverlayConfig())
    title = "FCTAC demo video"
    cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
    layered = None
    if sys.platform == "win32":
        from fctac.overlay import win32
        win32.set_dpi_aware()
        cv2.imshow(title, src.get(0))
        cv2.waitKey(50)
        hwnd = win32.find_window((title,))
        if hwnd:
            x, y, w, h = win32.client_rect_on_screen(hwnd)
            layered = win32.LayeredOverlay(x, y, src.width, src.height)
            print(f"layered overlay at {x},{y} {src.width}x{src.height}")
    n = len(src) if a.seconds <= 0 else min(len(src), int(a.seconds * src.fps))
    t_next = time.perf_counter()
    prev_dirty = None
    for i in range(n):
        frame = src.get(i)
        if frame is None:
            break
        fa = an.process(frame, i, src.time_of(i))
        t0 = time.perf_counter()
        canvas = ov.render(fa)
        if layered is not None:
            region = canvas.dirty if prev_dirty is None or canvas.dirty is None else [
                min(prev_dirty[0], canvas.dirty[0]), min(prev_dirty[1], canvas.dirty[1]),
                max(prev_dirty[2], canvas.dirty[2]), max(prev_dirty[3], canvas.dirty[3])]
            layered.present(canvas.img, region)
            prev_dirty = canvas.dirty
            show = frame
        else:
            show = canvas.composite_onto(frame.copy())
        ov_ms = (time.perf_counter() - t0) * 1000
        cv2.setWindowTitle(title, f"{title}  analysis {fa.timings_ms.get('total', 0):.1f} ms  overlay {ov_ms:.1f} ms")
        cv2.imshow(title, show)
        t_next += 1.0 / src.fps
        if cv2.waitKey(max(1, int((t_next - time.perf_counter()) * 1000))) & 0xFF == 27:
            break
    if layered is not None:
        layered.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
