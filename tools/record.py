"""Record FC 27 gameplay for the dataset (Windows, standard screen capture).

    python tools/record.py --out data/recordings/session01 --minutes 10

Writes <out>.mp4 plus <out>.frames.jsonl (capture timestamp per written frame)
and <out>.meta.json.  Any other recorder works too (OBS, ShadowPlay): the
replay/dataset tools only need a video file -- record at your gameplay
resolution (1440p) and 60 fps with high quality so the radar and the ball
stay sharp.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fctac.config import load_config  # noqa: E402
from fctac.capture.windows import open_source  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--minutes", type=float, default=5)
    ap.add_argument("--fps", type=int, default=30, help="written fps (frames are sampled from capture)")
    ap.add_argument("--config", default="")
    ap.add_argument("--video", default="", help="test the recorder on a video file instead of the screen")
    a = ap.parse_args()
    cfg = load_config(a.config or None)
    if sys.platform == "win32":
        from fctac.overlay.win32 import set_dpi_aware
        set_dpi_aware()
    src = open_source(cfg.capture, a.video or None).start()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    vw = None
    last_id = -1
    n = 0
    t_end = time.perf_counter() + a.minutes * 60
    next_t = time.perf_counter()
    with open(a.out + ".frames.jsonl", "w") as fl:
        while time.perf_counter() < t_end:
            f = src.buffer.get(last_id, timeout=1.0)
            if f is None:
                if src.error:
                    raise RuntimeError(src.error)
                if getattr(src, "finished", None) is not None and src.finished.is_set():
                    break
                continue
            last_id = f.frame_id
            if f.t_capture < next_t:
                continue
            next_t += 1.0 / a.fps
            if vw is None:
                h, w = f.image.shape[:2]
                vw = cv2.VideoWriter(a.out + ".mp4", cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (w, h))
            vw.write(f.image)
            fl.write(json.dumps({"i": n, "t": f.t_capture}) + "\n")
            n += 1
            if n % (a.fps * 10) == 0:
                print(f"{n} frames, capture {src.buffer.rate_in.rate:.0f} fps, dropped {src.buffer.dropped}")
    src.stop()
    if vw is not None:
        vw.release()
    meta = {"frames": n, "fps": a.fps, "backend": src.name, "dropped_capture_frames": src.buffer.dropped}
    with open(a.out + ".meta.json", "w") as f:
        json.dump(meta, f, indent=1)
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
