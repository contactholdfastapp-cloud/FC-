"""Live assistant: screen capture -> analysis -> transparent overlay.

    python -m fctac.live                              # Windows: capture the FC 27 window
    python -m fctac.live --preview                    # also show a preview window
    python -m fctac.live --video clip.mp4 --preview   # any OS: replay a file as if live
    python -m fctac.live --video clip.mp4 --headless --seconds 20 --report out.json   (benchmark)

Runtime rules: the capture thread overwrites a single latest-frame slot; the
analysis loop always takes the newest frame, so stale frames are dropped and
never queued.  Latency is measured per frame from frame availability to
overlay presentation.

Hotkeys (Windows, read-only key state): F7 swap which radar team is yours
(FC 27 radar), F8 toggle overlay, F9 debug panel, F10 quit.  The assistant
never sends input to the game.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Optional

import cv2
import numpy as np

from fctac import types as T
from fctac.config import AppConfig, load_config
from fctac.overlay.renderer import OverlayRenderer
from fctac.timing import LatencyStats, RateMeter


class NullPresenter:
    name = "none"

    def present(self, canvas, frame=None):
        pass

    def close(self):
        pass


class PreviewPresenter:
    """Composites the overlay onto the captured frame in an OpenCV window."""
    name = "preview"

    def __init__(self, title="FC27 tactical preview", max_w=1280):
        self.title = title
        self.max_w = max_w
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        self.key = -1

    def present(self, canvas, frame=None):
        if frame is None:
            return
        img = canvas.composite_onto(frame.copy())
        if img.shape[1] > self.max_w:
            s = self.max_w / img.shape[1]
            img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        cv2.imshow(self.title, img)
        self.key = cv2.waitKey(1)

    def close(self):
        cv2.destroyWindow(self.title)


class Win32Presenter:
    """The real transparent click-through overlay above the game."""
    name = "win32"

    def __init__(self, x, y, w, h):
        from fctac.overlay.win32 import LayeredOverlay
        self.ov = LayeredOverlay(x, y, w, h)
        self.prev_dirty = None
        self.n = 0

    def present(self, canvas, frame=None):
        d = canvas.dirty
        p = self.prev_dirty
        region = d if p is None else (p if d is None else [min(p[0], d[0]), min(p[1], d[1]), max(p[2], d[2]), max(p[3], d[3])])
        if region is not None:
            self.ov.present(canvas.img, region)
        else:
            self.ov.pump()
        self.prev_dirty = d
        self.n += 1
        if self.n % 120 == 0:            # games can grab top-most z-order: re-assert
            self.ov.keep_on_top()

    def close(self):
        self.ov.close()


class Hotkeys:
    VK = {"F7": 0x76, "F8": 0x77, "F9": 0x78, "F10": 0x79}

    def __init__(self):
        self.ok = sys.platform == "win32"
        self.prev = {}
        if self.ok:
            import ctypes
            self.get = ctypes.windll.user32.GetAsyncKeyState

    def pressed(self, name) -> bool:
        if not self.ok:
            return False
        down = bool(self.get(self.VK[name]) & 0x8000)
        was = self.prev.get(name, False)
        self.prev[name] = down
        return down and not was


class LiveRuntime:
    def __init__(self, cfg: AppConfig, source, presenter, analyzer=None, show_overlay=True, sysmon=None):
        self.cfg = cfg
        self.src = source
        self.presenter = presenter
        if analyzer is None:
            from fctac.vision.analyzer import VisionAnalyzer
            analyzer = VisionAnalyzer(cfg)
        self.an = analyzer
        self.overlay: Optional[OverlayRenderer] = None
        self.stats = LatencyStats(window=100000)
        self.rate = RateMeter(2.0)
        self.show_overlay = show_overlay
        self.keys = Hotkeys()
        self.sysmon = sysmon
        self.mask_overlay = getattr(source, "name", "") in ("dxgi", "mss")   # desktop capture sees our overlay
        self.recs = 0
        self.frames = 0

    def _debug_lines(self, fa: T.FrameAnalysis, e2e_ms: float, queue_ms: float) -> list:
        s = self.stats.summary()
        tot = s.get("e2e", {})
        lines = [f"processed {self.rate.rate:5.1f} fps  capture {self.src.buffer.rate_in.rate:5.1f} fps  dropped {self.src.buffer.dropped}",
                 f"latency e2e {e2e_ms:5.1f} ms (mean {tot.get('mean', 0):.1f}, p95 {tot.get('p95', 0):.1f})  frame age {queue_ms:4.1f} ms"]
        for k in ("radar", "detect", "calib", "track", "state", "decision"):
            if k in fa.timings_ms:
                lines.append(f"  {k:<8s} {fa.timings_ms[k]:5.2f} ms")
        rd = getattr(self.an, "radar", None)
        if hasattr(rd, "us_shape"):
            you = {0: "triangles", 1: "circles"}.get(rd.us_shape, "not decided yet - press F7")
            lines.append(f"radar: your team = {you}  (F7 swaps), position {getattr(rd, 'align_status', '-')}")
        if self.sysmon is not None:
            lines.append(self.sysmon.line())
        st = fa.state
        if st is not None:
            me = st.controlled
            b = st.ball
            lines.append(f"ball {'-' if b is None else f'({b.pos[0]:.0f},{b.pos[1]:.0f}) c={b.confidence:.2f}'}  "
                         f"ctrl {'-' if me is None else me.label}  calib {st.calib_conf:.2f}")
        r = fa.recommendation
        lines.append(f"rec {r.status} {r.action.text() if r.action else ''} {r.confidence * 100:.0f}% {r.reason}")
        for a in r.alternatives[:2]:
            lines.append(f"  alt {a.text()} {a.score:+.3f}")
        return lines

    def run(self, seconds: float = 0.0) -> dict:
        src = self.src.start()
        t_stop = time.perf_counter() + seconds if seconds > 0 else None
        last_id = -1
        prev_alpha = None
        idx = 0
        try:
            while True:
                if t_stop is not None and time.perf_counter() > t_stop:
                    break
                if self.keys.pressed("F10"):
                    break
                if self.keys.pressed("F8"):
                    self.show_overlay = not self.show_overlay
                if self.keys.pressed("F9"):
                    self.cfg.overlay.show_debug = not self.cfg.overlay.show_debug
                if self.keys.pressed("F7") and hasattr(getattr(self.an, "radar", None), "swap"):
                    self.an.radar.swap()
                f = src.buffer.get(last_id, timeout=0.5)
                if f is None:
                    if src.error:
                        raise RuntimeError(f"capture failed: {src.error}")
                    if getattr(src, "finished", None) is not None and src.finished.is_set():
                        break
                    continue
                last_id = f.frame_id
                t0 = time.perf_counter()
                queue_ms = (t0 - f.t_capture) * 1000.0
                img = f.image
                if self.overlay is None:
                    self.overlay = OverlayRenderer(img.shape[1], img.shape[0], self.cfg.overlay)
                if self.mask_overlay and prev_alpha is not None:
                    img = img.copy()
                    img[prev_alpha] = (66, 140, 62)          # our own overlay pixels -> neutral grass
                fa = self.an.process(img, idx, f.t_capture)
                idx += 1
                t1 = time.perf_counter()
                if self.show_overlay:
                    dbg = self._debug_lines(fa, self.stats.mean("e2e"), queue_ms) if self.cfg.overlay.show_debug else None
                    canvas = self.overlay.render(fa, dbg)
                else:
                    self.overlay.canvas.clear()
                    canvas = self.overlay.canvas
                t2 = time.perf_counter()
                self.presenter.present(canvas, f.image)
                t3 = time.perf_counter()
                if self.mask_overlay:
                    prev_alpha = canvas.img[..., 3] > 0
                self.rate.tick(t3)
                self.frames += 1
                self.recs += int(fa.recommendation.status == T.STATUS_ACTIVE)
                tm = dict(fa.timings_ms)
                tm.update({"frame_age": queue_ms, "analysis": (t1 - t0) * 1000, "overlay_render": (t2 - t1) * 1000,
                           "present": (t3 - t2) * 1000, "e2e": (t3 - f.t_capture) * 1000})
                self.stats.add(tm)
                if isinstance(self.presenter, PreviewPresenter) and self.presenter.key == 27:
                    break
        finally:
            src.stop()
            self.presenter.close()
        return self.report()

    def report(self) -> dict:
        s = self.stats.summary()
        b = self.src.buffer
        return {"frames_processed": self.frames, "frames_captured": b.produced, "frames_dropped": b.dropped,
                "active_recommendation_rate": round(self.recs / max(self.frames, 1), 3),
                "latency_ms": {k: {kk: round(vv, 2) for kk, vv in v.items() if kk != "n"} for k, v in s.items()}}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="")
    ap.add_argument("--calib", default="")
    ap.add_argument("--video", default="", help="use a video file as the live source (testing)")
    ap.add_argument("--speed", type=float, default=1.0, help="video source speed (stress test with >1)")
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--seconds", type=float, default=0)
    ap.add_argument("--report", default="")
    ap.add_argument("--ranker", default="", help="heuristic | registry | path/to/ranker.json")
    ap.add_argument("--predictor", default="", help="kinematic | registry | path/to/predictor.json")
    a = ap.parse_args(argv)
    from fctac.replay.viewer import runtime_overrides
    ov = runtime_overrides(a)
    cfg = load_config(a.config or None, ov)
    cfg.overlay.show_debug = a.debug
    if sys.platform == "win32":
        from fctac.overlay.win32 import set_dpi_aware
        set_dpi_aware()
    from fctac.capture.windows import open_source
    if a.video:
        from fctac.capture.base import VideoFileSource
        src = VideoFileSource(a.video, speed=a.speed)
    else:
        src = open_source(cfg.capture)
    from fctac.vision.analyzer import VisionAnalyzer
    if a.calib or a.config:
        w, h = getattr(src, "size", (0, 0)) or (0, 0)
        an = VisionAnalyzer.from_files(a.config, a.calib or None, w, h, overrides=ov)
    else:
        an = VisionAnalyzer(cfg)
    if a.headless:
        presenter = NullPresenter()
    elif a.preview or sys.platform != "win32" or a.video:
        presenter = PreviewPresenter()
    else:
        from fctac.overlay.win32 import client_rect_on_screen
        if getattr(src, "hwnd", None):
            x, y, w, h = client_rect_on_screen(src.hwnd)
        else:
            x, y = (cfg.capture.region[:2] if cfg.capture.region else (0, 0))
            w, h = src.size
        presenter = Win32Presenter(x, y, w, h)
    from fctac.benchmark.sysmon import SysMonitor
    mon = SysMonitor().start()
    rt = LiveRuntime(cfg, src, presenter, an, sysmon=mon)
    print(f"FC27 assistant running (capture: {getattr(src, 'name', '?')}). "
          "F7 = swap your radar team, F8 = overlay on/off, F9 = debug info, F10 = quit.", flush=True)
    rep = rt.run(a.seconds)
    rep["system"] = mon.latest
    mon.stop()
    print(json.dumps(rep, indent=1))
    if a.report:
        with open(a.report, "w") as f:
            json.dump(rep, f, indent=1)


if __name__ == "__main__":
    main()
