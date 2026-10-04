"""Windows screen-capture backends (standard OS capture APIs only).

* ``WGCSource``  - Windows Graphics Capture of the FC 27 *window* via the
  ``windows-capture`` package.  Preferred: GPU-composited, low overhead, and
  window capture contains only the game (never our overlay).
* ``DXGISource`` - DXGI Desktop Duplication of a monitor via ``dxcam``.  Also
  fast; captures the composed desktop, so the runtime masks the overlay's
  own pixels before analysis.
* ``MSSSource``  - GDI BitBlt via ``mss``; slow fallback for testing.

None of these touch the game process; they read what is on screen like any
recording/streaming software.
"""
from __future__ import annotations

import sys
import time
from typing import Optional

import numpy as np

from fctac.capture.base import FrameSource


def _crop(img: np.ndarray, region: tuple) -> np.ndarray:
    if not region:
        return img
    x, y, w, h = region
    return img[y:y + h, x:x + w]


class WGCSource(FrameSource):
    name = "wgc"

    def __init__(self, window_title: str, region: tuple = (), draw_border: bool = False):
        super().__init__()
        from windows_capture import WindowsCapture   # pip install windows-capture
        from fctac.overlay import win32
        hwnd = win32.find_window((window_title,))
        if not hwnd:
            raise RuntimeError(f"no visible window containing '{window_title}' (is FC 27 running in windowed/borderless mode?)")
        import ctypes
        n = win32.user32.GetWindowTextLengthW(hwnd)
        buf = ctypes.create_unicode_buffer(n + 1)
        win32.user32.GetWindowTextW(hwnd, buf, n + 1)
        self.title = buf.value
        self.hwnd = hwnd
        self.region = region
        if not region:
            # WGC captures the whole visible window; analyse (and overlay) only the
            # client area so capture pixels and overlay pixels line up exactly
            fx, fy, fw, fh = win32.frame_bounds(hwnd)
            cx, cy, cw, ch = win32.client_rect_on_screen(hwnd)
            if (cw, ch) != (fw, fh) and cw > 0 and ch > 0:
                self.region = (max(0, cx - fx), max(0, cy - fy), cw, ch)
        self._cap = WindowsCapture(cursor_capture=False, draw_border=draw_border, window_name=self.title)
        self._size = (0, 0)
        self._control = None
        import cv2
        cv2_cvt = cv2.cvtColor
        code = cv2.COLOR_BGRA2BGR

        @self._cap.event
        def on_frame_arrived(frame, capture_control):
            t = time.perf_counter()
            img = cv2_cvt(_crop(frame.frame_buffer, self.region), code)   # copies out of the shared buffer
            self._size = (img.shape[1], img.shape[0])
            self.buffer.put(img, t)
            if self._stop.is_set():
                capture_control.stop()

        @self._cap.event
        def on_closed():
            self._stop.set()

    def start(self):
        self._control = self._cap.start_free_threaded()
        return self

    def stop(self):
        self._stop.set()
        if self._control is not None:
            try:
                self._control.stop()
            except Exception:
                pass

    @property
    def size(self):
        return self._size


class DXGISource(FrameSource):
    name = "dxgi"

    def __init__(self, monitor: int = 0, region: tuple = (), target_fps: int = 120):
        super().__init__()
        import dxcam   # pip install dxcam
        self.cam = dxcam.create(output_idx=monitor, output_color="BGR")
        self.region = None
        if region:
            x, y, w, h = region
            self.region = (x, y, x + w, y + h)
            self._size = (w, h)
        else:
            self._size = (self.cam.width, self.cam.height)
        self.min_dt = 1.0 / target_fps

    def _run(self):
        last = 0.0
        while not self._stop.is_set():
            img = self.cam.grab(region=self.region)
            t = time.perf_counter()
            if img is None:          # no new desktop frame yet
                time.sleep(0.001)
                continue
            self.buffer.put(img, t)
            wait = self.min_dt - (t - last)
            last = t
            if wait > 0:
                time.sleep(wait)

    def stop(self):
        super().stop()
        try:
            self.cam.release()
        except Exception:
            pass

    @property
    def size(self):
        return self._size


class MSSSource(FrameSource):
    name = "mss"

    def __init__(self, monitor: int = 0, region: tuple = (), target_fps: int = 60):
        super().__init__()
        import mss   # pip install mss
        self.mss = mss
        with mss.mss() as s:
            mon = s.monitors[monitor + 1]
        if region:
            x, y, w, h = region
            self.box = {"left": mon["left"] + x, "top": mon["top"] + y, "width": w, "height": h}
        else:
            self.box = {"left": mon["left"], "top": mon["top"], "width": mon["width"], "height": mon["height"]}
        self.min_dt = 1.0 / target_fps

    def _run(self):
        with self.mss.mss() as s:
            while not self._stop.is_set():
                t0 = time.perf_counter()
                img = np.asarray(s.grab(self.box))[:, :, :3].copy()
                self.buffer.put(img, time.perf_counter())
                wait = self.min_dt - (time.perf_counter() - t0)
                if wait > 0:
                    time.sleep(wait)

    @property
    def size(self):
        return self.box["width"], self.box["height"]


def open_source(cfg, video: Optional[str] = None) -> FrameSource:
    """Pick a capture backend from CaptureConfig ('auto' tries WGC, DXGI, MSS)."""
    if video or cfg.backend == "video":
        from fctac.capture.base import VideoFileSource
        return VideoFileSource(video)
    if sys.platform != "win32":
        raise RuntimeError("live screen capture backends are Windows-only; use --video for testing")
    errors = []
    order = [cfg.backend] if cfg.backend != "auto" else ["wgc", "dxgi", "mss"]
    for b in order:
        try:
            if b == "wgc":
                return WGCSource(cfg.window_title, tuple(cfg.region))
            if b == "dxgi":
                return DXGISource(cfg.monitor, tuple(cfg.region), cfg.target_fps)
            if b == "mss":
                return MSSSource(cfg.monitor, tuple(cfg.region), cfg.target_fps)
        except Exception as e:     # try the next backend
            errors.append(f"{b}: {e}")
    raise RuntimeError("no capture backend available:\n  " + "\n  ".join(errors))
