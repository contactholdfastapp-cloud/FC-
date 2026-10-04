"""Capture primitives: a latest-frame buffer that never queues stale frames.

Producers (capture backends) overwrite a single slot; the consumer always
takes the newest frame.  Frames the consumer was too slow for are dropped
and counted -- freshness beats completeness for a live assistant.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from fctac.timing import RateMeter


@dataclass
class CapturedFrame:
    image: np.ndarray        # BGR (or BGRA) pixels
    t_capture: float         # perf_counter() when the frame became available to us
    frame_id: int


class LatestFrameBuffer:
    def __init__(self):
        self._cond = threading.Condition()
        self._frame: Optional[CapturedFrame] = None
        self._next_id = 0
        self._taken = -1             # id of the last frame handed to the consumer
        self.dropped = 0
        self.produced = 0
        self.consumed = 0
        self.rate_in = RateMeter()

    def put(self, image: np.ndarray, t_capture: Optional[float] = None):
        with self._cond:
            if self._frame is not None and self._frame.frame_id > self._taken:
                # previous frame was never consumed -> dropped in favour of this one
                self.dropped += 1
            self._frame = CapturedFrame(image, time.perf_counter() if t_capture is None else t_capture, self._next_id)
            self._next_id += 1
            self.produced += 1
            self.rate_in.tick()
            self._cond.notify_all()

    def get(self, after_id: int = -1, timeout: float = 0.5) -> Optional[CapturedFrame]:
        """Newest frame with id > after_id (blocks up to ``timeout``)."""
        deadline = time.perf_counter() + timeout
        with self._cond:
            while self._frame is None or self._frame.frame_id <= after_id:
                left = deadline - time.perf_counter()
                if left <= 0:
                    return None
                self._cond.wait(left)
            f = self._frame
            self._taken = f.frame_id
            self.consumed += 1
            return f


class FrameSource:
    """Base class: a thread (or OS callback) pushing frames into ``self.buffer``."""

    name = "base"

    def __init__(self):
        self.buffer = LatestFrameBuffer()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.error: Optional[str] = None

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._run_safe, name=f"capture-{self.name}", daemon=True)
        self._thread.start()
        return self

    def _run_safe(self):
        try:
            self._run()
        except Exception as e:      # surfaced to the runtime instead of dying silently
            self.error = f"{type(e).__name__}: {e}"

    def _run(self):
        raise NotImplementedError

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    @property
    def size(self) -> tuple[int, int]:
        raise NotImplementedError


class VideoFileSource(FrameSource):
    """Plays a video file in real time as if it were a live capture (testing
    the live runtime on any OS).  Slow consumers make it drop frames, exactly
    like a real capture would."""

    name = "video"

    def __init__(self, path: str, speed: float = 1.0, loop: bool = False):
        super().__init__()
        import cv2
        self.cv2 = cv2
        self.path = path
        self.speed = speed
        self.loop = loop
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise FileNotFoundError(path)
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        self.finished = threading.Event()
        self.video_index_of: dict[int, int] = {}     # buffer frame_id -> video frame index

    @property
    def size(self):
        return self._w, self._h

    def _run(self):
        cap = self.cv2.VideoCapture(self.path)
        t0 = time.perf_counter()
        k = 0
        while not self._stop.is_set():
            ok, frame = cap.read()
            if not ok:
                if self.loop:
                    cap.set(self.cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break
            due = t0 + k / (self.fps * self.speed)
            dt = due - time.perf_counter()
            if dt > 0:
                time.sleep(dt)
            self.video_index_of[self.buffer._next_id] = k
            self.buffer.put(frame)
            k += 1
        cap.release()
        self.finished.set()
