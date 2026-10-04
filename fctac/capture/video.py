"""Video-file frame source for replay mode (random access + small LRU cache)."""
from __future__ import annotations

from collections import OrderedDict

import cv2
import numpy as np


class VideoSource:
    def __init__(self, path: str, cache: int = 48):
        self.path = path
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"cannot open video: {path}")
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS)) or 30.0
        self.n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._next = 0            # index the decoder will return on read()
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._cache_n = cache

    def __len__(self):
        return self.n

    def get(self, idx: int) -> np.ndarray | None:
        if idx < 0 or (self.n > 0 and idx >= self.n):
            return None
        if idx in self._cache:
            self._cache.move_to_end(idx)
            return self._cache[idx]
        if idx != self._next:
            if idx < self._next or idx - self._next > 90:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                self._next = idx
            while self._next < idx:          # decode forward (faster than seeking short gaps)
                if not self.cap.grab():
                    return None
                self._next += 1
        ok, frame = self.cap.read()
        if not ok:
            if self.n <= 0 or idx < self.n:
                self.n = idx              # stream shorter than its header claimed
            return None
        self._next = idx + 1
        self._cache[idx] = frame
        if len(self._cache) > self._cache_n:
            self._cache.popitem(last=False)
        return frame

    def time_of(self, idx: int) -> float:
        return idx / self.fps

    def close(self):
        self.cap.release()
