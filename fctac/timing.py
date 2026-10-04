"""Lightweight latency instrumentation.

``StageTimer`` measures named pipeline stages for one frame; ``LatencyStats``
keeps rolling statistics so the debug panel and benchmarks report measured
numbers (never assumed ones).
"""
from __future__ import annotations

import time
from collections import defaultdict, deque
from contextlib import contextmanager

import numpy as np

now = time.perf_counter


class StageTimer:
    __slots__ = ("ms",)

    def __init__(self):
        self.ms: dict[str, float] = {}

    @contextmanager
    def stage(self, name: str):
        t0 = now()
        try:
            yield
        finally:
            self.ms[name] = self.ms.get(name, 0.0) + (now() - t0) * 1000.0

    def add(self, name: str, ms: float):
        self.ms[name] = self.ms.get(name, 0.0) + ms


class LatencyStats:
    """Rolling window of per-stage timings."""

    def __init__(self, window: int = 300):
        self.window = window
        self.samples: dict[str, deque] = defaultdict(lambda: deque(maxlen=window))

    def add(self, timings_ms: dict):
        for k, v in timings_ms.items():
            self.samples[k].append(float(v))

    def summary(self) -> dict:
        out = {}
        for k, d in self.samples.items():
            if not d:
                continue
            a = np.fromiter(d, dtype=np.float64)
            out[k] = {
                "mean": float(a.mean()),
                "p50": float(np.percentile(a, 50)),
                "p95": float(np.percentile(a, 95)),
                "max": float(a.max()),
                "n": int(a.size),
            }
        return out

    def mean(self, key: str) -> float:
        d = self.samples.get(key)
        return float(np.mean(d)) if d else 0.0


class RateMeter:
    """Events-per-second over a sliding time window."""

    def __init__(self, window_s: float = 1.0):
        self.window_s = window_s
        self.ts: deque = deque()

    def tick(self, t: float | None = None):
        t = now() if t is None else t
        self.ts.append(t)
        while self.ts and t - self.ts[0] > self.window_s:
            self.ts.popleft()

    @property
    def rate(self) -> float:
        if len(self.ts) < 2:
            return 0.0
        span = self.ts[-1] - self.ts[0]
        return (len(self.ts) - 1) / span if span > 0 else 0.0
