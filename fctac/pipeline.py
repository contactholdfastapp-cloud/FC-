"""Analyzers: frame -> FrameAnalysis.

* ``OracleAnalyzer`` uses synthetic ground truth instead of perception, to
  develop/evaluate tactics in isolation (and to measure perception cost).
* ``VisionAnalyzer`` (fctac.vision.analyzer) is the real pixels-only pipeline.

Both expose ``process(frame, idx, t) -> FrameAnalysis`` and ``reset()``.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from fctac import types as T
from fctac.prediction.kinematic import KinematicPredictor
from fctac.state.gt import state_from_gt
from fctac.tactics.engine import TacticsEngine
from fctac.timing import StageTimer


class OracleAnalyzer:
    name = "oracle"

    def __init__(self, gt: list[dict], engine: Optional[TacticsEngine] = None, predictor=None,
                 visible_only: bool = False, pos_noise: float = 0.0, seed: int = 0):
        self.gt = gt
        self.engine = engine or TacticsEngine()
        self.pred = predictor or KinematicPredictor()
        self.visible_only = visible_only
        self.pos_noise = pos_noise
        self.rng = np.random.default_rng(seed)

    def reset(self):
        self.engine.reset()

    def process(self, frame: Optional[np.ndarray], idx: int, t: float) -> T.FrameAnalysis:
        tm = StageTimer()
        with tm.stage("state"):
            st = state_from_gt(self.gt[idx], rng=self.rng, pos_noise=self.pos_noise,
                               visible_only=self.visible_only)
        with tm.stage("prediction"):
            self.pred.annotate(st)
        with tm.stage("decision"):
            cands, rec = self.engine.decide(st)
        tm.ms["total"] = sum(tm.ms.values())
        return T.FrameAnalysis(frame=idx, t=t, state=st, candidates=cands, recommendation=rec, timings_ms=tm.ms)
