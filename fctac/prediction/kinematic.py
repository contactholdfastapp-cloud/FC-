"""Baseline movement prediction: damped constant velocity.

pos(t) = p + v * tau * (1 - exp(-t / tau))

tau -> inf is constant velocity; finite tau models players not holding a
direction for long.  tau is fitted on data (see prediction.evaluate); the
learned predictor must beat this baseline to be deployed.
"""
from __future__ import annotations

import numpy as np

from fctac.types import PITCH_SIZE

HORIZONS = (0.25, 0.5, 0.75, 1.0, 1.5)


class KinematicPredictor:
    name = "kinematic"

    def __init__(self, tau: float = 1.6, horizons=HORIZONS):
        self.tau = tau
        self.horizons = tuple(horizons)

    def factor(self, t) -> np.ndarray:
        t = np.asarray(t, dtype=np.float64)
        if self.tau <= 0:
            return t
        return self.tau * (1.0 - np.exp(-t / self.tau))

    def predict_point(self, pos: np.ndarray, vel: np.ndarray, t) -> np.ndarray:
        """Position after t seconds; t may be a scalar or broadcast against pos."""
        p = pos + vel * self.factor(t)
        return np.clip(p, [-2, -2], PITCH_SIZE + 2)

    def predict(self, pos: np.ndarray, vel: np.ndarray, history=None) -> np.ndarray:
        """(N,2),(N,2) -> (N,H,2) predicted positions at self.horizons."""
        pos = np.atleast_2d(pos)
        vel = np.atleast_2d(vel)
        f = self.factor(np.array(self.horizons))
        out = pos[:, None, :] + vel[:, None, :] * f[None, :, None]
        return np.clip(out, -2, PITCH_SIZE + 2)

    def annotate(self, state) -> None:
        """Fill PlayerState.predicted for every player."""
        if not state.players:
            return
        pos = np.array([p.pos for p in state.players])
        vel = np.array([p.vel for p in state.players])
        pred = self.predict(pos, vel)
        for i, p in enumerate(state.players):
            p.predicted = {h: pred[i, k] for k, h in enumerate(self.horizons)}
