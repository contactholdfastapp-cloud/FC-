"""Learned movement predictor (runtime, numpy).  Keeps a short per-player
history and predicts displacements at the standard horizons; same interface
as KinematicPredictor (``annotate``, ``predict_point``)."""
from __future__ import annotations

import json
from collections import defaultdict, deque

import numpy as np

from fctac.prediction.kinematic import HORIZONS, KinematicPredictor
from fctac.tactics.learned_ranker import MLP
from fctac.training.predictor import HIST_STEPS, features


class LearnedPredictor:
    name = "learned"

    def __init__(self, path: str, fps: float = 30.0):
        spec = json.load(open(path))
        self.mlp = MLP(spec)
        self.horizons = tuple(spec.get("horizons", HORIZONS))
        self.fallback = KinematicPredictor(tau=spec.get("tau_fallback", 1.6), horizons=self.horizons)
        self.hist: dict = defaultdict(lambda: deque(maxlen=64))   # id -> (t, pos)

    def predict_disp(self, X: np.ndarray) -> np.ndarray:
        out = np.stack([self.mlp_out(X)], 0)[0]
        return out.reshape(len(X), -1, 2)

    def mlp_out(self, X):
        h = (np.asarray(X, np.float32) - self.mlp.mean) / self.mlp.std
        for i, (Wm, b) in enumerate(self.mlp.layers):
            h = h @ Wm + b
            if i < len(self.mlp.layers) - 1:
                h = np.maximum(h, 0.0)
        return h

    def factor(self, t):
        return self.fallback.factor(t)

    def predict_point(self, pos, vel, t):
        return self.fallback.predict_point(pos, vel, t)

    def annotate(self, state) -> None:
        if not state.players:
            return
        rows, idx = [], []
        for i, p in enumerate(state.players):
            h = self.hist[p.id]
            h.append((state.t, p.pos.copy()))
            past = []
            for dt in HIST_STEPS:
                tgt = state.t - dt
                q = next((pp for tt, pp in reversed(h) if tt <= tgt + 1e-3), None)
                if q is None:
                    break
                past.append(q - p.pos)
            if len(past) == len(HIST_STEPS) and state.ball is not None:
                rows.append(features(np.array(past), p.vel, state.ball.pos - p.pos, state.ball.vel, p.team,
                                     state.possession, p.pos[0]))
                idx.append(i)
        fb = self.fallback.predict(np.array([p.pos for p in state.players]), np.array([p.vel for p in state.players]))
        for i, p in enumerate(state.players):
            p.predicted = {h: fb[i, k] for k, h in enumerate(self.horizons)}
        if rows:
            disp = self.predict_disp(np.array(rows))
            for j, i in enumerate(idx):
                p = state.players[i]
                p.predicted = {h: p.pos + disp[j, k] for k, h in enumerate(self.horizons)}
