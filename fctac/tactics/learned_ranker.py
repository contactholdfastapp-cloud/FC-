"""Learned candidate ranker (runtime, numpy only).

Two small MLPs trained by fctac.training.ranker:
  * success model  p(success | candidate features)   -> replaces physics p
  * policy prior   s(candidate features)             -> softmax over candidates
Final score = p * value_if_success - (1 - p) * turnover_cost + beta * log pi.
The ranker is only deployed if it beats the heuristic on held-out data.
"""
from __future__ import annotations

import json

import numpy as np


class MLP:
    """Dense ReLU network stored as JSON {"layers": [[W, b], ...], "mean": [...], "std": [...]}."""

    def __init__(self, spec: dict):
        self.layers = [(np.asarray(W, np.float32), np.asarray(b, np.float32)) for W, b in spec["layers"]]
        self.mean = np.asarray(spec["mean"], np.float32)
        self.std = np.asarray(spec["std"], np.float32)

    def __call__(self, X: np.ndarray) -> np.ndarray:
        h = (np.asarray(X, np.float32) - self.mean) / self.std
        for i, (W, b) in enumerate(self.layers):
            h = h @ W + b
            if i < len(self.layers) - 1:
                h = np.maximum(h, 0.0)
        return h[:, 0]


class LearnedRanker:
    name = "learned"

    MODES = ("learned_ev", "platt_ev", "policy_only")

    def __init__(self, path: str, beta: float | None = None, mode: str | None = None):
        spec = json.load(open(path))
        self.success = MLP(spec["success"])
        self.policy = MLP(spec["policy"]) if spec.get("policy") else None
        self.platt = spec.get("platt", [1.0, 0.0])
        mode = mode or spec.get("mode", "learned_ev")
        # "x+policy(b=0.01)" style modes from the training report
        if "+policy" in mode:
            base, b = mode.split("+policy")
            beta = float(b.strip("()").split("=")[-1]) if beta is None else beta
            mode = base
        self.mode = mode if mode in self.MODES else "learned_ev"
        self.beta = (spec.get("beta", 0.0) if beta is None else beta) or 0.0
        self.meta = spec.get("meta", {})

    def p_success(self, F: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(self.success(F), -30, 30)))

    def p_platt(self, p_phys: np.ndarray) -> np.ndarray:
        p = np.clip(p_phys, 1e-4, 1 - 1e-4)
        return 1.0 / (1.0 + np.exp(-(self.platt[0] * np.log(p / (1 - p)) + self.platt[1])))

    def score(self, cands: list, st=None) -> np.ndarray:
        if not cands:
            return np.zeros(0)
        F = np.stack([a.features for a in cands])
        if self.mode == "policy_only" and self.policy is not None:
            return self.policy(F)
        p = self.p_platt(np.array([a.p_success for a in cands])) if self.mode == "platt_ev" else self.p_success(F)
        v = np.array([a.value_success for a in cands])
        c = np.array([a.detail.get("cost", 0.0) for a in cands])
        ev = p * v - (1.0 - p) * c
        for a, pi in zip(cands, p):
            a.p_success = float(pi)
        if self.policy is not None and self.beta > 0:
            ev = ev + self.beta * self.policy(F)
        return ev
