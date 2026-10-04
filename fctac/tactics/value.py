"""Position value in expected-goal units (xT-like), attack-aligned metres.

V(p): probability-weighted value of having the ball at p (our attack).
C(p): cost of losing the ball at p (= opponent's value at the mirrored point).
Every candidate action is scored in these units, so passes, dribbles and
shots are directly comparable.
"""
from __future__ import annotations

import numpy as np

from fctac.types import PITCH_LENGTH as L, PITCH_WIDTH as W

GOAL_W = 7.32


def xg(p) -> np.ndarray:
    """Open-play shot xG from position only (vectorised)."""
    p = np.atleast_2d(np.asarray(p, dtype=np.float64))
    dx = np.maximum(L - p[:, 0], 0.05)
    dy = np.abs(p[:, 1] - W / 2)
    d = np.hypot(dx, dy)
    a = np.arctan2(GOAL_W * dx, dx * dx + dy * dy - (GOAL_W / 2) ** 2)
    a = np.where(a < 0, a + np.pi, a)
    z = -1.2 - 0.11 * d + 1.6 * a
    return 1.0 / (1.0 + np.exp(-z))


XG_WEIGHT = 0.45   # share of the shot chance credited to merely *having* the ball at p:
                   # V(p) must stay below xG(p) where shooting is the best option, otherwise
                   # holding the ball "double counts" the shot and shooting never wins


def zone_value(p) -> np.ndarray:
    p = np.atleast_2d(np.asarray(p, dtype=np.float64))
    x = np.clip(p[:, 0], 0, L) / L
    return 0.01 + 0.06 * x ** 2 + XG_WEIGHT * xg(p)


def turnover_cost(p) -> np.ndarray:
    p = np.atleast_2d(np.asarray(p, dtype=np.float64))
    mirrored = np.stack([L - p[:, 0], W - p[:, 1]], 1)
    return zone_value(mirrored) + 0.01
