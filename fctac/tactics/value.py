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


def _load_xt():
    import json
    import os
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "open_xt_12x8_v1.json")
    try:
        return np.array(json.load(open(path)), float)          # (8 rows across, 12 cols along)
    except OSError:
        return None


# Expected Threat grid (Karun Singh, "Introducing Expected Threat", open 12x8 grid fitted on
# real match event data): probability that possession at a zone leads to a goal within
# the next few actions.  Real-data value surface instead of a hand-tuned one.
XT = _load_xt()


def xt(p) -> np.ndarray:
    """Bilinear-interpolated xT at attack-aligned pitch points."""
    p = np.atleast_2d(np.asarray(p, dtype=np.float64))
    if XT is None:
        x = np.clip(p[:, 0], 0, L) / L
        return 0.006 + 0.03 * x ** 3
    ny, nx = XT.shape
    gx = np.clip(p[:, 0] / L * nx - 0.5, 0, nx - 1)
    gy = np.clip(p[:, 1] / W * ny - 0.5, 0, ny - 1)
    x0, y0 = np.floor(gx).astype(int), np.floor(gy).astype(int)
    x1, y1 = np.minimum(x0 + 1, nx - 1), np.minimum(y0 + 1, ny - 1)
    fx, fy = gx - x0, gy - y0
    return (XT[y0, x0] * (1 - fx) * (1 - fy) + XT[y0, x1] * fx * (1 - fy)
            + XT[y1, x0] * (1 - fx) * fy + XT[y1, x1] * fx * fy)


def zone_value(p) -> np.ndarray:
    """Value of having the ball at p: real-data xT, never below a fraction of a direct shot
    there (xT averages over contested possessions; inside the box a clean shot is the floor)."""
    p = np.atleast_2d(np.asarray(p, dtype=np.float64))
    return np.maximum(xt(p), XG_WEIGHT * xg(p))


def turnover_cost(p) -> np.ndarray:
    p = np.atleast_2d(np.asarray(p, dtype=np.float64))
    mirrored = np.stack([L - p[:, 0], W - p[:, 1]], 1)
    return zone_value(mirrored) + 0.01
