"""Pitch geometry in raw pitch metres (see fctac.types for conventions)."""
from __future__ import annotations

import numpy as np

from fctac.types import PITCH_LENGTH as L, PITCH_WIDTH as W

GOAL_WIDTH = 7.32
GOAL_HEIGHT = 2.44
PEN_DEPTH, PEN_WIDTH = 16.5, 40.32
SIX_DEPTH, SIX_WIDTH = 5.5, 18.32
CIRCLE_R = 9.15
PEN_SPOT = 11.0


def line_segments() -> list[tuple[np.ndarray, np.ndarray]]:
    """Straight pitch markings as (start, end) pairs in raw metres."""
    segs = []

    def add(x0, y0, x1, y1):
        segs.append((np.array([x0, y0], float), np.array([x1, y1], float)))

    # outline + halfway line
    add(0, 0, L, 0)
    add(0, W, L, W)
    add(0, 0, 0, W)
    add(L, 0, L, W)
    add(L / 2, 0, L / 2, W)
    for x_goal, s in ((0.0, 1.0), (L, -1.0)):
        y0, y1 = (W - PEN_WIDTH) / 2, (W + PEN_WIDTH) / 2
        xd = x_goal + s * PEN_DEPTH
        add(x_goal, y0, xd, y0)
        add(x_goal, y1, xd, y1)
        add(xd, y0, xd, y1)
        y0, y1 = (W - SIX_WIDTH) / 2, (W + SIX_WIDTH) / 2
        xd = x_goal + s * SIX_DEPTH
        add(x_goal, y0, xd, y0)
        add(x_goal, y1, xd, y1)
        add(xd, y0, xd, y1)
    return segs


def arcs(n: int = 48) -> list[np.ndarray]:
    """Curved markings as polylines (centre circle, penalty arcs)."""
    out = []
    a = np.linspace(0, 2 * np.pi, n * 2, endpoint=True)
    out.append(np.stack([L / 2 + CIRCLE_R * np.cos(a), W / 2 + CIRCLE_R * np.sin(a)], 1))
    # penalty arcs: part of circle around the spot outside the box
    half = np.arccos((PEN_DEPTH - PEN_SPOT) / CIRCLE_R)
    a = np.linspace(-half, half, n)
    out.append(np.stack([PEN_SPOT + CIRCLE_R * np.cos(a), W / 2 + CIRCLE_R * np.sin(a)], 1))
    out.append(np.stack([L - PEN_SPOT - CIRCLE_R * np.cos(a), W / 2 + CIRCLE_R * np.sin(a)], 1))
    return out


def sample_line_points(step: float = 0.5) -> np.ndarray:
    """Dense points on every marking; used for model-based calibration."""
    pts = []
    for a, b in line_segments():
        n = max(2, int(np.linalg.norm(b - a) / step) + 1)
        t = np.linspace(0, 1, n)[:, None]
        pts.append(a + (b - a) * t)
    for poly in arcs():
        seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
        cum = np.concatenate([[0], np.cumsum(seg)])
        s = np.arange(0, cum[-1], step)
        pts.append(np.stack([np.interp(s, cum, poly[:, 0]), np.interp(s, cum, poly[:, 1])], 1))
    return np.concatenate(pts, 0)


# Named landmarks used for manual calibration (raw metres).
LANDMARKS = {
    "corner_near_left": (0.0, 0.0),
    "corner_far_left": (0.0, W),
    "corner_near_right": (L, 0.0),
    "corner_far_right": (L, W),
    "halfway_near": (L / 2, 0.0),
    "halfway_far": (L / 2, W),
    "centre_spot": (L / 2, W / 2),
    "circle_near": (L / 2, W / 2 - CIRCLE_R),
    "circle_far": (L / 2, W / 2 + CIRCLE_R),
    "left_box_near_corner": (PEN_DEPTH, (W - PEN_WIDTH) / 2),
    "left_box_far_corner": (PEN_DEPTH, (W + PEN_WIDTH) / 2),
    "left_box_near_goalline": (0.0, (W - PEN_WIDTH) / 2),
    "left_box_far_goalline": (0.0, (W + PEN_WIDTH) / 2),
    "left_six_near_corner": (SIX_DEPTH, (W - SIX_WIDTH) / 2),
    "left_six_far_corner": (SIX_DEPTH, (W + SIX_WIDTH) / 2),
    "left_pen_spot": (PEN_SPOT, W / 2),
    "right_box_near_corner": (L - PEN_DEPTH, (W - PEN_WIDTH) / 2),
    "right_box_far_corner": (L - PEN_DEPTH, (W + PEN_WIDTH) / 2),
    "right_box_near_goalline": (L, (W - PEN_WIDTH) / 2),
    "right_box_far_goalline": (L, (W + PEN_WIDTH) / 2),
    "right_six_near_corner": (L - SIX_DEPTH, (W - SIX_WIDTH) / 2),
    "right_six_far_corner": (L - SIX_DEPTH, (W + SIX_WIDTH) / 2),
    "right_pen_spot": (L - PEN_SPOT, W / 2),
}


def in_pitch(p: np.ndarray, margin: float = 0.0) -> bool:
    return (-margin <= p[0] <= L + margin) and (-margin <= p[1] <= W + margin)
