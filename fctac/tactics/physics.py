"""Ball-flight and player-reach physics for pass evaluation (vectorised).

The core question for every pass is a race: does the ball reach each point of
its path before any opponent can, and does the receiver reach the end point
before the ball (and before opponents)?  Arrival-time margins are turned into
probabilities with a logistic of width ``sigma_t`` (pitch-control style).
All candidates of a frame are evaluated in one batched numpy call.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PhysicsConfig:
    vmax: float = 7.8            # player top speed (m/s)
    reaction: float = 0.25       # s before a player changes course
    sigma_t: float = 0.30        # s, uncertainty of time comparisons
    control_r: float = 1.0       # m, reach radius
    pass_decel: float = 2.8      # m/s^2 rolling deceleration
    arrive_pass: float = 6.0     # desired arrival speed, ground pass
    arrive_through: float = 4.0
    v_min: float = 8.0
    v_max: float = 26.0
    lob_speed: float = 25.0      # flight time = 1 + d / lob_speed
    lofted_accuracy: float = 0.85
    path_samples: int = 14


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def time_to_reach(pos: np.ndarray, vel: np.ndarray, targets: np.ndarray, cfg: PhysicsConfig) -> np.ndarray:
    """(N,2) players, (M,2) targets -> (N,M) seconds."""
    pos = np.atleast_2d(pos)
    vel = np.atleast_2d(vel)
    targets = np.atleast_2d(targets)
    p_react = pos + vel * cfg.reaction
    d = np.linalg.norm(targets[None, :, :] - p_react[:, None, :], axis=2)
    d = np.maximum(d - cfg.control_r, 0.0)
    return cfg.reaction + d / cfg.vmax


def ground_v0(dist, arrive: float, cfg: PhysicsConfig):
    return np.clip(np.sqrt(arrive ** 2 + 2 * cfg.pass_decel * np.asarray(dist, float)), cfg.v_min, cfg.v_max)


def ground_times(s, v0, cfg: PhysicsConfig):
    """Time and speed of a rolling ball at path distances s (broadcasting)."""
    a = cfg.pass_decel
    s = np.asarray(s, float)
    v0 = np.asarray(v0, float)
    disc = v0 * v0 - 2 * a * s
    ok = disc > 0
    v = np.sqrt(np.where(ok, disc, 0.0))
    t = np.where(ok, (v0 - v) / a, np.inf)
    return t, v


def lob_time(dist, cfg: PhysicsConfig):
    return 1.0 + np.asarray(dist, float) / cfg.lob_speed


def evaluate_passes(origin: np.ndarray, targets: np.ndarray, lofted: np.ndarray, v0: np.ndarray,
                    flight_t: np.ndarray, recv_pos: np.ndarray, recv_vel: np.ndarray,
                    opp_pos: np.ndarray, opp_vel: np.ndarray, cfg: PhysicsConfig) -> dict:
    """Batch-evaluate M passes. Returns dict of (M,) arrays (+ intercept points (M,2))."""
    M = len(targets)
    S = cfg.path_samples
    d = np.linalg.norm(targets - origin[None, :], axis=1)                      # (M,)
    frac = np.linspace(0.0, 1.0, S)[None, :]                                    # (1,S)
    s = d[:, None] * frac                                                       # (M,S)
    tg, vg = ground_times(s, v0[:, None], cfg)
    tf = flight_t[:, None]
    tl = frac * tf
    vl = np.broadcast_to(d[:, None] / np.maximum(tf, 1e-6), (M, S))
    lo = lofted[:, None]
    t = np.where(lo, tl, tg)
    v = np.where(lo, vl, vg)
    mask = np.where(lo, (frac < 0.12) | (frac > 0.85), frac > 0)               # interceptable samples
    u = (targets - origin[None, :]) / np.maximum(d, 1e-6)[:, None]
    pts = origin[None, None, :] + u[:, None, :] * s[:, :, None]                 # (M,S,2)
    t_end = t[:, -1]
    v_end = v[:, -1]
    finite = np.isfinite(t_end)
    t_safe = np.where(np.isfinite(t), t, 1e3)

    N = len(opp_pos)
    if N:
        p_react = opp_pos + opp_vel * cfg.reaction
        dist = np.linalg.norm(pts[:, :, None, :] - p_react[None, None, :, :], axis=3)   # (M,S,N)
        T = cfg.reaction + np.maximum(dist - cfg.control_r, 0.0) / cfg.vmax
        margin = t_safe[:, :, None] - T
        ctrl = np.clip(1.3 - v / 25.0, 0.3, 1.0)[:, :, None]
        p = sigmoid(margin / cfg.sigma_t) * ctrl * mask[:, :, None]
        pj = p.max(axis=1)                                                      # (M,N)
        p_int = 1.0 - np.prod(1.0 - pj, axis=1)
        jbest = np.argmax(pj, axis=1)
        kbest = np.argmax(p[np.arange(M), :, jbest], axis=1)
        ipt = pts[np.arange(M), kbest]
        geo = np.linalg.norm(pts[:, :, None, :] - opp_pos[None, None, :, :], axis=3)
        lane = geo.min(axis=(1, 2))
        t_o = T[:, -1, :].min(axis=1)
    else:
        p_int = np.zeros(M)
        ipt = targets.copy()
        lane = np.full(M, 99.0)
        t_o = np.full(M, 99.0)

    pr = recv_pos + recv_vel * cfg.reaction
    t_r = cfg.reaction + np.maximum(np.linalg.norm(targets - pr, axis=1) - cfg.control_r, 0.0) / cfg.vmax
    t_end_s = np.where(finite, t_end, 1e3)
    recv_margin = t_end_s + 0.25 - t_r
    p_reach = sigmoid(recv_margin / cfg.sigma_t)
    opp_margin = t_o - np.maximum(t_end_s, t_r)
    p_contest = sigmoid((opp_margin + 0.15) / cfg.sigma_t)
    p_receive = p_reach * p_contest * np.where(lofted, cfg.lofted_accuracy, 1.0)
    p_success = np.where(finite, (1.0 - p_int) * p_receive, 0.0)
    return {"p_success": p_success, "p_intercept": np.where(finite, p_int, 1.0), "p_receive": p_receive,
            "t_arrive": t_end, "v_arrive": v_end, "intercept_point": ipt, "lane": lane,
            "recv_margin": recv_margin, "opp_margin": opp_margin}
