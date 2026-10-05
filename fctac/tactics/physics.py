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
    accel: float = 5.5           # m/s^2: nobody goes from standing to sprint instantly
    reaction: float = 0.3        # s before a player changes course
    sigma_t: float = 0.30        # s, uncertainty of time comparisons
    control_r: float = 1.0       # m, reach radius
    pass_decel: float = 2.8      # m/s^2 rolling deceleration
    arrive_pass: float = 6.0     # desired arrival speed, ground pass
    arrive_through: float = 4.0
    v_min: float = 8.0
    v_max: float = 26.0
    lob_speed: float = 25.0      # flight time = 1 + d / lob_speed
    lofted_accuracy: float = 0.85
    pass_error_per_m: float = 0.002  # execution error (FC 27: less pass assistance), per metre
    path_samples: int = 14


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def run_time(d, v0, cfg: PhysicsConfig):
    """Time to cover d metres starting at speed v0 (towards the target), accelerating to vmax."""
    d = np.asarray(d, float)
    v0 = np.clip(np.asarray(v0, float), 0.0, cfg.vmax)
    a, vm = cfg.accel, cfg.vmax
    tv = (vm - v0) / a
    dv = v0 * tv + 0.5 * a * tv * tv
    t_acc = (-v0 + np.sqrt(v0 * v0 + 2 * a * d)) / a
    return np.where(d <= dv, t_acc, tv + (d - dv) / vm)


def time_to_reach(pos: np.ndarray, vel: np.ndarray, targets: np.ndarray, cfg: PhysicsConfig) -> np.ndarray:
    """(N,2) players, (M,2) targets -> (N,M) seconds (reaction, then accelerate; momentum counts)."""
    pos = np.atleast_2d(pos)
    vel = np.atleast_2d(vel)
    targets = np.atleast_2d(targets)
    p_react = pos + vel * cfg.reaction
    diff = targets[None, :, :] - p_react[:, None, :]
    dist = np.linalg.norm(diff, axis=2)
    u = diff / np.maximum(dist, 1e-6)[:, :, None]
    v0 = np.sum(u * vel[:, None, :], axis=2)                      # speed already heading there
    d = np.maximum(dist - cfg.control_r, 0.0)
    return cfg.reaction + run_time(d, v0, cfg)


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

    # the receiver claims the ball at the first path point he reaches no later than the ball;
    # beyond that point nobody can intercept (pitch-control "first to the ball" logic)
    pr = recv_pos + recv_vel * cfg.reaction
    rdiff = pts - pr[:, None, :]
    rdist = np.linalg.norm(rdiff, axis=2)
    ru = rdiff / np.maximum(rdist, 1e-6)[:, :, None]
    rv0 = np.sum(ru * recv_vel[:, None, :], axis=2)
    t_rk = cfg.reaction + run_time(np.maximum(rdist - cfg.control_r, 0.0), rv0, cfg)       # (M,S)
    claim = (t_rk <= t_safe + 0.1) & (frac > 0.3)
    first = np.where(claim.any(axis=1), np.argmax(claim, axis=1), S - 1)
    before_claim = np.arange(S)[None, :] <= first[:, None]
    N = len(opp_pos)
    if N:
        p_react = opp_pos + opp_vel * cfg.reaction
        diff = pts[:, :, None, :] - p_react[None, None, :, :]
        dist = np.linalg.norm(diff, axis=3)                                       # (M,S,N)
        ou = diff / np.maximum(dist, 1e-6)[..., None]
        ov0 = np.sum(ou * opp_vel[None, None, :, :], axis=3)
        T = cfg.reaction + run_time(np.maximum(dist - cfg.control_r, 0.0), ov0, cfg)
        margin = t_safe[:, :, None] - T
        ctrl = np.clip(1.3 - v / 25.0, 0.3, 1.0)[:, :, None]
        valid = (mask & before_claim & (frac < 0.97))[:, :, None]
        p = sigmoid(margin / cfg.sigma_t) * ctrl * valid
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

    t_r = t_rk[:, -1]
    t_end_s = np.where(finite, t_end, 1e3)
    recv_margin = t_end_s + 0.25 - t_r
    p_reach = sigmoid(recv_margin / cfg.sigma_t)
    # contest at the receiving point: if both are there before the ball it is a duel won by
    # whoever got there first (body position); otherwise first to the ball after it arrives
    opp_margin = t_o - np.maximum(t_end_s, t_r)
    p_race = sigmoid((opp_margin + 0.15) / cfg.sigma_t)
    both_early = sigmoid((t_end_s - np.maximum(t_r, t_o)) / 0.3)
    p_duel = 0.25 + 0.5 * sigmoid((t_o - t_r) / 0.5)
    p_contest = both_early * p_duel + (1.0 - both_early) * p_race
    acc = np.where(lofted, np.clip(cfg.lofted_accuracy + 0.05 - 0.002 * d, 0.6, 0.95),
                   np.clip(1.0 - cfg.pass_error_per_m * d, 0.7, 1.0))
    p_receive = p_reach * p_contest * acc
    p_success = np.where(finite, (1.0 - p_int) * p_receive, 0.0)
    return {"p_success": p_success, "p_intercept": np.where(finite, p_int, 1.0), "p_receive": p_receive,
            "t_arrive": t_end, "v_arrive": v_end, "intercept_point": ipt, "lane": lane,
            "recv_margin": recv_margin, "opp_margin": opp_margin}
