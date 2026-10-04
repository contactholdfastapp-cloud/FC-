"""Shot planning: where to aim, how hard, which shot type, and its xG."""
from __future__ import annotations

import numpy as np

from fctac import types as T
from fctac.tactics.value import GOAL_W, xg

L, W = T.PITCH_LENGTH, T.PITCH_WIDTH


def _goalkeeper(st: T.GameState):
    best, bd = None, 1e9
    for p in st.team(T.TEAM_THEM):
        d = np.hypot(p.pos[0] - L, p.pos[1] - W / 2)
        if p.role == "GK":
            return p
        if d < bd and d < 14:
            best, bd = p, d
    return best


def _blockers(origin, aim, opp_pos, gk_id_pos):
    """Defenders inside the shooting cone towards ``aim`` (excluding the GK)."""
    n = 0
    v = aim - origin
    dist = np.linalg.norm(v)
    u = v / max(dist, 1e-6)
    for o in opp_pos:
        if gk_id_pos is not None and np.allclose(o, gk_id_pos):
            continue
        rel = o - origin
        along = float(np.dot(rel, u))
        if along <= 0.5 or along >= dist:
            continue
        lateral = abs(rel[0] * u[1] - rel[1] * u[0])
        if lateral < 0.6 + 0.04 * along:
            n += 1
    return n


def plan_shot(origin: np.ndarray, st: T.GameState) -> dict:
    gk = _goalkeeper(st)
    opp_pos = np.array([p.pos for p in st.team(T.TEAM_THEM)]).reshape(-1, 2)
    base = float(xg(origin)[0])
    side = 1.0 if origin[1] >= W / 2 else -1.0       # which post is "near"
    inset = 0.65
    aims = {
        "NEAR POST": np.array([L, W / 2 + side * (GOAL_W / 2 - inset)]),
        "FAR POST": np.array([L, W / 2 - side * (GOAL_W / 2 - inset)]),
    }
    best = None
    d_goal = float(np.hypot(L - origin[0], W / 2 - origin[1]))
    for name, aim in aims.items():
        f = 1.0
        if gk is not None:
            # lateral distance between aim point and GK, measured across the shot line
            u = (aim - origin) / max(np.linalg.norm(aim - origin), 1e-6)
            rel = gk.pos - origin
            along = float(np.dot(rel, u))
            lateral = abs(rel[0] * u[1] - rel[1] * u[0])
            reach = 1.4 + 0.06 * max(along, 0.0)       # GK covers more the further he is from the shooter
            f = 1.0 / (1.0 + np.exp(-(lateral - reach) / 0.6))
            f = 0.25 + 0.95 * f
        nb = _blockers(origin, aim, opp_pos, None if gk is None else gk.pos)
        f *= 0.6 ** nb
        if name == "FAR POST":
            f *= 1.05        # far post: rebounds/off-target outcomes are less costly
        p = float(np.clip(base * f * 1.6, 0.0, 0.95))
        if best is None or p > best["p_goal"]:
            best = {"p_goal": p, "target": aim, "placement": name, "blockers": nb}
    angle = abs(np.arctan2(origin[1] - W / 2, L - origin[0]))
    if d_goal < 12:
        power, stype = "1.5 BAR", "LOW" if best["placement"] == "NEAR POST" else "PLACED"
    elif d_goal < 22:
        power = "2 BAR"
        stype = "FINESSE" if best["placement"] == "FAR POST" and angle > 0.25 else "NORMAL"
    else:
        power, stype = "2.5 BAR", "POWER"
    best["power"] = power
    best["shot_type"] = stype
    return best
