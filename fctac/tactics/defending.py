"""Defensive recommendations: SWITCH, PRESS, JOCKEY, COVER.

Scores are priorities in [0, 1] (not goal units); defensive and attacking
candidates are never ranked against each other.
"""
from __future__ import annotations

import numpy as np

from fctac import types as T
from fctac.tactics import physics as ph
from fctac.tactics.value import zone_value

L, W = T.PITCH_LENGTH, T.PITCH_WIDTH
OWN_GOAL = np.array([0.0, W / 2])


def _goal_side(p: np.ndarray, carrier: np.ndarray) -> float:
    """1 when p sits between carrier and our goal, 0 when behind the play."""
    to_goal = OWN_GOAL - carrier
    n = np.linalg.norm(to_goal)
    if n < 1e-6:
        return 1.0
    along = float(np.dot(p - carrier, to_goal / n))
    return float(ph.sigmoid(along / 1.5))


def defensive_candidates(st: T.GameState, pc: ph.PhysicsConfig = ph.PhysicsConfig()) -> list[T.Action]:
    me = st.controlled
    if me is None or st.ball is None:
        return []
    opps = st.team(T.TEAM_THEM)
    ours = [p for p in st.team(T.TEAM_US) if p.role != "GK"]
    if not opps or not ours:
        return []
    carrier = st.player(st.ball.owner_id) if st.ball.owner_team == T.TEAM_THEM else None
    if carrier is None:
        carrier = min(opps, key=lambda o: np.linalg.norm(o.pos - st.ball.pos))
    c_future = carrier.pos + carrier.vel * 0.5
    out: list[T.Action] = []

    # who should engage the carrier?
    pos = np.array([p.pos for p in ours])
    vel = np.array([p.vel for p in ours])
    t_reach = ph.time_to_reach(pos, vel, c_future[None, :], pc)[:, 0]
    gs = np.array([_goal_side(p.pos, carrier.pos) for p in ours])
    cost = t_reach + 0.7 * (1.0 - gs)
    i_best = int(np.argmin(cost))
    i_me = next((i for i, p in enumerate(ours) if p.id == me.id), None)
    danger = float(zone_value(np.array([L - carrier.pos[0], W - carrier.pos[1]]))[0])
    urgency = float(np.clip(0.4 + 4.0 * danger, 0.4, 1.0))

    if i_me is None or (i_best != i_me and cost[i_me] - cost[i_best] > 0.6):
        b = ours[i_best]
        gap = 1.0 if i_me is None else float(cost[i_me] - cost[i_best])
        out.append(T.Action(kind=T.SWITCH, target_id=b.id, target_label=b.label,
                            score=float(np.clip(0.55 + 0.15 * gap, 0.55, 0.95)) * urgency,
                            p_success=0.0, detail={"gap_s": gap}))
    if i_me is not None:
        d = float(np.linalg.norm(me.pos - carrier.pos))
        support = sum(1 for p in ours if p.id != me.id and np.linalg.norm(p.pos - carrier.pos) < 10)
        mine_gs = gs[i_me]
        deep = carrier.pos[0] < 35
        if d < 9 and (deep or mine_gs < 0.5 or support == 0):
            out.append(T.Action(kind=T.JOCKEY, target_id=carrier.id, target_label=carrier.label,
                                score=0.7 * urgency, detail={"dist": d, "support": support}))
        else:
            s = 0.75 if support >= 1 else 0.6
            out.append(T.Action(kind=T.PRESS, target_id=carrier.id, target_label=carrier.label,
                                score=s * urgency * (1.0 if d < 15 else 0.8), detail={"dist": d, "support": support}))

        # COVER the most dangerous free runner close to me
        best = None
        for o in opps:
            if o.id == carrier.id or o.role == "GK":
                continue
            threat = float(zone_value(np.array([L - o.pos[0], W - o.pos[1]]))[0])
            near_def = float(np.min(np.linalg.norm(pos - o.pos, axis=1)))
            free = float(np.clip(near_def / 6.0, 0.0, 1.5))
            runs = float(np.clip(-o.vel[0] / 5.0, 0.0, 1.0))      # running at our goal
            dm = float(np.linalg.norm(me.pos - o.pos))
            s = threat * (0.6 + free) * (1.0 + runs) * np.exp(-dm / 15.0)
            if best is None or s > best[0]:
                best = (s, o)
        if best is not None and best[0] > 0.02:
            o = best[1]
            cover_pt = o.pos + 2.0 * (OWN_GOAL - o.pos) / max(np.linalg.norm(OWN_GOAL - o.pos), 1e-6)
            out.append(T.Action(kind=T.COVER, target_id=o.id, target_label=o.label, target_point=cover_pt,
                                score=float(np.clip(0.3 + 6.0 * best[0], 0.3, 0.85)) * (0.8 if d < 8 else 1.0),
                                detail={"threat": best[0]}))
    return out
