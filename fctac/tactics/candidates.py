"""Candidate action generation + baseline (physics/heuristic) evaluation.

Every candidate gets
  * ``p_success``  from the physics race model,
  * ``value_success`` (xT-like value if it works),
  * ``score`` = expected value in goal units,
  * ``features``  fixed-length vector for the learned ranker.

The learned model re-scores the same candidates; generation is shared so
heuristic and learned rankers are compared like-for-like.  All passes of a
frame are evaluated in one vectorised batch (see physics.evaluate_passes).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from fctac import types as T
from fctac.prediction.kinematic import KinematicPredictor
from fctac.tactics import physics as ph
from fctac.tactics.value import turnover_cost, xg, zone_value

L, W = T.PITCH_LENGTH, T.PITCH_WIDTH
GOAL = np.array([L, W / 2])

KIND_INDEX = {k: i for i, k in enumerate(T.ATTACK_KINDS)}
FEATURE_NAMES = (
    [f"kind_{k}" for k in T.ATTACK_KINDS]
    + ["dist", "dir_goal_cos", "lane_clear", "p_intercept", "p_receive", "recv_margin", "opp_margin",
       "t_arrive", "v_arrive", "target_x", "target_dy", "progress", "v_target", "v_current",
       "xg_target", "space_target", "congestion_target", "defenders_ahead", "offside_risk",
       "recv_speed", "recv_dir_goal_cos", "carrier_pressure", "carrier_x", "carrier_dy",
       "p_success_phys", "ev_phys"]
)
N_FEATURES = len(FEATURE_NAMES)
NK = len(T.ATTACK_KINDS)


@dataclass
class TacticsConfig:
    physics: ph.PhysicsConfig = field(default_factory=ph.PhysicsConfig)
    max_pass_dist: float = 50.0
    min_pass_dist: float = 3.0
    lob_min_dist: float = 18.0
    shoot_max_dist: float = 32.0
    through_lead_times: tuple = (0.9, 1.3, 1.7, 2.2, 2.8)
    dribble_dirs: int = 9
    dribble_dist: float = 5.0
    owner_radius: float = 2.2     # controlled player counts as on the ball within this
    hold_tempo_cost: float = 0.003   # holding lets the defence re-organise (goal units)


def _unit(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.where(n > 1e-9, v / np.maximum(n, 1e-9), 0.0)


class CandidateGenerator:
    def __init__(self, cfg: TacticsConfig = TacticsConfig(), predictor=None):
        self.cfg = cfg
        self.pred = predictor or KinematicPredictor()

    # ------------------------------------------------------------------------
    def attacking(self, st: T.GameState, origin: np.ndarray | None = None, t_offset: float = 0.0) -> list:
        """Candidates for the controlled player on the ball.

        ``origin``/``t_offset`` evaluate from a future reception point (ball
        travelling to the controlled player): everyone is first moved forward
        by ``t_offset`` seconds with the movement predictor.
        """
        me = st.controlled
        if me is None or st.ball is None:
            return []
        cfg, pc = self.cfg, self.cfg.physics
        opps = st.team(T.TEAM_THEM)
        mates = [p for p in st.team(T.TEAM_US) if p.id != me.id]
        opp_pos = np.array([o.pos for o in opps]).reshape(-1, 2)
        opp_vel = np.array([o.vel for o in opps]).reshape(-1, 2)
        R_pos = np.array([p.pos for p in mates]).reshape(-1, 2)
        R_vel = np.array([p.vel for p in mates]).reshape(-1, 2)
        if t_offset > 0:
            opp_pos = self.pred.predict_point(opp_pos, opp_vel, t_offset) if len(opp_pos) else opp_pos
            R_pos = self.pred.predict_point(R_pos, R_vel, t_offset) if len(R_pos) else R_pos
        if origin is None:
            origin = st.ball.pos.copy() if np.linalg.norm(st.ball.pos - me.pos) < 3 else me.pos.copy()
        origin = np.asarray(origin, float)
        ctx = self._context(origin, opp_pos)

        specs_kind, specs_r, tg, lofted, v0s, fts = [], [], [], [], [], []

        def add(kind, ri, tgt, lo, v0, ft):
            specs_kind.append(kind)
            specs_r.append(ri)
            tg.append(tgt)
            lofted.append(lo)
            v0s.append(v0)
            fts.append(ft)

        if len(mates):
            d0 = np.linalg.norm(R_pos - origin, axis=1)
            ok = (d0 >= cfg.min_pass_dist) & (d0 <= cfg.max_pass_dist)
            # PASS to feet: aim at where the receiver will be when the ball arrives
            tgt = R_pos.copy()
            for _ in range(2):
                dd = np.linalg.norm(tgt - origin, axis=1)
                v0 = ph.ground_v0(dd, pc.arrive_pass, pc)
                tt, _ = ph.ground_times(dd, v0, pc)
                tt = np.where(np.isfinite(tt), tt, 1.5)
                tgt = self.pred.predict_point(R_pos, R_vel, tt[:, None])
            dd = np.linalg.norm(tgt - origin, axis=1)
            v0 = ph.ground_v0(dd, pc.arrive_pass, pc)
            for i in np.where(ok)[0]:
                add(T.PASS, i, tgt[i], False, v0[i], 0.0)
            # LOB / CROSS over the lines
            tf = ph.lob_time(d0, pc)
            tl = self.pred.predict_point(R_pos, R_vel, tf[:, None])
            for i in np.where(ok & (d0 > cfg.lob_min_dist))[0]:
                kind = T.CROSS if self._is_cross(origin, tl[i]) else T.LOB
                add(kind, i, tl[i], True, 0.0, tf[i])
            # THROUGH balls into the receiver's run: target = predicted receiving point
            speed = np.linalg.norm(R_vel, axis=1)
            goal_dir = _unit(GOAL[None, :] - R_pos)
            run = np.where((speed > 2.5)[:, None], _unit(R_vel), goal_dir)
            u = _unit(0.65 * run + 0.35 * goal_dir)
            vrun = np.maximum(speed, 6.5)
            for i in np.where(ok & (u[:, 0] > 0.2))[0]:
                for tau in cfg.through_lead_times:
                    q = np.clip(R_pos[i] + u[i] * vrun[i] * tau * 0.92, [1, 1], [L - 1, W - 1])
                    d = float(np.linalg.norm(q - origin))
                    if d < 6 or d > cfg.max_pass_dist or q[0] < R_pos[i, 0] + 2:
                        continue
                    v0t = (d + 0.5 * pc.pass_decel * tau * tau) / tau
                    if v0t > pc.v_max or v0t - pc.pass_decel * tau < 1.5:
                        continue
                    add(T.THROUGH, i, q, False, v0t, 0.0)

        out: list[T.Action] = []
        if specs_kind:
            out.extend(self._passes(origin, ctx, mates, R_pos, R_vel, opp_pos, opp_vel,
                                    specs_kind, np.array(specs_r), np.array(tg), np.array(lofted),
                                    np.array(v0s, float), np.array(fts, float)))
        if np.linalg.norm(GOAL - origin) < cfg.shoot_max_dist and origin[0] > L / 2:
            out.append(self._shot(origin, st, ctx))
        out.append(self._dribble(origin, opp_pos, opp_vel, ctx))
        out.append(self._hold(origin, ctx))
        return out

    # ------------------------------------------------------------------------
    def _context(self, origin, opp_pos) -> dict:
        if len(opp_pos):
            dd = np.linalg.norm(opp_pos - origin, axis=1)
            pressure = float(np.sum(np.exp(-dd / 3.0)))
            nearest = float(dd.min())
            xs = np.sort(opp_pos[:, 0])[::-1]
            line = float(xs[1]) if len(xs) > 1 else L
        else:
            pressure, nearest, line = 0.0, 99.0, L
        line = max(line, origin[0], L / 2)       # offside line (attack-aligned x)
        return {"pressure": pressure, "nearest": nearest, "line": line,
                "v_current": float(zone_value(origin)[0]), "origin": origin}

    def _is_cross(self, origin, tgt) -> bool:
        return origin[0] > L - 32 and abs(origin[1] - W / 2) > 15 and tgt[0] > L - 18 and abs(tgt[1] - W / 2) < 16

    def _passes(self, origin, ctx, mates, R_pos, R_vel, opp_pos, opp_vel, kinds, ridx, tg, lofted, v0, ft):
        pc = self.cfg.physics
        ev = ph.evaluate_passes(origin, tg, lofted, v0, ft, R_pos[ridx], R_vel[ridx], opp_pos, opp_vel, pc)
        M = len(kinds)
        offside = ph.sigmoid((R_pos[ridx, 0] - ctx["line"] - 0.3) / 0.5)
        p = ev["p_success"] * (1.0 - offside)
        if len(opp_pos):
            dt = np.linalg.norm(opp_pos[None, :, :] - tg[:, None, :], axis=2)
            space = dt.min(axis=1)
            congestion = (dt < 5.0).sum(axis=1).astype(float)
            ahead = (opp_pos[None, :, 0] > tg[:, None, 0]).sum(axis=1).astype(float)
        else:
            space, congestion, ahead = np.full(M, 30.0), np.zeros(M), np.zeros(M)
        v_t = zone_value(tg)
        xg_t = xg(tg)
        v_succ = v_t * (1.0 + 0.25 * np.clip((space - 3.0) / 5.0, -1.0, 1.0))
        is_cross = np.array([k == T.CROSS for k in kinds])
        v_succ = np.where(is_cross, np.maximum(v_succ, 0.6 * xg_t), v_succ)
        cost_pt = np.where((ev["p_intercept"] > 0.3)[:, None], ev["intercept_point"], tg)
        cost = turnover_cost(cost_pt)
        score = p * v_succ - (1.0 - p) * cost
        dist = np.linalg.norm(tg - origin, axis=1)
        goal_cos = np.sum(_unit(tg - origin) * _unit(GOAL - origin)[None, :], axis=1)
        rs = np.linalg.norm(R_vel[ridx], axis=1)
        rcos = np.where(rs > 0.5, np.sum(_unit(R_vel[ridx]) * _unit(GOAL[None, :] - R_pos[ridx]), axis=1), 0.0)
        F = np.zeros((M, N_FEATURES), np.float32)
        F[np.arange(M), [KIND_INDEX[k] for k in kinds]] = 1.0
        F[:, NK:] = np.stack([
            dist / 50.0, goal_cos, np.minimum(ev["lane"], 15.0) / 15.0, ev["p_intercept"], ev["p_receive"],
            np.clip(ev["recv_margin"], -3, 3) / 3.0, np.clip(ev["opp_margin"], -3, 3) / 3.0,
            np.minimum(np.where(np.isfinite(ev["t_arrive"]), ev["t_arrive"], 5.0), 5.0) / 5.0,
            ev["v_arrive"] / 25.0, tg[:, 0] / L, np.abs(tg[:, 1] - W / 2) / (W / 2), (tg[:, 0] - origin[0]) / 50.0,
            v_t, np.full(M, ctx["v_current"]), xg_t, np.minimum(space, 20.0) / 20.0, congestion / 5.0,
            ahead / 11.0, offside, rs / 9.0, rcos, np.full(M, min(ctx["pressure"], 3.0) / 3.0),
            np.full(M, origin[0] / L), np.full(M, abs(origin[1] - W / 2) / (W / 2)), p, score * 10,
        ], 1)
        best_through: dict[int, int] = {}
        keep = []
        for m in range(M):
            if kinds[m] == T.THROUGH:
                r = int(ridx[m])
                if r not in best_through or score[m] > score[best_through[r]]:
                    best_through[r] = m
            else:
                keep.append(m)
        keep += list(best_through.values())
        out = []
        for m in keep:
            r = mates[int(ridx[m])]
            out.append(T.Action(
                kind=kinds[m], target_id=r.id, target_label=r.label, target_point=tg[m].copy(),
                score=float(score[m]), p_success=float(p[m]), value_success=float(v_succ[m]),
                detail={"p_intercept": float(ev["p_intercept"][m]), "t_arrive": float(ev["t_arrive"][m]),
                        "offside": float(offside[m]), "space": float(space[m]), "lane": float(ev["lane"][m])},
                features=F[m]))
        return out

    def _generic_features(self, kind, tgt, origin, ctx, p, score, space, congestion, ahead) -> np.ndarray:
        f = np.zeros(N_FEATURES, np.float32)
        f[KIND_INDEX[kind]] = 1.0
        v_t = float(zone_value(tgt)[0])
        dv = tgt - origin
        goal_cos = float(np.dot(_unit(dv), _unit(GOAL - origin))) if np.linalg.norm(dv) > 0.1 else 0.0
        f[NK:] = [
            np.linalg.norm(dv) / 50.0, goal_cos, 1.0, 1.0 - p, p, 0.0, 0.0, 0.0, 0.0,
            tgt[0] / L, abs(tgt[1] - W / 2) / (W / 2), dv[0] / 50.0, v_t, ctx["v_current"],
            float(xg(tgt)[0]), min(space, 20.0) / 20.0, congestion / 5.0, ahead / 11.0, 0.0, 0.0, 0.0,
            min(ctx["pressure"], 3.0) / 3.0, origin[0] / L, abs(origin[1] - W / 2) / (W / 2), p, score * 10,
        ]
        return f

    def _shot(self, origin, st, ctx) -> T.Action:
        from fctac.tactics.shooting import plan_shot
        plan = plan_shot(origin, st)
        p_goal = plan["p_goal"]
        score = p_goal + (1 - p_goal) * 0.015 - (1 - p_goal) * float(turnover_cost(np.array([L - 6, W / 2]))[0])
        f = self._generic_features(T.SHOOT, plan["target"], origin, ctx, p_goal, score, 0.0, plan["blockers"], 0.0)
        return T.Action(kind=T.SHOOT, target_point=plan["target"], score=float(score), p_success=float(p_goal),
                        value_success=1.0, detail={k: plan[k] for k in ("placement", "power", "shot_type", "blockers")},
                        features=f)

    def _dribble(self, origin, opp_pos, opp_vel, ctx) -> T.Action:
        pc = self.cfg.physics
        n = self.cfg.dribble_dirs
        a = np.linspace(-np.pi / 2, np.pi / 2, n)          # forward half-plane
        q = np.clip(origin[None, :] + self.cfg.dribble_dist * np.stack([np.cos(a), np.sin(a)], 1), [1, 1], [L - 1, W - 1])
        t_me = self.cfg.dribble_dist / 6.0
        if len(opp_pos):
            t_o = ph.time_to_reach(opp_pos, opp_vel, q, pc).min(axis=0)
            margin = t_o - t_me
            dq = np.linalg.norm(opp_pos[None, :, :] - q[:, None, :], axis=2)
            space = dq.min(axis=1)
            congestion = (dq < 5).sum(axis=1)
            ahead = (opp_pos[None, :, 0] > q[:, None, 0]).sum(axis=1)
        else:
            margin, space = np.full(n, 3.0), np.full(n, 30.0)
            congestion, ahead = np.zeros(n), np.zeros(n)
        p = ph.sigmoid((margin + 0.2) / 0.35) * 0.95
        score = p * zone_value(q) - (1 - p) * turnover_cost(q)
        k = int(np.argmax(score))
        f = self._generic_features(T.DRIBBLE, q[k], origin, ctx, float(p[k]), float(score[k]),
                                   float(space[k]), float(congestion[k]), float(ahead[k]))
        return T.Action(kind=T.DRIBBLE, target_point=q[k].copy(), score=float(score[k]), p_success=float(p[k]),
                        value_success=float(zone_value(q[k])[0]), detail={"space": float(space[k])}, features=f)

    def _hold(self, origin, ctx) -> T.Action:
        p = float(np.clip(1.0 - 0.28 * ctx["pressure"], 0.15, 0.97))
        v = ctx["v_current"]
        score = p * v - (1 - p) * float(turnover_cost(origin)[0]) - self.cfg.hold_tempo_cost
        f = self._generic_features(T.HOLD, origin, origin, ctx, p, score, ctx["nearest"], 0.0, 0.0)
        return T.Action(kind=T.HOLD, target_point=origin.copy(), score=float(score), p_success=p,
                        value_success=v, features=f)
