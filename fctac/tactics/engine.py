"""Decision layer: state -> candidates -> ranking -> stabilised recommendation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from fctac import types as T
from fctac.tactics import physics as ph
from fctac.tactics.candidates import CandidateGenerator, TacticsConfig
from fctac.tactics.defending import defensive_candidates
from fctac.tactics.stabilizer import Stabilizer, StabilizerConfig


@dataclass
class EngineConfig:
    tactics: TacticsConfig = field(default_factory=TacticsConfig)
    stabilizer: StabilizerConfig = field(default_factory=StabilizerConfig)
    min_calib_conf: float = 0.5
    min_ball_conf: float = 0.35
    min_players: int = 6
    loose_ball_speed: float = 6.0     # free ball faster than this = pass in flight


class HeuristicRanker:
    """Baseline ranker: physics p_success x xT value (Phase 6 baseline)."""
    name = "heuristic"

    def score(self, cands: list, st: T.GameState) -> np.ndarray:
        return np.array([a.score for a in cands], dtype=np.float64)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class TacticsEngine:
    def __init__(self, cfg: EngineConfig = EngineConfig(), ranker=None, predictor=None):
        self.cfg = cfg
        self.gen = CandidateGenerator(cfg.tactics, predictor)
        self.ranker = ranker or HeuristicRanker()
        self.stab = Stabilizer(cfg.stabilizer)

    def reset(self):
        self.stab.reset()

    # ------------------------------------------------------------------------
    def phase(self, st: T.GameState) -> str:
        me = st.controlled
        b = st.ball
        if b is None or me is None:
            return "unknown"
        if b.owner_team == T.TEAM_US or (b.owner_id is None and st.possession == T.TEAM_US):
            on_ball = b.owner_id == me.id or np.linalg.norm(b.pos - me.pos) < self.cfg.tactics.owner_radius
            if on_ball and np.linalg.norm(b.vel - me.vel) < self.cfg.loose_ball_speed:
                return "attack"
            if b.owner_id is None and self.reception(st) is not None:
                return "receiving"
            return "attack_off_ball"
        if b.owner_team == T.TEAM_THEM or st.possession == T.TEAM_THEM:
            return "defence"
        return "loose"

    def reception(self, st: T.GameState):
        """Where/when the controlled player can first take a moving ball, or None."""
        me, b = st.controlled, st.ball
        pc = self.cfg.tactics.physics
        sp = float(np.linalg.norm(b.vel))
        if sp < 1.0:
            return None
        u = b.vel / sp
        stop = sp * sp / (2 * pc.pass_decel)
        s = np.linspace(0.0, min(stop, 60.0), 24)
        tb, _ = ph.ground_times(s, sp, pc)
        pts = b.pos[None, :] + u[None, :] * s[:, None]
        tm = ph.time_to_reach(me.pos, me.vel, pts, pc)[0]
        ok = np.where(tm <= tb + 0.1)[0]
        if not len(ok):
            return None
        k = int(ok[0])
        # an opponent who can cut the ball out before (or at) that point wins the race:
        # then it is a contested ball, not ours to plan from
        opps = st.team(T.TEAM_THEM)
        if opps:
            to = ph.time_to_reach(np.array([o.pos for o in opps]), np.array([o.vel for o in opps]), pts[:k + 1], pc)
            t_ours = max(float(tm[k]), float(tb[k]))
            if np.any(to.min(axis=0) < np.minimum(np.maximum(tb[:k + 1], 0.0), t_ours) - 0.05):
                return None
        return pts[k], float(tb[k])

    def quality(self, st: T.GameState) -> tuple[float, str]:
        if not st.valid:
            return 0.0, st.invalid_reason or "invalid state"
        if st.state_conf < self.cfg.min_calib_conf:
            return 0.0, "pitch state unreliable"
        if st.ball is None or st.ball.confidence < self.cfg.min_ball_conf:
            return 0.0, "ball not found"
        if st.controlled is None:
            return 0.0, "controlled player unknown"
        if len(st.players) < self.cfg.min_players:
            return 0.0, "too few players tracked"
        q = min(st.state_conf, st.ball.confidence, st.controlled.confidence)
        return float(q), ""

    def decide(self, st: Optional[T.GameState]) -> tuple[list, T.Recommendation]:
        if st is None:
            return [], T.Recommendation(status=T.STATUS_ANALYSING, reason="no state")
        q, why = self.quality(st)
        if q <= 0:
            self.stab.reset_soft(None)
            return [], T.Recommendation(status=T.STATUS_ANALYSING, reason=why)
        phase = self.phase(st)
        if phase == "attack":
            cands = self.gen.attacking(st)
        elif phase == "receiving":
            rp, tr = self.reception(st)
            cands = self.gen.attacking(st, origin=rp, t_offset=tr)
            phase = "attack"            # same decision context as being on the ball
        elif phase == "defence":
            cands = defensive_candidates(st, self.cfg.tactics.physics)
        else:
            self.stab.reset_soft((phase, st.controlled_id))
            return [], T.Recommendation(status=T.STATUS_HIDDEN, reason=phase)
        if not cands:
            return [], T.Recommendation(status=T.STATUS_HIDDEN, reason="no candidates")
        scores = self.ranker.score(cands, st) if phase == "attack" else np.array([a.score for a in cands])
        order = np.argsort(-scores)
        ranked = [cands[i] for i in order]
        for i, a in zip(order, ranked):
            a.score = float(scores[i])
        conf = self._confidence(ranked, phase) * q
        rec = self.stab.update(st.t, ranked, conf, (phase, st.controlled_id))
        return ranked, rec

    def _confidence(self, ranked, ph) -> float:
        s1 = ranked[0].score
        s2 = ranked[1].score if len(ranked) > 1 else s1 - 1.0
        if ph == "attack":
            gap = _sigmoid((s1 - s2 - 0.004) / 0.006)
            ps = ranked[0].p_success
            return float(np.clip(0.35 + 0.35 * gap + 0.3 * ps, 0.0, 1.0))
        gap = _sigmoid((s1 - s2 - 0.05) / 0.08)
        return float(np.clip(0.35 + 0.35 * gap + 0.3 * s1, 0.0, 1.0))
