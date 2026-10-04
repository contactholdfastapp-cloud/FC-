"""Decision-window dataset: auto-label what the human did and what happened.

Streaming over GameStates (from the vision pipeline on a recording, or the
oracle on synthetic ground truth):

* release   - the controlled player loses the ball at speed -> PASS / THROUGH /
              LOB / CROSS / SHOOT, classified from the ball trajectory and the
              receiver's movement;
* resolution- next owner (team-mate = complete, opponent = intercepted), ball
              out of play, or a shot reaching the goal line;
* carry     - while the controlled player keeps the ball, every window becomes
              a DRIBBLE (moved >= 4 m) or HOLD example; success = still ours.

Each example stores the state *before* the decision (``decision_lag_s``), all
candidate actions with their features (same generator as the runtime), the
index of the human's choice, and outcome/progression measures.  Human
choices are NOT assumed optimal: outcomes are stored separately so models
can be trained/evaluated on value, not just imitation.
"""
from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from fctac import types as T
from fctac.tactics.candidates import CandidateGenerator
from fctac.tactics.value import zone_value

L, W = T.PITCH_LENGTH, T.PITCH_WIDTH


@dataclass
class LabelerConfig:
    decision_lag_s: float = 0.15
    resolve_timeout_s: float = 4.0
    min_release_speed: float = 5.0
    carry_window_s: float = 1.0
    dribble_min_dist: float = 4.0
    future_s: float = 3.0
    history_s: float = 5.0


def compact_state(st: T.GameState) -> dict:
    return {
        "frame": st.frame, "t": round(st.t, 4), "attack_sign": st.attack_sign, "controlled": st.controlled_id,
        "possession": st.possession,
        "players": [[p.id, p.team, round(float(p.pos[0]), 2), round(float(p.pos[1]), 2),
                     round(float(p.vel[0]), 2), round(float(p.vel[1]), 2), p.role] for p in st.players],
        "ball": None if st.ball is None else [round(float(st.ball.pos[0]), 2), round(float(st.ball.pos[1]), 2),
                                              round(float(st.ball.vel[0]), 2), round(float(st.ball.vel[1]), 2),
                                              st.ball.owner_id],
    }


def state_from_compact(d: dict) -> T.GameState:
    gs = T.GameState(frame=d["frame"], t=d["t"], players=[], ball=None, attack_sign=d["attack_sign"],
                     controlled_id=d["controlled"], possession=d["possession"], calib_conf=1.0)
    for pid, team, x, y, vx, vy, role in d["players"]:
        gs.players.append(T.PlayerState(id=pid, team=team, pos=np.array([x, y]), vel=np.array([vx, vy]),
                                        acc=np.zeros(2), controlled=pid == d["controlled"], role=role))
    if d["ball"] is not None:
        bx, by, bvx, bvy, owner = d["ball"]
        own_team = T.TEAM_UNKNOWN
        if owner is not None:
            p = gs.player(owner)
            own_team = p.team if p is not None else T.TEAM_UNKNOWN
        gs.ball = T.BallState(pos=np.array([bx, by]), vel=np.array([bvx, bvy]), confidence=1.0,
                              owner_id=owner, owner_team=own_team)
    return gs


@dataclass
class _Pending:
    kind: str
    frame: int
    t: float
    passer: int
    decision_state: T.GameState
    release_state: T.GameState
    ball_path: list = field(default_factory=list)


class DecisionLabeler:
    def __init__(self, clip: str = "", cfg: LabelerConfig = LabelerConfig(), generator: Optional[CandidateGenerator] = None):
        self.clip = clip
        self.cfg = cfg
        self.gen = generator or CandidateGenerator()
        self.hist: deque = deque()
        self.prev: Optional[T.GameState] = None
        self.pending: Optional[_Pending] = None
        self.lost = None
        self.carry_start: Optional[T.GameState] = None
        self.waiting_future: list = []          # examples waiting for +future_s state
        self.out: list = []

    # ------------------------------------------------------------------------
    def _state_at(self, t: float) -> Optional[T.GameState]:
        best = None
        for s in self.hist:
            if s.t <= t:
                best = s
            else:
                break
        return best

    def push(self, st: T.GameState) -> list:
        """Feed one state; returns finished examples."""
        if st is None or not st.valid or st.ball is None:
            self.prev = st
            return []
        self.hist.append(st)
        while self.hist and st.t - self.hist[0].t > self.cfg.history_s:
            self.hist.popleft()
        prev = self.prev
        self.prev = st
        b = st.ball
        me = st.controlled_id
        if self.pending is not None:
            self.pending.ball_path.append((st.t, b.pos.copy(), b.vel.copy()))
            self._try_resolve(st)
        elif prev is not None and prev.ball is not None:
            # in FC the human controls our ball carrier, and control jumps to the
            # receiver as the pass is played -> any team-mate carrier counts
            had = prev.ball.owner_id is not None and prev.ball.owner_team == T.TEAM_US
            if had and b.owner_id != prev.ball.owner_id:
                # ownership ended; decide what it was once the ball's motion is clear
                self.lost = (prev, st.t)
            if self.lost is not None:
                lprev, lt = self.lost
                speed = float(np.linalg.norm(b.vel))
                owner = st.player(b.owner_id) if b.owner_id is not None else None
                if owner is not None and owner.id == lprev.ball.owner_id:
                    self.lost = None                       # flicker: still dribbling
                elif owner is not None and owner.team == T.TEAM_THEM and speed < self.cfg.min_release_speed:
                    self.lost = None
                    self._finish_carry(st, lost=True)      # tackled
                elif speed > self.cfg.min_release_speed or owner is not None:
                    self.lost = None
                    self._finish_carry(st, lost=False, end=False)
                    dstate = self._state_at(lprev.t - self.cfg.decision_lag_s) or lprev
                    self.pending = _Pending("?", st.frame, lt, lprev.ball.owner_id, dstate, lprev,
                                            [(st.t, b.pos.copy(), b.vel.copy())])
                    if owner is not None:
                        self._try_resolve(st)
                elif st.t - lt > 0.4:
                    self.lost = None                       # ball just stopped near him: ignore
        # carries (dribble / hold windows)
        if self.pending is None and self.lost is None and b.owner_id is not None and b.owner_team == T.TEAM_US:
            if self.carry_start is None or self.carry_start.ball.owner_id != b.owner_id:
                self.carry_start = st
            elif st.t - self.carry_start.t >= self.cfg.carry_window_s:
                self._finish_carry(st, lost=False)
                self.carry_start = st
        elif self.pending is None and self.carry_start is not None and b.owner_id != self.carry_start.ball.owner_id:
            self.carry_start = None
        self._fill_future(st)
        done, self.out = self.out, []
        return done

    # ------------------------------------------------------------------------
    def _finish_carry(self, st: T.GameState, lost: bool, end: bool = True):
        s0 = self.carry_start
        self.carry_start = None
        if s0 is None or st.t - s0.t < 0.5 * self.cfg.carry_window_s:
            return
        me0 = s0.player(s0.ball.owner_id)
        me1 = st.player(s0.ball.owner_id)
        if me0 is None or me1 is None:
            return
        moved = float(np.linalg.norm(me1.pos - me0.pos))
        kind = T.DRIBBLE if moved >= self.cfg.dribble_min_dist else T.HOLD
        ex = self._example(s0, kind, None, None, actor=s0.ball.owner_id)
        if ex is None:
            return
        ex["outcome"] = {"result": "lost" if lost else "kept", "success": not lost, "end_frame": st.frame,
                         "receiver_id": None, "progress_m": round(float(me1.pos[0] - me0.pos[0]), 2),
                         "xt_gain": round(float(zone_value(st.ball.pos)[0] - zone_value(s0.ball.pos)[0]), 4)}
        self._queue(ex, s0.t)

    def _classify(self, p: _Pending, end_pos: np.ndarray, receiver: Optional[T.PlayerState], st: T.GameState) -> str:
        path = p.ball_path
        rs = p.release_state
        origin = rs.ball.pos
        v0 = path[0][2]
        d = float(np.linalg.norm(end_pos - origin))
        to_goal = np.array([L, W / 2]) - origin
        u = v0 / max(np.linalg.norm(v0), 1e-6)
        # shot: fast, heading at the goal mouth from shooting range
        if origin[0] > L - 38 and np.linalg.norm(v0) > 12 and u[0] > 0.5:
            y_at_goal = origin[1] + u[1] / u[0] * (L - origin[0])
            if abs(y_at_goal - W / 2) < 8 and (receiver is None or receiver.team != T.TEAM_US):
                return T.SHOOT
        # lofted: little deceleration over the flight
        lofted = False
        if len(path) >= 6 and d > 18:
            speeds = np.array([np.linalg.norm(q[2]) for q in path])
            ts = np.array([q[0] for q in path])
            k = len(speeds) // 2
            dec = (speeds[:k].mean() - speeds[k:].mean()) / max(ts[k:].mean() - ts[:k].mean(), 1e-3)
            lofted = dec < 1.0
        if lofted:
            wide = abs(origin[1] - W / 2) > 15 and origin[0] > L - 32
            in_box = end_pos[0] > L - 18 and abs(end_pos[1] - W / 2) < 16
            return T.CROSS if wide and in_box else T.LOB
        if receiver is not None:
            r0 = rs.player(receiver.id)
            if r0 is not None:
                gd = to_goal / max(np.linalg.norm(to_goal), 1e-6)
                lead = float(np.dot(end_pos - r0.pos, gd))
                if lead > 3.0 and np.linalg.norm(end_pos - r0.pos) > 4.0:
                    return T.THROUGH
        return T.PASS

    def _intended(self, p: _Pending) -> Optional[int]:
        """Intended receiver of a failed pass: team-mate best aligned with the ball direction."""
        rs = p.release_state
        v = p.ball_path[0][2]
        if np.linalg.norm(v) < 1:
            return None
        u = v / np.linalg.norm(v)
        best, bs = None, 0.0
        for q in rs.team(T.TEAM_US):
            if q.id == p.passer:
                continue
            dv = q.pos + q.vel * 0.8 - rs.ball.pos
            dist = np.linalg.norm(dv)
            if dist < 3 or dist > 55:
                continue
            cos = float(np.dot(dv / dist, u))
            if cos > 0.96 and cos - dist / 400 > bs:
                best, bs = q.id, cos - dist / 400
        return best

    def _try_resolve(self, st: T.GameState):
        p = self.pending
        b = st.ball
        timeout = st.t - p.t > self.cfg.resolve_timeout_s
        out = not (-1 <= b.pos[0] <= L + 1 and -1 <= b.pos[1] <= W + 1)
        owner = st.player(b.owner_id) if b.owner_id is not None else None
        at_goal = b.pos[0] > L - 0.5 and abs(b.pos[1] - W / 2) < 4.5
        if owner is None and not out and not timeout and not at_goal:
            return
        receiver = owner if (owner is not None and owner.id != p.passer) else None
        if owner is not None and owner.id == p.passer and st.t - p.t < 0.6:
            # touched it on again (dribble touch), not a pass
            self.pending = None
            self.carry_start = None
            return
        kind = self._classify(p, b.pos, receiver, st)
        if kind == T.SHOOT:
            result = "on_target" if at_goal else ("saved" if owner is not None and owner.role == "GK" else "off_target")
            success = result == "on_target"
            target_id = None
        else:
            if out or timeout or at_goal:
                result, success = ("out" if out else "unresolved"), False
                target_id = self._intended(p)
            elif receiver is not None and receiver.team == T.TEAM_US:
                result, success, target_id = "complete", True, receiver.id
            else:
                result, success, target_id = "intercepted", False, self._intended(p)
        self.pending = None
        ex = self._example(p.decision_state, kind, target_id, b.pos, actor=p.passer)
        if ex is None:
            return
        ex["outcome"] = {"result": result, "success": bool(success), "end_frame": st.frame,
                         "receiver_id": None if receiver is None else receiver.id,
                         "progress_m": round(float(b.pos[0] - p.release_state.ball.pos[0]), 2),
                         "xt_gain": round(float(zone_value(b.pos)[0] - zone_value(p.release_state.ball.pos)[0]), 4)}
        self._queue(ex, p.decision_state.t)

    # ------------------------------------------------------------------------
    def _example(self, ds: T.GameState, kind: str, target_id: Optional[int], end_pos, actor: Optional[int] = None) -> Optional[dict]:
        if ds.ball is None:
            return None
        actor = actor if actor is not None else ds.ball.owner_id
        if actor is not None and ds.player(actor) is not None and ds.controlled_id != actor:
            # evaluate from the carrier's point of view (the human was controlling him)
            ds = T.GameState(**{f: getattr(ds, f) for f in ds.__dataclass_fields__})
            ds.players = [T.PlayerState(**{f: getattr(p, f) for f in p.__dataclass_fields__}) for p in ds.players]
            for p in ds.players:
                p.controlled = p.id == actor
            ds.controlled_id = actor
        if ds.controlled is None:
            return None
        cands = self.gen.attacking(ds)
        if not cands:
            return None
        idx = -1
        if kind in T.PASS_KINDS and target_id is not None:
            same = [i for i, a in enumerate(cands) if a.target_id == target_id]
            exact = [i for i in same if cands[i].kind == kind]
            idx = exact[0] if exact else (same[0] if same else -1)
        elif kind in (T.SHOOT, T.DRIBBLE, T.HOLD):
            idx = next((i for i, a in enumerate(cands) if a.kind == kind), -1)
        return {
            "clip": self.clip, "frame": ds.frame, "t": round(ds.t, 4), "state": compact_state(ds),
            "candidates": [{"kind": a.kind, "target_id": a.target_id, "target_label": a.target_label,
                            "target_point": None if a.target_point is None else [round(float(v), 2) for v in a.target_point],
                            "score_phys": round(a.score, 5), "p_phys": round(a.p_success, 4)} for a in cands],
            "features": [[round(float(v), 5) for v in a.features] for a in cands],
            "human": {"kind": kind, "target_id": target_id, "index": idx,
                      "end_point": None if end_pos is None else [round(float(v), 2) for v in end_pos]},
        }

    def _queue(self, ex: dict, t0: float):
        self.waiting_future.append((t0 + self.cfg.future_s, ex))

    def _fill_future(self, st: T.GameState):
        keep = []
        for due, ex in self.waiting_future:
            if st.t >= due:
                ex["outcome"]["possession_after"] = int(st.possession == T.TEAM_US)
                ex["outcome"]["ball_x_after"] = round(float(st.ball.pos[0]), 2)
                self.out.append(ex)
            else:
                keep.append((due, ex))
        self.waiting_future = keep

    def flush(self) -> list:
        done = [ex for _, ex in self.waiting_future]
        for ex in done:
            ex["outcome"].setdefault("possession_after", None)
        self.waiting_future = []
        return done


def write_examples(path: str, examples: list):
    with open(path, "w") as f:
        for e in examples:
            f.write(json.dumps(e) + "\n")


def read_examples(path: str) -> list:
    with open(path) as f:
        return [json.loads(x) for x in f if x.strip()]
