"""Synthetic 11v11 match simulator producing ground truth.

Purpose: a deterministic, labelled stand-in for FC 27 footage so that every
pipeline stage (detection, tracking, calibration, state, tactics, training,
evaluation, latency) can be developed and regression-tested without game
footage.  It is NOT a tactical oracle: the carrier policy is deliberately
noisy so that "what the human did" and "what was good" differ, as in real
play.  Real FC 27 recordings replace it for training the deployed models.

All positions here are raw pitch metres.  Team 0 is "us" (has the controlled
player); team 0 attacks towards +x when ``us_attack_sign == +1``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from fctac.state.formations import ATTACKING_ROLES, template
from fctac.types import PITCH_LENGTH as L, PITCH_SIZE, PITCH_WIDTH as W

G = 9.81
PASS_DECEL = 2.8          # m/s^2 rolling deceleration
CONTROL_R = 1.1           # receiver control radius (m)
INTERCEPT_R = 1.2         # opponent interception radius (m)
TACKLE_R = 1.6


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else np.zeros_like(v)


def seg_point_dist(a, b, p):
    ab = b - a
    t = np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-9), 0.0, 1.0)
    return float(np.linalg.norm(a + t * ab - p)), float(t)


def xg_simple(pos_att: np.ndarray) -> float:
    """Tiny xG model (attack-aligned position) used by sim and baseline."""
    dx = L - pos_att[0]
    dy = abs(pos_att[1] - W / 2)
    d = np.hypot(dx, dy)
    # opening angle of the goal mouth
    a = np.arctan2(7.32 * dx, dx * dx + dy * dy - (7.32 / 2) ** 2)
    a = a if a > 0 else a + np.pi
    z = -1.2 - 0.11 * d + 1.6 * a
    return float(1 / (1 + np.exp(-z)))


@dataclass
class Event:
    kind: str
    t: float
    frame: int
    team: int
    from_id: int
    to_id: Optional[int]
    origin: list
    target: list
    outcome: str = "pending"
    end_t: float = -1.0
    end_frame: int = -1
    receiver_id: Optional[int] = None
    end_pos: Optional[list] = None

    def to_dict(self):
        return dict(self.__dict__)


@dataclass
class SimConfig:
    fps: int = 30
    seed: int = 0
    us_attack_sign: int = 1
    formation_us: str = "4-3-3"
    formation_them: str = "4-4-2"
    max_speed: float = 8.0
    accel: float = 5.0
    decision_temp: float = 0.35     # carrier policy noise


class MatchSim:
    N = 22

    def __init__(self, cfg: SimConfig = SimConfig()):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.dt = 1.0 / cfg.fps
        self.t = 0.0
        self.frame = 0
        self.sign = np.array([cfg.us_attack_sign, -cfg.us_attack_sign])
        roles_us, tpl_us = template(cfg.formation_us)
        roles_th, tpl_th = template(cfg.formation_them)
        self.roles = roles_us + roles_th
        self.team = np.array([0] * 11 + [1] * 11)
        self.tpl = np.concatenate([tpl_us, tpl_th])          # team-attack frame
        self.speed_factor = self.rng.uniform(0.9, 1.05, self.N)
        self.pos = np.zeros((self.N, 2))
        self.vel = np.zeros((self.N, 2))
        self.run_target: list[Optional[np.ndarray]] = [None] * self.N
        self.run_until = np.zeros(self.N)
        self.ref_pos = np.array([L / 2, W / 2 - 10.0])
        self.ref_vel = np.zeros(2)
        # ball
        self.ball = np.array([L / 2, W / 2])
        self.ball_z = 0.0
        self.ball_vel = np.zeros(2)
        self.ball_vz = 0.0
        self.owner: Optional[int] = None
        self.last_touch_team = 0
        self.flight: Optional[Event] = None     # pass/shot in progress
        self.shot_result: Optional[str] = None
        self.decision_at = 0.0
        self.tackle_cooldown = 0.0
        self.restart_at = -1.0
        self.restart_team = 0
        self.controlled = 0
        self.events: list[Event] = []
        self.new_events: list[Event] = []
        self.resolved_events: list[Event] = []
        self.score = [0, 0]
        self._kickoff(0)

    # --- frames --------------------------------------------------------------
    def to_team(self, p, team):
        return p if self.sign[team] > 0 else PITCH_SIZE - p

    def from_team(self, p, team):
        return self.to_team(p, team)

    def _kickoff(self, team):
        for i in range(self.N):
            tp = self.tpl[i].copy()
            tp[0] = tp[0] * 0.48          # own half
            self.pos[i] = self.from_team(tp, self.team[i])
            self.vel[i] = 0.0
            self.run_target[i] = None
        kicker = 9 if team == 0 else 20   # a striker of that team
        self.pos[kicker] = np.array([L / 2, W / 2]) - self.sign[team] * np.array([0.6, 0.0])
        self.ball[:] = (L / 2, W / 2)
        self.ball_z, self.ball_vz = 0.0, 0.0
        self.ball_vel[:] = 0.0
        self.owner = kicker
        self.flight = None
        self.last_touch_team = team
        self.decision_at = self.t + 0.6

    # --- helpers -------------------------------------------------------------
    def poss_team(self) -> int:
        if self.owner is not None:
            return int(self.team[self.owner])
        if self.flight is not None:
            return self.flight.team
        return -1

    def team_idx(self, team):
        return np.where(self.team == team)[0]

    def def_line_x(self, att_team) -> float:
        """Offside-line x in the attacking team's frame (2nd-deepest defender)."""
        d = self.team_idx(1 - att_team)
        xs = np.sort([self.to_team(self.pos[i], att_team)[0] for i in d])[::-1]
        return float(max(xs[1], L / 2))

    def _move(self, i, target, speed):
        d = target - self.pos[i]
        dist = np.linalg.norm(d)
        desired = _unit(d) * min(speed * self.speed_factor[i], 1.8 * dist)
        dv = desired - self.vel[i]
        n = np.linalg.norm(dv)
        amax = self.cfg.accel * self.dt
        if n > amax:
            dv *= amax / n
        self.vel[i] += dv

    # --- off-ball targets ----------------------------------------------------
    def _shape_target(self, i, poss) -> tuple[np.ndarray, float]:
        team = self.team[i]
        ball_t = self.to_team(self.ball, team)
        tp = self.tpl[i].copy()
        role = self.roles[i]
        if role == "GK":
            gx = 3.0 + 0.06 * max(0.0, 50 - (ball_t[0])) * 0.0 + 0.05 * ball_t[0]
            gy = W / 2 + 0.12 * (ball_t[1] - W / 2)
            return self.from_team(np.array([min(gx, 8.0), gy]), team), 4.0
        if poss == team:
            x = tp[0] * 0.75 + (ball_t[0] - 45) * 0.6 + 12
            y = W / 2 + (tp[1] - W / 2) * 1.1 + 0.25 * (ball_t[1] - W / 2)
            spd = 6.0
        else:
            x = tp[0] * 0.65 + (ball_t[0] - 45) * 0.5 + 4
            y = W / 2 + (tp[1] - W / 2) * 0.78 + 0.35 * (ball_t[1] - W / 2)
            spd = 6.5
        x = float(np.clip(x, 4, 101))
        y = float(np.clip(y, 2, W - 2))
        return self.from_team(np.array([x, y]), team), spd

    def _update_offball(self, poss):
        carrier = self.owner
        for i in range(self.N):
            if i == carrier:
                continue
            team = self.team[i]
            target, spd = self._shape_target(i, poss)
            if poss == team:
                # runs in behind for attackers
                rt = self.run_target[i]
                if rt is not None and (self.t > self.run_until[i] or np.linalg.norm(rt - self.pos[i]) < 1.5):
                    self.run_target[i] = rt = None
                if rt is None and self.roles[i] in ATTACKING_ROLES and self.rng.random() < 0.25 * self.dt:
                    me = self.to_team(self.pos[i], team)
                    line = self.def_line_x(team)
                    if me[0] < line - 1.0 and line < 96:
                        tgt = np.array([min(line + self.rng.uniform(8, 16), 100.0),
                                        np.clip(me[1] + 0.35 * (W / 2 - me[1]) + self.rng.normal(0, 4), 4, W - 4)])
                        self.run_target[i] = self.from_team(tgt, team)
                        self.run_until[i] = self.t + 4.0
                        rt = self.run_target[i]
                if rt is not None:
                    target, spd = rt, self.cfg.max_speed
                # receivers of a pass move towards the ball path end
                if self.flight is not None and self.flight.to_id == i and self.flight.kind != "shot":
                    target, spd = self._receive_point(i), self.cfg.max_speed
            else:
                self.run_target[i] = None
            self._move(i, target, spd)
        self._defend(poss)

    def _receive_point(self, i):
        # project where the moving ball will stop / pass the receiver
        b, v = self.ball, self.ball_vel
        s = np.linalg.norm(v)
        if s < 0.5:
            return b.copy()
        stop = b + _unit(v) * (s * s / (2 * PASS_DECEL))
        d, t = seg_point_dist(b, stop, self.pos[i])
        return b + (stop - b) * t

    def _defend(self, poss):
        if poss < 0:
            # loose ball: nearest two of each team chase
            for team in (0, 1):
                idx = self.team_idx(team)
                d = np.linalg.norm(self.pos[idx] - self.ball, axis=1)
                for j in idx[np.argsort(d)[:2]]:
                    if self.roles[j] != "GK":
                        self._move(j, self.ball, self.cfg.max_speed)
            return
        dteam = 1 - poss
        idx = [j for j in self.team_idx(dteam) if self.roles[j] != "GK"]
        if self.flight is not None and self.flight.kind != "shot" and self.ball_z < 1.5:
            # defenders that can reach the remaining ball path in time attack it
            b, v = self.ball, self.ball_vel
            s = np.linalg.norm(v)
            if s > 1.0:
                stop = b + _unit(v) * (s * s / (2 * PASS_DECEL))
                chasers = []
                for j in idx:
                    dd, tt = seg_point_dist(b, stop, self.pos[j])
                    p = b + (stop - b) * tt
                    along = np.linalg.norm(p - b)
                    t_ball = along / max(s, 1e-3)
                    t_def = 0.25 + dd / (self.cfg.max_speed * self.speed_factor[j])
                    if t_def < t_ball + 0.15 and dd < 9:
                        chasers.append((t_def - t_ball, j, p))
                for _, j, p in sorted(chasers, key=lambda c: c[0])[:2]:
                    self._move(j, p, self.cfg.max_speed)
                    idx = [k for k in idx if k != j]
                if not idx:
                    return
        target_pt = self.ball if self.owner is not None else self.ball + self.ball_vel * 0.4
        d = np.linalg.norm(self.pos[idx] - target_pt, axis=1)
        order = np.argsort(d)
        presser = idx[order[0]]
        self._move(presser, target_pt, self.cfg.max_speed * 0.95)
        # cover: second defender goal-side of the ball
        cover = idx[order[1]]
        own_goal = self.from_team(np.array([0.0, W / 2]), dteam)
        self._move(cover, target_pt + 0.35 * (own_goal - target_pt), 7.0)
        # markers: others blend zone with goal-side marking of nearest attacker
        attackers = self.team_idx(poss)
        for j in idx:
            if j in (presser, cover):
                continue
            zone, _ = self._shape_target(j, poss)
            da = np.linalg.norm(self.pos[attackers] - zone, axis=1)
            k = attackers[np.argmin(da)]
            if da.min() < 12:
                mark = self.pos[k] + 1.8 * _unit(own_goal - self.pos[k])
                zone = 0.45 * zone + 0.55 * mark
            self._move(j, zone, 7.0)

    # --- carrier ----------------------------------------------------------------
    def _carrier(self):
        i = self.owner
        team = self.team[i]
        fd = getattr(self, "forced_dribble", None)
        if fd is not None:
            if fd[0] == i and self.t < fd[2]:
                self._move(i, fd[1], 6.3 if np.linalg.norm(fd[1] - self.pos[i]) > 0.5 else 1.0)
                return
            self.forced_dribble = None
        me = self.to_team(self.pos[i], team)
        opp = self.team_idx(1 - team)
        opp_t = np.array([self.to_team(self.pos[j], team) for j in opp])
        dmin = np.min(np.linalg.norm(opp_t - me, axis=1))
        if self.t >= self.decision_at or (dmin < 2.0 and self.rng.random() < 2.5 * self.dt):
            if self._decide(i, me, opp_t, dmin):
                return
            self.decision_at = self.t + self.rng.uniform(0.35, 1.3)
        # dribble towards goal, away from nearest opponents
        goal = np.array([L, W / 2])
        dirv = _unit(goal - me) * 1.0
        for o in opp_t:
            dv = me - o
            dn = np.linalg.norm(dv)
            if dn < 7:
                dirv += _unit(dv) * (7 - dn) / 7 * 0.9
        tgt_t = me + _unit(dirv) * 5
        tgt_t = np.clip(tgt_t, [1, 1], [L - 1, W - 1])
        self._move(i, self.from_team(tgt_t, team), 6.3)

    def _decide(self, i, me, opp_t, dmin) -> bool:
        team = self.team[i]
        mates = [j for j in self.team_idx(team) if j != i]
        opts = []   # (score, kind, receiver, target_raw)
        xg = xg_simple(me)
        if me[0] > L - 30:
            opts.append((xg * 3.0 - 0.05, "shot", None, None))
        for j in mates:
            r = self.to_team(self.pos[j], team)
            rv = self.vel[j] * self.sign[team]
            dist = np.linalg.norm(r - me)
            if dist < 4 or dist > 45:
                continue
            kind = "pass"
            tgt = r + rv * (dist / 15.0)
            if self.run_target[j] is not None and np.linalg.norm(rv) > 4:
                kind = "through"
                tgt = r + rv * (dist / 13.0) + _unit(rv) * 2.5
            elif dist > 30 and self.rng.random() < 0.5:
                kind = "lob"
            elif r[0] > L - 20 and me[0] > L - 30 and abs(me[1] - W / 2) > 18 and self.rng.random() < 0.6:
                kind = "cross"
            tgt = np.clip(tgt, [1, 1], [L - 1, W - 1])
            clear = min(seg_point_dist(me, tgt, o)[0] for o in opp_t)
            space = min(np.linalg.norm(o - tgt) for o in opp_t)
            if kind in ("lob", "cross"):
                clear = max(clear, 3.0)
            p = 1 / (1 + np.exp(-(clear - 1.8) / 0.7)) * 1 / (1 + np.exp(-(space - 2.0) / 1.2))
            p *= np.exp(-dist / 70)
            gain = (xg_simple(tgt) - xg) * 4 + (tgt[0] - me[0]) / 60
            opts.append((p * (0.2 + gain) - (1 - p) * 0.25, kind, j, self.from_team(tgt, team)))
        hold_bias = -0.02 if dmin < 2.5 else 0.06
        opts.append((hold_bias, "dribble", None, None))
        scores = np.array([o[0] for o in opts])
        g = self.rng.gumbel(size=len(opts)) * self.cfg.decision_temp * 0.3
        k = int(np.argmax(scores + g))
        _, kind, j, tgt = opts[k]
        if kind == "dribble":
            return False
        if kind == "shot":
            self._shoot(i, me)
        else:
            self._pass(i, kind, j, tgt)
        return True

    def _new_event(self, kind, i, j, target):
        ev = Event(kind, self.t, self.frame, int(self.team[i]), int(i),
                   None if j is None else int(j), self.pos[i].tolist(), list(map(float, target)))
        self.events.append(ev)
        self.new_events.append(ev)
        return ev

    def _pass(self, i, kind, j, tgt):
        d = tgt - self.ball
        dist = np.linalg.norm(d)
        u = _unit(d)
        if kind in ("pass", "through"):
            arrive = 5.0 if kind == "pass" else 3.5
            v0 = np.sqrt(arrive ** 2 + 2 * PASS_DECEL * dist)
            v0 = min(v0, 26.0)
            self.ball_vel = u * v0
            self.ball_vz = 0.0
        else:
            tflight = 1.0 + dist / 25.0
            self.ball_vel = u * dist / tflight
            self.ball_vz = 0.5 * G * tflight
        noise = self.rng.normal(0, 0.03)
        c, s = np.cos(noise), np.sin(noise)
        self.ball_vel = np.array([c * self.ball_vel[0] - s * self.ball_vel[1],
                                  s * self.ball_vel[0] + c * self.ball_vel[1]])
        self.flight = self._new_event(kind, i, j, tgt)
        self.owner = None
        self.last_touch_team = int(self.team[i])
        self.release_t = self.t

    def _shoot(self, i, me):
        team = self.team[i]
        post = self.rng.choice([-1, 1])
        ty = W / 2 + post * self.rng.uniform(1.0, 3.4)
        target_t = np.array([L + 0.5, ty])
        target = self.from_team(target_t, team)
        u = _unit(target - self.ball)
        self.ball_vel = u * 24.0
        self.ball_vz = self.rng.uniform(0.5, 3.0)
        p_goal = xg_simple(me)
        r = self.rng.random()
        self.shot_result = "goal" if r < p_goal else ("saved" if r < p_goal + 0.45 else "wide")
        self.flight = self._new_event("shot", i, None, target)
        self.owner = None
        self.last_touch_team = int(team)
        self.release_t = self.t

    # --- external control (counterfactual evaluation) --------------------------
    def force_action(self, kind: str, receiver: Optional[int] = None, target_raw=None):
        """Make the current carrier execute an action now (used to evaluate
        recommendations by rolling the match forward)."""
        i = self.owner
        if i is None:
            return False
        team = self.team[i]
        if kind in ("pass", "through", "lob", "cross") and receiver is not None:
            tgt = np.asarray(target_raw if target_raw is not None else self.pos[receiver], float)
            self._pass(i, kind, int(receiver), np.clip(tgt, [1, 1], [L - 1, W - 1]))
        elif kind == "shot":
            self._shoot(i, self.to_team(self.pos[i], team))
        elif kind == "dribble" and target_raw is not None:
            self.forced_dribble = (i, np.asarray(target_raw, float), self.t + 1.0)
            self.decision_at = self.t + 1.0
        elif kind == "hold":
            self.forced_dribble = (i, self.pos[i].copy(), self.t + 1.0)
            self.decision_at = self.t + 1.0
        return True

    # --- ball -------------------------------------------------------------
    def _ball_step(self):
        if self.owner is not None:
            o = self.owner
            dirv = _unit(self.vel[o]) if np.linalg.norm(self.vel[o]) > 0.3 else self.sign[self.team[o]] * np.array([1.0, 0])
            self.ball = self.pos[o] + dirv * 0.55
            self.ball_vel = self.vel[o].copy()
            self.ball_z, self.ball_vz = 0.0, 0.0
            return
        # free ball physics
        self.ball = self.ball + self.ball_vel * self.dt
        if self.ball_z > 0 or self.ball_vz > 0:
            self.ball_vz -= G * self.dt
            self.ball_z += self.ball_vz * self.dt
            if self.ball_z <= 0:
                self.ball_z = 0.0
                self.ball_vz = -self.ball_vz * 0.35 if abs(self.ball_vz) > 2 else 0.0
                self.ball_vel *= 0.75
        else:
            s = np.linalg.norm(self.ball_vel)
            if s > 0:
                ns = max(0.0, s - PASS_DECEL * self.dt)
                self.ball_vel *= ns / s

    def _resolve(self, outcome, receiver=None):
        ev = self.flight
        if ev is not None:
            ev.outcome = outcome
            ev.end_t = self.t
            ev.end_frame = self.frame
            ev.receiver_id = None if receiver is None else int(receiver)
            ev.end_pos = self.ball.tolist()
            self.resolved_events.append(ev)
        self.flight = None

    def _check_control(self):
        if self.owner is not None:
            return
        ev = self.flight
        if ev is not None and ev.kind == "shot":
            return self._shot_flight(ev)
        if self.ball_z > 1.9:
            return
        if self.t - getattr(self, "release_t", -1) < 0.15:
            return
        d = np.linalg.norm(self.pos - self.ball, axis=1)
        bs = np.linalg.norm(self.ball_vel)
        order = np.argsort(d)
        for j in order[:4]:
            if d[j] > CONTROL_R:
                break
            same = ev is None or self.team[j] == ev.team
            if same:
                p = 0.9
            else:
                # faster balls are harder to cut out
                p = 0.7 * np.clip(1.5 - bs / 25.0, 0.35, 1.0) if d[j] < INTERCEPT_R else 0.0
            if self.rng.random() < p:
                if ev is not None:
                    self._resolve("complete" if self.team[j] == ev.team else "intercepted", j)
                self.owner = int(j)
                self.last_touch_team = int(self.team[j])
                self.decision_at = self.t + self.rng.uniform(0.25, 0.8)
                return

    def _shot_flight(self, ev):
        team = ev.team
        b = self.to_team(self.ball, team)
        gk = int([j for j in self.team_idx(1 - team) if self.roles[j] == "GK"][0])
        if self.shot_result == "saved" and b[0] > L - 6:
            self._resolve("saved", gk)
            self.owner = gk
            self.pos[gk] = self.ball.copy()
            self.last_touch_team = 1 - team
            self.decision_at = self.t + 1.0
        elif b[0] >= L:
            if self.shot_result == "goal":
                self._resolve("goal")
                self.score[team] += 1
                self.restart_at = self.t + 1.0
                self.restart_team = 1 - team
            else:
                self._resolve("wide")
                self._goal_kick(1 - team)

    def _goal_kick(self, team):
        gk = int([j for j in self.team_idx(team) if self.roles[j] == "GK"][0])
        p = self.from_team(np.array([5.5, W / 2 + self.rng.uniform(-8, 8)]), team)
        self.pos[gk] = p
        self.ball = p.copy()
        self.ball_vel[:] = 0
        self.ball_z = self.ball_vz = 0.0
        self.owner = gk
        self.last_touch_team = team
        self.decision_at = self.t + 1.0

    def _check_out(self):
        b = self.ball
        if 0 <= b[0] <= L and 0 <= b[1] <= W:
            return
        if self.flight is not None and self.flight.kind == "shot":
            return
        team = 1 - self.last_touch_team
        if self.flight is not None:
            self._resolve("out")
        if not (0 <= b[0] <= L):
            # over the goal line: goal kick or corner simplified to goal kick / throw
            defending_goal_team = 0 if (b[0] < 0) == (self.sign[0] > 0) else 1
            if team == defending_goal_team:
                return self._goal_kick(team)
        p = np.clip(b, [0.5, 0.5], [L - 0.5, W - 0.5])
        idx = self.team_idx(team)
        j = idx[np.argmin(np.linalg.norm(self.pos[idx] - p, axis=1))]
        self.pos[j] = p
        self.ball = p.copy()
        self.ball_vel[:] = 0
        self.ball_z = self.ball_vz = 0.0
        self.owner = int(j)
        self.last_touch_team = team
        self.decision_at = self.t + 0.8

    def _tackles(self):
        if self.owner is None or self.t < self.tackle_cooldown:
            return
        c = self.owner
        opp = self.team_idx(1 - self.team[c])
        d = np.linalg.norm(self.pos[opp] - self.pos[c], axis=1)
        k = int(np.argmin(d))
        if d[k] < TACKLE_R and self.rng.random() < 1.6 * self.dt:
            self.owner = int(opp[k])
            self.last_touch_team = int(self.team[opp[k]])
            self.tackle_cooldown = self.t + 1.0
            self.decision_at = self.t + 0.5
            ev = self._new_event("tackle", opp[k], c, self.pos[c])
            ev.outcome, ev.end_t, ev.end_frame, ev.receiver_id = "won", self.t, self.frame, int(opp[k])
            self.resolved_events.append(ev)

    def _update_controlled(self):
        poss = self.poss_team()
        if self.owner is not None and self.team[self.owner] == 0:
            self.controlled = self.owner
            return
        if self.flight is not None and self.flight.team == 0 and self.flight.to_id is not None:
            self.controlled = self.flight.to_id
            return
        idx = [j for j in self.team_idx(0) if self.roles[j] != "GK"]
        d = np.linalg.norm(self.pos[idx] - self.ball, axis=1)
        best = idx[int(np.argmin(d))]
        cur = self.controlled
        dcur = np.linalg.norm(self.pos[cur] - self.ball) if self.team[cur] == 0 else 1e9
        if best != cur and (dcur - d.min() > 3.0 or poss != 1):
            self.controlled = int(best)

    def _referee(self):
        tgt = self.ball + np.array([-8.0 * self.sign[self.last_touch_team], -9.0])
        tgt = np.clip(tgt, [2, 2], [L - 2, W - 2])
        d = tgt - self.ref_pos
        desired = _unit(d) * min(6.0, 1.5 * np.linalg.norm(d))
        dv = desired - self.ref_vel
        n = np.linalg.norm(dv)
        if n > 4 * self.dt:
            dv *= 4 * self.dt / n
        self.ref_vel += dv
        self.ref_pos += self.ref_vel * self.dt

    # --- main step ---------------------------------------------------------
    def step(self):
        self.new_events = []
        self.resolved_events = []
        if self.restart_at > 0 and self.t >= self.restart_at:
            self.restart_at = -1
            self._kickoff(self.restart_team)
        poss = self.poss_team()
        if self.owner is not None:
            self._carrier()
        self._update_offball(poss)
        self.pos += self.vel * self.dt
        self.pos = np.clip(self.pos, [-2, -2], [L + 2, W + 2])
        self._referee()
        self._ball_step()
        self._check_control()
        self._check_out()
        self._tackles()
        self._update_controlled()
        self.t += self.dt
        self.frame += 1

    def snapshot(self) -> dict:
        return {
            "frame": self.frame,
            "t": round(self.t, 4),
            "us_attack_sign": int(self.sign[0]),
            "players": [
                {"id": i, "team": int(self.team[i]), "role": self.roles[i],
                 "x": float(self.pos[i, 0]), "y": float(self.pos[i, 1]),
                 "vx": float(self.vel[i, 0]), "vy": float(self.vel[i, 1]),
                 "controlled": bool(i == self.controlled)}
                for i in range(self.N)
            ],
            "referee": {"x": float(self.ref_pos[0]), "y": float(self.ref_pos[1])},
            "ball": {"x": float(self.ball[0]), "y": float(self.ball[1]), "z": float(self.ball_z),
                     "vx": float(self.ball_vel[0]), "vy": float(self.ball_vel[1]),
                     "owner": self.owner},
            "possession": self.poss_team(),
            "score": list(self.score),
            "events": [e.to_dict() for e in self.new_events],
            "resolved": [e.to_dict() for e in self.resolved_events],
        }
