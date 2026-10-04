"""Tracks -> structured GameState (attack-aligned pitch coordinates)."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

from fctac import types as T
from fctac.pitch.camera import apply_h
from fctac.state.roles import RoleTracker


@dataclass
class StateConfig:
    attack_sign: int = 0              # +1 / -1 fixed, 0 = infer from goalkeeper positions
    owner_radius: float = 1.6
    owner_rel_speed: float = 3.5
    owner_max_ball_speed: float = 9.5     # faster balls are passes/shots, not dribbles
    kick_accel: float = 90.0              # m/s^2 (3 m/s per frame at 30 fps) marks a touch
    kick_radius: float = 3.0
    isolated_ball_m: float = 8.0          # slow ball this far from every player for >1 s = implausible
    min_tracks: int = 8


class StateBuilder:
    def __init__(self, cfg: StateConfig = StateConfig()):
        self.cfg = cfg
        self.roles_us = RoleTracker()
        self.roles_them = RoleTracker()
        self.sign_votes: deque = deque(maxlen=150)
        self.possession = T.TEAM_UNKNOWN
        self.owner_id: Optional[int] = None
        self.last_touch_id: Optional[int] = None
        self._prev_ball_vel: Optional[np.ndarray] = None
        self._prev_t: Optional[float] = None
        self._isolated_since: Optional[float] = None

    def reset(self):
        self.__init__(self.cfg)

    def attack_sign(self, tracks) -> int:
        if self.cfg.attack_sign in (1, -1):
            return self.cfg.attack_sign
        us = [tr.kf.pos[0] for tr in tracks if tr.team == T.TEAM_US]
        them = [tr.kf.pos[0] for tr in tracks if tr.team == T.TEAM_THEM]
        if len(us) >= 6 and len(them) >= 6:
            # each goalkeeper is the deepest player towards its own goal
            left = min(us) - min(them)            # < 0: our goalkeeper guards the left goal
            right = max(them) - max(us)           # > 0: their goalkeeper guards the right goal
            self.sign_votes.append(1 if (right - left) > 0 else -1)
        if not self.sign_votes:
            return 1
        return 1 if sum(self.sign_votes) >= 0 else -1

    def build(self, frame: int, t: float, tracks, ball, controlled_id, H_img2pitch, calib_conf) -> T.GameState:
        sign = self.attack_sign(tracks)
        gs = T.GameState(frame=frame, t=t, players=[], ball=None, attack_sign=sign,
                         H_img2pitch=H_img2pitch, calib_conf=float(calib_conf), controlled_id=controlled_id)
        H_p2i = None
        if H_img2pitch is not None:
            H_p2i = np.linalg.inv(H_img2pitch)
        for tr in tracks:
            if tr.team not in (T.TEAM_US, T.TEAM_THEM):
                continue
            pos = gs.to_attack(tr.kf.pos)
            vel = gs.vel_to_raw(tr.kf.vel)
            acc = gs.vel_to_raw(tr.acc)
            screen = None
            if tr.screen is not None and abs(tr.screen_t - t) < 1e-6:
                screen = tr.screen.copy()
            elif H_p2i is not None:
                screen = apply_h(H_p2i, tr.kf.pos)
            gs.players.append(T.PlayerState(id=tr.id, team=tr.team, pos=pos, vel=vel, acc=acc,
                                            controlled=tr.id == controlled_id, confidence=tr.confidence,
                                            screen=screen, visible=abs(tr.last_view_t - t) < 1e-6))
        self._roles(gs)
        if ball is not None and ball.kf is not None and ball.conf > 0.05:
            bpos = gs.to_attack(ball.kf.pos)
            bvel = gs.vel_to_raw(ball.kf.vel)
            bscreen = None
            if ball.screen is not None and abs(ball.screen_t - t) < 1e-6:
                bscreen = ball.screen.copy()
            elif H_p2i is not None:
                bscreen = apply_h(H_p2i, ball.kf.pos)
            conf = float(ball.conf)
            # plausibility: a slow ball nobody is near for > 1 s is a pitch marking /
            # false detection, not the ball -> never base advice on it
            near = min((float(np.linalg.norm(p.pos - bpos)) for p in gs.players), default=99.0)
            if near > self.cfg.isolated_ball_m and float(np.linalg.norm(bvel)) < 3.0:
                self._isolated_since = self._isolated_since if self._isolated_since is not None else t
                if t - self._isolated_since > 1.0:
                    conf = min(conf, 0.1)
            else:
                self._isolated_since = None
            gs.ball = T.BallState(pos=bpos, vel=bvel, confidence=conf, screen=bscreen)
            self._possession(gs)
        gs.possession = self.possession
        if len(gs.players) < self.cfg.min_tracks:
            gs.valid, gs.invalid_reason = False, "too few players tracked"
        return gs

    def _roles(self, gs: T.GameState):
        for team, rt, attr in ((T.TEAM_US, self.roles_us, "formation_us"), (T.TEAM_THEM, self.roles_them, "formation_them")):
            ps = [p for p in gs.players if p.team == team]
            if not ps:
                continue
            pos = np.array([p.pos if team == T.TEAM_US else T.PITCH_SIZE - p.pos for p in ps])
            roles = rt.update([p.id for p in ps], pos)
            seen = set()
            # unique labels: first come keeps it, duplicates fall back to the track id
            for p in sorted(ps, key=lambda p: -len(rt.votes[p.id])):
                r = roles.get(p.id, "")
                if r and r not in seen:
                    p.role = r
                    seen.add(r)
            setattr(gs, attr, rt.formation)

    def _possession(self, gs: T.GameState):
        """Ball ownership with hysteresis + touch detection.

        * owner = player moving *with* the ball (distance + velocity match), kept
          while plausible, replaced only after a challenger wins for a few frames
          (a presser next to the dribbler is not the owner);
        * a sudden ball acceleration is a touch: by the previous owner if there
          was one (pass/shot: possession unchanged), else by the nearest player
          (interception, goal kick, loose ball).
        """
        b = gs.ball
        if not gs.players:
            return
        speed = float(np.linalg.norm(b.vel))
        raw_v = gs.vel_to_raw(b.vel)
        prev_owner = self.owner_id
        kicked = False
        if self._prev_ball_vel is not None and self._prev_t is not None and gs.t > self._prev_t:
            acc = float(np.linalg.norm(raw_v - self._prev_ball_vel)) / (gs.t - self._prev_t)
            kicked = acc > self.cfg.kick_accel
        self._prev_ball_vel, self._prev_t = raw_v, gs.t
        d = np.array([np.linalg.norm(p.pos - b.pos) for p in gs.players])
        dv = np.array([np.linalg.norm(p.vel - b.vel) for p in gs.players])
        cost = d + 0.35 * dv
        cand = (d < self.cfg.owner_radius) & (dv < self.cfg.owner_rel_speed) & (speed < self.cfg.owner_max_ball_speed)
        if kicked:
            if prev_owner is not None:
                p = gs.player(prev_owner)
                if p is not None:
                    self.possession = p.team
                    self.last_touch_id = p.id
            elif d.min() < self.cfg.kick_radius:
                p = gs.players[int(np.argmin(d))]
                self.possession = p.team
                self.last_touch_id = p.id
            self.owner_id = None
            self._challenger, self._challenger_n = None, 0
        elif cand.any():
            k = int(np.argmin(np.where(cand, cost, np.inf)))
            best = gs.players[k]
            cur_idx = next((i for i, p in enumerate(gs.players) if p.id == self.owner_id), None)
            if cur_idx is not None and cand[cur_idx] and cost[cur_idx] < cost[k] + 0.8:
                best = gs.players[cur_idx]
                self._challenger, self._challenger_n = None, 0
            elif cur_idx is not None and cand[cur_idx]:
                if getattr(self, "_challenger", None) == best.id:
                    self._challenger_n += 1
                else:
                    self._challenger, self._challenger_n = best.id, 1
                if self._challenger_n < 3:
                    best = gs.players[cur_idx]
            self.owner_id = best.id
            self.possession = best.team
            self.last_touch_id = best.id
        else:
            self.owner_id = None
        if self.owner_id is not None:
            p = gs.player(self.owner_id)
            b.owner_id, b.owner_team = p.id, p.team
        else:
            b.owner_id, b.owner_team = None, T.TEAM_UNKNOWN
