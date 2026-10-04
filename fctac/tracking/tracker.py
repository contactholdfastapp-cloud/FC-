"""Multi-object tracker in pitch space fusing radar dots and main-view detections.

Each player gets a persistent internal id (PLAYER_01, ...).  Radar dots
(all players, team-labelled, ~0.2-0.5 m noise) and projected main-view
detections (visible players, accurate on screen) are associated with
Hungarian matching in two passes and fused by per-track Kalman filters.
The ball has its own, more agile filter.  When the radar is unavailable the
tracker runs on main-view detections alone.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

from fctac import types as T
from fctac.pitch.camera import apply_h
from fctac.tracking.kalman import CVKalman


@dataclass
class TrackerConfig:
    radar_var: float = 0.35 ** 2
    view_var: float = 0.45 ** 2
    ball_radar_var: float = 0.3 ** 2
    ball_view_var: float = 0.35 ** 2
    gate_radar: float = 4.0
    gate_view: float = 2.5
    max_missed_s: float = 0.6
    q_player: float = 8.0
    q_ball: float = 25.0
    ctrl_alpha: float = 0.35          # EMA rate of controlled evidence
    ctrl_switch_margin: float = 0.15
    ball_gate_m: float = 4.0
    ball_reinit_after: int = 6


@dataclass
class Track:
    id: int
    team: int
    kf: CVKalman
    hits: int = 1
    age: float = 0.0
    last_seen_t: float = 0.0
    last_view_t: float = -1.0
    screen: Optional[np.ndarray] = None
    screen_t: float = -1.0
    ctrl: float = 0.0
    acc: np.ndarray = field(default_factory=lambda: np.zeros(2))
    prev_vel: np.ndarray = field(default_factory=lambda: np.zeros(2))
    team_votes: np.ndarray = field(default_factory=lambda: np.zeros(3))
    role: str = ""

    @property
    def name(self) -> str:
        return f"PLAYER_{self.id:02d}"

    @property
    def confidence(self) -> float:
        return float(np.clip(1.0 - self.kf.pos_std() / 3.0, 0.0, 1.0))


@dataclass
class BallTrack:
    kf: Optional[CVKalman] = None
    last_seen_t: float = -1e9
    screen: Optional[np.ndarray] = None
    screen_t: float = -1e9
    conf: float = 0.0
    rejects: int = 0


class Tracker:
    def __init__(self, cfg: TrackerConfig = TrackerConfig()):
        self.cfg = cfg
        self.tracks: list[Track] = []
        self.ball = BallTrack()
        self.next_id = 1
        self.t: Optional[float] = None
        self.controlled_id: Optional[int] = None

    def reset(self):
        self.__init__(self.cfg)

    # ------------------------------------------------------------------------
    def step(self, t: float, radar=None, dets: Optional[list] = None, H_img2pitch: Optional[np.ndarray] = None,
             calib_conf: float = 0.0) -> list[Track]:
        dt = 0.0 if self.t is None else max(0.0, t - self.t)
        self.t = t
        for tr in self.tracks:
            tr.kf.predict(dt)
            tr.age += dt
        if self.ball.kf is not None:
            self.ball.kf.predict(dt)

        ctrl_evidence: dict[int, float] = {}
        # --- pass 1: radar ---------------------------------------------------------
        if radar is not None and radar.ok and len(radar.points):
            matched = self._associate(radar.points, radar.teams, self.cfg.gate_radar, hard_team=True)
            for ti, mi in matched:
                tr = self.tracks[ti]
                tr.kf.update(radar.points[mi], self.cfg.radar_var)
                tr.hits += 1
                tr.last_seen_t = t
                tr.team_votes[min(radar.teams[mi], 2)] += 1
                if radar.controlled is not None:
                    ctrl_evidence[tr.id] = 1.0 if mi == radar.controlled else 0.0
            used = {mi for _, mi in matched}
            for mi in range(len(radar.points)):
                if mi in used:
                    continue
                team = int(radar.teams[mi])
                if sum(1 for tr in self.tracks if tr.team == team) >= 11:
                    continue
                tr = self._new_track(radar.points[mi], team, t)
                if radar.controlled == mi:
                    ctrl_evidence[tr.id] = 1.0
            if radar.ball is not None:
                self._ball_update(radar.ball, self.cfg.ball_radar_var, t)

        # --- pass 2: main-view detections ------------------------------------------
        players = [d for d in (dets or []) if d.cls == T.CLS_PLAYER and d.team != T.TEAM_REF]
        if players and H_img2pitch is not None and calib_conf > 0.2:
            img = np.array([[d.x, d.y] for d in players])
            pp = apply_h(H_img2pitch, img)
            # only confidently classified kits constrain the association
            teams = np.array([d.team if d.team_conf > 0.4 else T.TEAM_UNKNOWN for d in players])
            matched = self._associate(pp, teams, self.cfg.gate_view, hard_team=True)
            r = self.cfg.view_var / max(calib_conf, 0.3)
            for ti, mi in matched:
                tr = self.tracks[ti]
                tr.kf.update(pp[mi], r)
                tr.screen = img[mi].copy()
                tr.screen_t = t
                tr.last_view_t = t
                tr.last_seen_t = t
                if players[mi].team in (T.TEAM_US, T.TEAM_THEM):
                    tr.team_votes[players[mi].team] += 0.5
                if players[mi].controlled:
                    ctrl_evidence[tr.id] = max(ctrl_evidence.get(tr.id, 0.0), 1.0)
            if radar is None or not radar.ok:
                used = {mi for _, mi in matched}
                for mi in range(len(players)):
                    if mi not in used and players[mi].team in (T.TEAM_US, T.TEAM_THEM):
                        if sum(1 for tr in self.tracks if tr.team == players[mi].team) < 11:
                            tr = self._new_track(pp[mi], int(players[mi].team), t)
                            tr.screen, tr.screen_t = img[mi].copy(), t
            balls = [d for d in (dets or []) if d.cls == T.CLS_BALL]
            if balls:
                b = balls[0]
                bp = apply_h(H_img2pitch, np.array([b.x, b.y]))
                radar_ball = radar.ball if (radar is not None and radar.ok) else None
                if radar_ball is not None:
                    # the radar arbitrates: a view ball elsewhere is a white kit/line
                    # (or a ball in the air, whose ground projection is wrong)
                    if np.linalg.norm(bp - radar_ball) < 1.5:
                        self.ball.screen, self.ball.screen_t = np.array([b.x, b.y]), t
                        self._ball_update(bp, self.cfg.ball_view_var / max(calib_conf, 0.3), t)
                elif self.ball.kf is None or np.linalg.norm(bp - self.ball.kf.pos) < 3.0:
                    self.ball.screen, self.ball.screen_t = np.array([b.x, b.y]), t
                    self._ball_update(bp, self.cfg.ball_view_var / max(calib_conf, 0.3), t)

        # --- maintenance --------------------------------------------------------------
        self.tracks = [tr for tr in self.tracks if t - tr.last_seen_t <= self.cfg.max_missed_s]
        for tr in self.tracks:
            v = tr.kf.vel
            if dt > 0:
                tr.acc = 0.7 * tr.acc + 0.3 * (v - tr.prev_vel) / dt
            tr.prev_vel = v.copy()
            if tr.team_votes.sum() > 0:
                tr.team = int(np.argmax(tr.team_votes))
        self._controlled(ctrl_evidence)
        self.ball.conf = float(np.exp(-max(0.0, t - self.ball.last_seen_t) / 0.4)) if self.ball.kf is not None else 0.0
        return self.tracks

    # ------------------------------------------------------------------------
    def _new_track(self, p, team, t) -> Track:
        tr = Track(id=self.next_id, team=team, kf=CVKalman(p, pos_var=0.5, vel_var=16.0, q_acc=self.cfg.q_player),
                   last_seen_t=t)
        if team in (0, 1, 2):
            tr.team_votes[team] = 1
        self.next_id += 1
        self.tracks.append(tr)
        return tr

    def _associate(self, pts, teams, gate, hard_team) -> list[tuple[int, int]]:
        if not self.tracks or not len(pts):
            return []
        tp = np.array([tr.kf.pos for tr in self.tracks])
        D = np.linalg.norm(tp[:, None, :] - pts[None, :, :], axis=2)
        tt = np.array([tr.team for tr in self.tracks])
        known = (tt[:, None] >= 0) & (np.asarray(teams)[None, :] >= 0)
        mismatch = known & (tt[:, None] != np.asarray(teams)[None, :])
        D = D + np.where(mismatch, 1e3 if hard_team else 2.0, 0.0)
        a, b = linear_sum_assignment(D)
        return [(int(i), int(j)) for i, j in zip(a, b) if D[i, j] < gate]

    def _ball_update(self, p, var, t):
        b = self.ball
        if b.kf is None or t - b.last_seen_t > 1.0 or b.rejects >= self.cfg.ball_reinit_after:
            b.kf = CVKalman(p, pos_var=var, vel_var=100.0, q_acc=self.cfg.q_ball)
            b.rejects = 0
        else:
            # gate: measurements far from the prediction are other white objects
            innov = float(np.linalg.norm(np.asarray(p) - b.kf.pos))
            if innov > self.cfg.ball_gate_m + 2.0 * b.kf.pos_std():
                b.rejects += 1
                return
            b.rejects = 0
            b.kf.update(p, var)
        b.last_seen_t = t

    def _controlled(self, ev: dict):
        a = self.cfg.ctrl_alpha
        for tr in self.tracks:
            if tr.id in ev:
                tr.ctrl = (1 - a) * tr.ctrl + a * ev[tr.id]
            elif ev:
                tr.ctrl *= (1 - a)        # evidence went elsewhere this frame
        ours = [tr for tr in self.tracks if tr.team == T.TEAM_US]
        if not ours:
            self.controlled_id = None
            return
        best = max(ours, key=lambda tr: tr.ctrl)
        cur = next((tr for tr in ours if tr.id == self.controlled_id), None)
        if cur is None or (best.id != cur.id and best.ctrl > cur.ctrl + self.cfg.ctrl_switch_margin):
            self.controlled_id = best.id if best.ctrl > 0.2 else None
