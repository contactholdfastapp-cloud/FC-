"""Screen <-> pitch calibration by registering main-view detections to the radar.

The radar gives every player's pitch position; the detector gives their foot
points on screen.  Matching the two sets yields point correspondences, from
which a free homography (screen -> raw pitch) is estimated each frame:
  * works for any camera style (no fixed camera model needed),
  * handles pan/zoom/tilt changes automatically,
  * self-validates: inlier count + residual -> calibration confidence.

Initialisation (no previous homography) searches a coarse grid of broadcast
cameras for the one that best explains detections given the radar, then
refines.  Manual landmark calibration (tools/calibrate.py) can seed it too.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from fctac import types as T
from fctac.pitch.camera import CameraParams, apply_h


@dataclass
class CalibConfig:
    min_matches: int = 6
    gate_init_m: float = 6.0
    gate_m: float = 2.5
    ransac_m: float = 1.5
    team_penalty_m: float = 4.0
    lost_after: int = 20             # frames of failed updates before re-initialising
    cam_pos: tuple = (52.5, -40.0, 20.0)   # typical broadcast camera for the init search
    hold_conf_decay: float = 0.93    # confidence decay per frame while coasting
    line_tracking: bool = True       # keep the homography alive from pitch lines when radar registration fails
    offpitch_margin_m: float = 2.0   # detections projected further outside the pitch are dropped
    # init search camera family: "pan" = fixed position that pans (synthetic renderer), "dolly" = camera
    # that slides along the touchline following the ball, looking straight across (FC 27 broadcast cam,
    # fitted on real footage: ~50 m back, ~27 m up, tilt ~0.34 rad, f ~1.65 x width), "both"
    init_mode: str = "both"
    dolly_back: tuple = ((-45.0, 25.0), (-50.0, 27.5), (-56.0, 30.5))   # (y, z) camera distance / height
    init_every: int = 3              # while lost, run the (costly) init search on every Nth update
    init_accept: float = 0.035       # max init-search score (normalised point distance) to accept


class RadarViewCalibrator:
    def __init__(self, cfg: CalibConfig = CalibConfig()):
        self.cfg = cfg
        self.H: Optional[np.ndarray] = None      # screen -> raw pitch
        self.conf = 0.0
        self.fails = 0
        self.last_matches: list = []
        self.last_rmse = 0.0
        self._init_tries = 0

    def set_manual(self, H_img2pitch: np.ndarray, conf: float = 0.9):
        self.H = H_img2pitch / H_img2pitch[2, 2]
        self.conf = conf
        self.fails = 0

    def reset(self):
        self.H = None
        self.conf = 0.0
        self.fails = 0

    # ------------------------------------------------------------------------
    def update(self, dets: list, radar_pts: np.ndarray, radar_teams: np.ndarray, width: int, height: int,
               ball: Optional[np.ndarray] = None):
        players = [d for d in dets if d.cls == T.CLS_PLAYER and d.team != T.TEAM_REF]
        if len(players) < self.cfg.min_matches or len(radar_pts) < self.cfg.min_matches:
            return self._coast()
        img = np.array([[d.x, d.y] for d in players], float)
        dteam = np.array([d.team for d in players], int)
        H = self.H
        gate = self.cfg.gate_m
        if H is None:
            self._init_tries += 1
            if (self._init_tries - 1) % max(1, self.cfg.init_every):
                return self._coast()
            H = self._init_search(img, radar_pts, width, height, ball)
            if H is None:
                return self._coast()
            gate = self.cfg.gate_init_m
        best = None
        for it in range(3):
            pp = apply_h(H, img)
            D = np.linalg.norm(pp[:, None, :] - radar_pts[None, :, :], axis=2)
            known = (dteam[:, None] >= 0) & (radar_teams[None, :] >= 0)
            D = D + np.where(known & (dteam[:, None] != radar_teams[None, :]), self.cfg.team_penalty_m, 0.0)
            a, b = linear_sum_assignment(D)
            ok = D[a, b] < (max(gate, 5.0) if it == 0 else self.cfg.gate_m)
            a, b = a[ok], b[ok]
            if len(a) < self.cfg.min_matches:
                break
            Hn, mask = cv2.findHomography(img[a], radar_pts[b], cv2.RANSAC, self.cfg.ransac_m)
            if Hn is None or not self._plausible(Hn, width, height):
                break
            inl = mask.ravel().astype(bool)
            if inl.sum() < self.cfg.min_matches:
                break
            res = np.linalg.norm(apply_h(Hn, img[a][inl]) - radar_pts[b][inl], axis=1)
            H = Hn / Hn[2, 2]
            best = (H, int(inl.sum()), float(np.sqrt(np.mean(res ** 2))), list(zip(a[inl], b[inl])))
        if best is None:
            return self._coast()
        H, n, rmse, matches = best
        self.H = H
        self.fails = 0
        self._init_tries = 0
        self.last_matches = matches
        self.last_rmse = rmse
        self.conf = float(np.clip((n - 4) / 8.0, 0, 1) * np.exp(-rmse / 1.5))
        return self.H, self.conf

    def reacquire(self, dets: list, radar_pts: np.ndarray, radar_teams: np.ndarray, width: int, height: int,
                  ball: Optional[np.ndarray] = None) -> bool:
        """Fresh radar-based init ignoring the current homography (which may come from line
        tracking that slid onto a wrong lock).  Keeps the old state if the init fails."""
        old = (self.H, self.conf, self.fails, self._init_tries)
        self.H, self._init_tries = None, 0
        H, conf = self.update(dets, radar_pts, radar_teams, width, height, ball)
        if H is None or len(self.last_matches) < max(8, self.cfg.min_matches):
            self.H, self.conf, self.fails, self._init_tries = old
            return False
        return True

    def _coast(self):
        self.last_matches = []          # matches index this frame's radar/detections: stale when coasting
        self.fails += 1
        self.conf *= self.cfg.hold_conf_decay
        if self.fails > self.cfg.lost_after:
            self.reset()
        return self.H, self.conf

    @staticmethod
    def _plausible(H, width, height) -> bool:
        """Reject degenerate fits: image corners must map to a sane, non-flipped area."""
        c = np.array([[0, height * 0.35], [width, height * 0.35], [width, height], [0, height]], float)
        q = apply_h(H, c)
        if not np.all(np.isfinite(q)) or np.abs(q).max() > 400:
            return False
        # the visible ground quad must have a real area (not collapsed to a line)
        def cross2(a, b):
            return a[0] * b[1] - a[1] * b[0]
        area = 0.5 * cross2(q[1] - q[0], q[3] - q[0]) + 0.5 * cross2(q[3] - q[2], q[1] - q[2])
        return abs(area) > 50

    def _grid_scores(self, fk, tl, yw, img, radar_pts, width, height, pos=None):
        """pos: (N,3) camera positions per hypothesis (default: cfg.cam_pos for all)."""
        if pos is None:
            pos = np.broadcast_to(np.array(self.cfg.cam_pos, float), (len(fk), 3))
        f = fk * width
        cyw, syw, ct, st = np.cos(yw), np.sin(yw), np.cos(tl), np.sin(tl)
        fwd = np.stack([syw * ct, cyw * ct, -st], 1)
        right = np.stack([cyw, -syw, np.zeros_like(yw)], 1)
        down = np.cross(fwd, right)
        R = np.stack([right, down, fwd], 1)                                  # (N,3,3)
        P = np.c_[radar_pts, np.zeros(len(radar_pts))][None, :, :] - pos[:, None, :]   # (N,P,3)
        Xc = np.einsum("nij,npj->npi", R, P)                                  # (N,P,3)
        z = Xc[..., 2]
        zs = np.where(np.abs(z) < 1e-6, 1e-6, z)
        u = (f[:, None] * Xc[..., 0] / zs + width / 2) / width
        v = (f[:, None] * Xc[..., 1] / zs + height / 2) / width
        vis = (z > 0) & (u > -0.04) & (u < 1.04) & (v > -0.04) & (v < height / width + 0.04)
        imgn = img / width
        d = np.hypot(imgn[None, :, 0, None] - u[:, None, :], imgn[None, :, 1, None] - v[:, None, :])   # (N,D,P)
        d = np.where(vis[:, None, :], d, 1.0)
        s1 = np.minimum(d.min(2), 0.05).mean(1)
        dp = np.where(vis, np.minimum(d.min(1), 0.05), 0.0)
        s2 = dp.sum(1) / np.maximum(vis.sum(1), 1)
        return np.where(vis.sum(1) >= 4, s1 + 0.5 * s2, np.inf)

    def _search(self, fk, tl, yw, pos, img, radar_pts, width, height):
        sc = self._grid_scores(fk, tl, yw, img, radar_pts, width, height, pos)
        k = int(np.argmin(sc))
        return (float(sc[k]), float(fk[k]), float(tl[k]), float(yw[k]), pos[k].copy()) if np.isfinite(sc[k]) else None

    def _pan_coarse(self, img, radar_pts, width, height):
        fk, tl, yw = np.meshgrid(np.array([1.0, 1.25, 1.5, 1.8, 2.2, 2.7]), np.linspace(0.14, 0.50, 10),
                                 np.linspace(-0.75, 0.75, 31), indexing="ij")
        fk, tl, yw = fk.ravel(), tl.ravel(), yw.ravel()
        pos = np.broadcast_to(np.array(self.cfg.cam_pos, float), (len(fk), 3))
        return self._search(fk, tl, yw, pos, img, radar_pts, width, height)

    def _dolly_coarse(self, img, radar_pts, width, height, ball):
        # the camera trails the ball by a few metres; the score is sharp in x -> 2 m steps
        if ball is not None:
            xs = np.clip(float(ball[0]) + np.arange(-16.0, 13.0, 2.0), -5.0, 110.0)
        else:
            xs = np.arange(-5.0, 111.0, 2.5)
        fks = np.array([1.45, 1.64, 1.85])
        tls = np.array([0.31, 0.343, 0.375])
        X, B, F, Tl = np.meshgrid(xs, np.arange(len(self.cfg.dolly_back)), fks, tls, indexing="ij")
        X, B, F, Tl = X.ravel(), B.ravel(), F.ravel(), Tl.ravel()
        back = np.array(self.cfg.dolly_back, float)
        pos = np.c_[X, back[B, 0], back[B, 1]]
        return self._search(F, Tl, np.zeros_like(F), pos, img, radar_pts, width, height)

    def _init_search(self, img: np.ndarray, radar_pts: np.ndarray, width: int, height: int,
                     ball: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """Coarse-to-fine grid search over broadcast cameras (vectorised): a panning camera at a
        fixed position and/or a dolly camera sliding along the touchline (see CalibConfig.init_mode)."""
        cands = []
        if self.cfg.init_mode in ("pan", "both"):
            r = self._pan_coarse(img, radar_pts, width, height)
            if r:
                cands.append(("pan", r))
        if self.cfg.init_mode in ("dolly", "both"):
            r = self._dolly_coarse(img, radar_pts, width, height, ball)
            if r:
                cands.append(("dolly", r))
        if not cands:
            return None
        kind, (sc, fk, tl, yw, p) = min(cands, key=lambda c: c[1][0])
        # refine around the best hypothesis
        if kind == "pan":
            f2, t2, y2 = np.meshgrid(fk * np.linspace(0.85, 1.15, 9), tl + np.linspace(-0.03, 0.03, 5),
                                     yw + np.linspace(-0.04, 0.04, 5), indexing="ij")
            pos2 = np.broadcast_to(p, (f2.size, 3))
        else:
            dx, f2, t2, y2 = np.meshgrid(np.arange(-3.0, 3.5, 1.0), fk * np.array([0.94, 1.0, 1.06]),
                                         tl + np.array([-0.015, 0.0, 0.015]), np.array([-0.04, 0.0, 0.04]),
                                         indexing="ij")
            pos2 = np.c_[p[0] + dx.ravel(), np.full(dx.size, p[1]), np.full(dx.size, p[2])]
        r = self._search(f2.ravel(), t2.ravel(), y2.ravel(), np.ascontiguousarray(pos2), img, radar_pts, width, height)
        # real footage is noisier than the renderer (missed/merged players): 0.035; the
        # Hungarian + RANSAC fit that follows still has to find >= min_matches inliers
        if r is None or r[0] > self.cfg.init_accept:
            return None
        sc, fk, tl, yw, p = r
        cam = CameraParams(float(p[0]), float(p[1]), float(p[2]), yw, tl, fk * width, width, height)
        return cam.H_img2pitch()
