"""Read FC's 2D radar (mini-map) into pitch coordinates.

The radar is an orthographic top-down map of the whole pitch drawn by the
game HUD: every player appears as a team-coloured dot, the ball as a white
dot, the controlled player is marked.  Reading it is cheap (a ~250x150 px
ROI, < 1 ms on CPU), covers off-screen players and needs no perspective
calibration, so it is the backbone of the global game state.  The main-view
detector refines positions near the ball.

Everything is configurable because HUD layout/colours depend on game
settings: ROI (normalised screen rect), pitch rect inside the ROI, dot
colours (or auto-clustering), controlled marker colour.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from fctac.types import PITCH_LENGTH as L, PITCH_WIDTH as W, TEAM_THEM, TEAM_US
from fctac.vision.colors import TeamPrototypes, bgr_to_lab


@dataclass
class RadarConfig:
    enabled: bool = True
    rect: tuple = (0.415, 0.795, 0.585, 0.985)      # x0, y0, x1, y1 normalised screen coords
    pitch_rect: tuple = (0.0, 0.0, 1.0, 1.0)        # pitch area inside the ROI (normalised to ROI)
    flip: bool = False                              # radar drawn rotated 180 deg vs. main camera
    color_us: Optional[tuple] = None                # BGR; None = auto (via controlled marker)
    color_them: Optional[tuple] = None
    marker_color: tuple = (0, 230, 255)             # controlled-player ring (BGR)
    color_tol: float = 32.0                         # Lab distance for dot colour membership
    min_dots: int = 10
    dot_area_px: float = 0.0                        # single-dot area; 0 = estimate online
    # Real FC 27 radar (2D, default HUD): the dark pitch panel, normalised
    # screen coords, measured on 1080p broadcast footage (813-1104 x 865-1025 px).
    panel: tuple = (0.4234, 0.8009, 0.5750, 0.9491)
    mode: str = "color"                             # "color" (dot radar, synthetic) | "fc27" (learned reader)
    fc27_model: str = "registry"                    # radar model for mode "fc27": "registry" or a path
    us_shape: str = "auto"                          # FC 27: which radar shape is your team: auto | triangle | circle
    auto_align: bool = True                         # FC 27: refine the panel position from the first radar frames


def panel_px(panel: tuple, w: int, h: int) -> tuple[int, int, int, int]:
    """Radar pitch panel in pixels (x0, y0, x1, y1)."""
    return (int(round(panel[0] * w)), int(round(panel[1] * h)), int(round(panel[2] * w)), int(round(panel[3] * h)))


def radar_crop_rect(panel: tuple, w: int, h: int, margin: float = 0.10) -> tuple[int, int, int, int]:
    """Panel plus a margin (fraction of the panel width) on every side, clipped to the frame."""
    x0, y0, x1, y1 = panel_px(panel, w, h)
    m = int(round(margin * (x1 - x0)))
    return max(0, x0 - m), max(0, y0 - m), min(w, x1 + m), min(h, y1 + m)


@dataclass
class RadarResult:
    ok: bool
    points: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))   # raw pitch metres
    teams: np.ndarray = field(default_factory=lambda: np.zeros(0, int))
    controlled: Optional[int] = None               # index into points
    ball: Optional[np.ndarray] = None
    roi_px: tuple = (0, 0, 0, 0)


class RadarReader:
    def __init__(self, cfg: RadarConfig = RadarConfig()):
        self.cfg = cfg
        self.protos = TeamPrototypes(max_dist=cfg.color_tol)
        if cfg.color_us is not None:
            self.protos.us = bgr_to_lab([cfg.color_us])[0]
        if cfg.color_them is not None:
            self.protos.them = bgr_to_lab([cfg.color_them])[0]
        self.marker_lab = bgr_to_lab([cfg.marker_color])[0]
        self.marker_hue = int(cv2.cvtColor(np.uint8([[cfg.marker_color]]), cv2.COLOR_BGR2HSV)[0, 0, 0])
        self.single_area = {TEAM_US: cfg.dot_area_px or 0.0, TEAM_THEM: cfg.dot_area_px or 0.0}
        self._k = np.ones((2, 2), np.uint8)

    def roi(self, h: int, w: int) -> tuple[int, int, int, int]:
        x0, y0, x1, y1 = self.cfg.rect
        return int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)

    def to_pitch(self, uv: np.ndarray, rw: int, rh: int) -> np.ndarray:
        px0, py0, px1, py1 = self.cfg.pitch_rect
        u = (uv[:, 0] / rw - px0) / (px1 - px0)
        v = (uv[:, 1] / rh - py0) / (py1 - py0)
        p = np.stack([u * L, (1.0 - v) * W], 1)
        if self.cfg.flip:
            p = np.array([L, W]) - p
        return p

    def read(self, frame: np.ndarray, ball_prior: Optional[np.ndarray] = None) -> RadarResult:
        h, w = frame.shape[:2]
        x0, y0, x1, y1 = self.roi(h, w)
        roi = frame[y0:y1, x0:x1]
        if roi.size == 0:
            return RadarResult(False)
        # the controlled marker is a thin ring: find it at full resolution
        hsv_full = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        dh = np.abs(hsv_full[..., 0].astype(np.int16) - self.marker_hue)
        dh = np.minimum(dh, 180 - dh)
        hue_mask = ((dh <= 7) & (hsv_full[..., 1] > 110) & (hsv_full[..., 2] > 140)).astype(np.uint8)
        marker, marker_mask = self._marker(hue_mask, roi.shape[0])
        f = int(round(self._px_scale(roi.shape[0])))
        if f >= 2:      # high-res HUD (e.g. 1440p): dots at ~720p HUD scale, ~4x cheaper
            roi = cv2.resize(roi, (roi.shape[1] // f, roi.shape[0] // f), interpolation=cv2.INTER_AREA)
            marker_mask = cv2.resize(marker_mask, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_NEAREST)
            if marker is not None:
                marker = marker / f
        rh, rw = roi.shape[:2]
        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
        labf = lab.reshape(-1, 3).astype(np.float32)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        sat = hsv[..., 1].reshape(-1)
        val = hsv[..., 2].reshape(-1)


        if not self.protos.ready():
            self._auto_colors(labf, sat, val, rh, rw, marker)
            if not self.protos.ready():
                return RadarResult(False, roi_px=(x0, y0, x1, y1))

        prior_px = None
        if ball_prior is not None:
            q = np.array([L, W]) - ball_prior if self.cfg.flip else np.asarray(ball_prior, float)
            px0, py0, px1, py1 = self.cfg.pitch_rect
            prior_px = np.array([(q[0] / L * (px1 - px0) + px0) * rw, ((1 - q[1] / W) * (py1 - py0) + py0) * rh])
        ball = self._ball(roi, sat, val, rh, rw, prior_px)
        pts, teams = [], []
        for team, proto in ((TEAM_US, self.protos.us), (TEAM_THEM, self.protos.them)):
            d = np.linalg.norm(labf - proto, axis=1).reshape(rh, rw)
            m = ((d < self.cfg.color_tol) & (marker_mask == 0)).astype(np.uint8)
            n, lbl, stats, cent = cv2.connectedComponentsWithStats(m, connectivity=8)
            areas = stats[1:, cv2.CC_STAT_AREA].astype(float)
            bw, bh = stats[1:, cv2.CC_STAT_WIDTH], stats[1:, cv2.CC_STAT_HEIGHT]
            ps = self._px_scale(rh)
            # dot-like blobs only (pitch lines are long and thin)
            expected = 28.0 * ps * ps                 # nominal single-dot area at this HUD scale
            keep = (areas >= 0.25 * expected) & (np.maximum(bw, bh) <= 3 * np.minimum(bw, bh) + 2) & (np.maximum(bw, bh) <= 22 * ps)
            if not keep.any():
                continue
            single = keep & (areas > 0.4 * expected) & (areas < 1.6 * expected)
            if single.sum() >= 4:
                est = float(np.median(areas[single]))
                cur = self.single_area[team]
                self.single_area[team] = est if cur <= 0 else 0.9 * cur + 0.1 * est
            sa = self.single_area[team] if self.single_area[team] > 0 else expected
            for i in np.where(keep)[0]:
                near_marker = marker is not None and np.hypot(*(cent[i + 1] - marker)) < 6 * ps
                # the ball icon is drawn over the carrier's dot: keep partly hidden dots there
                db = np.hypot(*(cent[i + 1] - ball)) if ball is not None else 1e9
                if db < 1.2 * ps and areas[i] < 0.8 * sa:
                    continue                      # this blob IS the ball (team colour close to white)
                near_ball = db < 5 * ps
                if areas[i] < 0.6 * sa and not (near_marker or near_ball):   # smaller than a player dot
                    continue
                k = int(np.clip(np.round(areas[i] / max(sa, 1.0)), 1, 3))
                for _ in range(k):
                    pts.append(cent[i + 1])
                    teams.append(team)
        if len(pts) < self.cfg.min_dots:
            return RadarResult(False, roi_px=(x0, y0, x1, y1))
        uv = np.array(pts, float)
        teams = np.array(teams, int)
        controlled = None
        if marker is not None:
            us = np.where(teams == TEAM_US)[0]
            if len(us):
                d = np.linalg.norm(uv[us] - marker, axis=1)
                if d.min() < 6 * self._px_scale(rh):
                    controlled = int(us[np.argmin(d)])
        res = RadarResult(True, self.to_pitch(uv, rw, rh), teams, controlled,
                          None if ball is None else self.to_pitch(ball[None, :], rw, rh)[0], (x0, y0, x1, y1))
        return res

    def _marker(self, hue_mask: np.ndarray, rh: int):
        """The controlled marker is a ring: pick the hollow hue-matched blob, so a
        team whose dots share the marker colour is not mistaken for it."""
        empty = np.zeros_like(hue_mask)
        if hue_mask.sum() < 3:
            return None, empty
        n, lbl, stats, cent = cv2.connectedComponentsWithStats(hue_mask, connectivity=8)
        ps = self._px_scale(rh)
        best, bscore = None, 1e9
        for i in range(1, n):
            x, y, bw, bh, a = stats[i]
            if a < 3:
                continue
            fill = a / float(bw * bh)
            size = max(bw, bh)
            if size < 6 * ps or size > 20 * ps:
                continue
            # a ring's centre pixel is not part of it
            cxi, cyi = int(round(cent[i][0])), int(round(cent[i][1]))
            hollow = lbl[cyi, cxi] != i
            score = fill + (0.0 if hollow else 1.0)
            if score < bscore:
                best, bscore = i, score
        if best is None or bscore > 0.8:
            return None, empty
        m = np.where(lbl == best, 1, 0).astype(np.uint8)
        return np.asarray(cent[best], float), cv2.dilate(m, self._k)

    def _px_scale(self, rh: int) -> float:
        return max(rh / 143.0, 0.5)          # ROI height at 720p is ~143 px

    def _ball(self, roi, sat, val, rh, rw, prior_px=None):
        m = ((sat.reshape(rh, rw) < 40) & (val.reshape(rh, rw) > 225)).astype(np.uint8)
        n, lbl, stats, cent = cv2.connectedComponentsWithStats(m, connectivity=8)
        best, bscore = None, 0.0
        k = self._px_scale(rh)
        known = [a for a in self.single_area.values() if a > 0]
        sa = min(known) if known else 28.0 * k * k
        for i in range(1, n):
            a = stats[i, cv2.CC_STAT_AREA]
            bw, bh = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            if a < 2 or a > 0.75 * sa or max(bw, bh) > 8 * k:
                continue
            fill = a / float(bw * bh)
            score = fill * min(bw, bh) / max(bw, bh)
            if prior_px is not None:          # continuity: the ball moves < ~1 m per frame
                score *= np.exp(-np.hypot(*(cent[i] - prior_px)) / (6.0 * k))
            if score > bscore:
                best, bscore = cent[i], score
        return None if best is None else np.asarray(best, float)

    def _auto_colors(self, labf, sat, val, rh, rw, marker):
        """Cluster foreground radar pixels into colours, keep the two that form
        many dot-sized blobs; the team whose dot sits in the controlled marker
        is ours.  Low-contrast kits should be set explicitly in the config."""
        if marker is None:
            return
        bg = np.median(labf, axis=0)
        fg = np.linalg.norm(labf - bg, axis=1) > 30
        fg &= np.linalg.norm(labf - self.marker_lab, axis=1) > 30   # lines/ball are rejected by blob shape below
        if fg.sum() < 60:
            return
        centres = TeamPrototypes.cluster(labf[fg], k=5)
        cands = []
        for c in centres:
            d = np.linalg.norm(labf - c, axis=1).reshape(rh, rw)
            m = (d < self.cfg.color_tol).astype(np.uint8)
            n, _, stats, cent = cv2.connectedComponentsWithStats(m, connectivity=8)
            a = stats[1:, cv2.CC_STAT_AREA]
            bw, bh = stats[1:, cv2.CC_STAT_WIDTH], stats[1:, cv2.CC_STAT_HEIGHT]
            dots = (a >= 6) & (a <= 40 * self._px_scale(rh) ** 2) & (np.maximum(bw, bh) < 3 * np.minimum(bw, bh) + 2)
            if dots.sum() < 4:
                continue
            cands.append({"c": c, "pts": cent[1:][dots], "contrast": float(np.linalg.norm(c - bg))})
        # anti-aliasing halos form their own colour cluster around each dot: when two
        # clusters mark the same objects keep the higher-contrast one
        cands.sort(key=lambda z: -z["contrast"])
        kept = []
        for z in cands:
            dup = False
            for k in kept:
                dd = np.linalg.norm(z["pts"][:, None, :] - k["pts"][None, :, :], axis=2).min(1)
                if np.mean(dd < 3.0 * self._px_scale(rh)) > 0.6:
                    dup = True
                    break
            if not dup:
                kept.append(z)
        kept.sort(key=lambda z: -len(z["pts"]))
        if len(kept) < 2:
            return
        scored = []
        for z in kept[:2]:
            dm = float(np.min(np.hypot(z["pts"][:, 0] - marker[0], z["pts"][:, 1] - marker[1])))
            scored.append((len(z["pts"]), dm, z["c"]))
        a, b = scored
        if np.linalg.norm(a[2] - b[2]) < self.cfg.color_tol:
            return
        us, them = (a, b) if a[1] <= b[1] else (b, a)
        if us[1] > 8 * self._px_scale(rh):
            return
        protos = [us[2].copy(), them[2].copy()]
        # refine each prototype to the median colour of its dots' pixels (k-means
        # centres are pulled towards anti-aliased edges)
        for _ in range(2):
            for t in range(2):
                d = np.linalg.norm(labf - protos[t], axis=1)
                sel = d < self.cfg.color_tol
                if sel.sum() >= 20:
                    core = labf[sel]
                    dd = np.linalg.norm(core - np.median(core, 0), axis=1)
                    protos[t] = np.median(core[dd < np.percentile(dd, 60)], 0)
        self.protos.us, self.protos.them = protos[0], protos[1]
