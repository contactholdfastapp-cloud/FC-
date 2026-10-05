"""Homography tracking without the radar: camera motion + pitch-line refinement.

1. Frame-to-frame camera motion: ECC image registration (affine) of the pitch
   texture (grass stripes, lines) on a small grey image, players and HUD
   masked out.  pitch->screen is propagated through that motion.
2. Pitch-line refinement: the line model is projected with the propagated
   homography and an image-space affine correction is fitted so projected
   markings fall on observed white lines (distance-transform residuals).
   This removes the drift of step 1 whenever enough markings are visible.

Used when the radar is hidden or too few players are visible for radar
registration; seeded by the last good calibration (or manual landmarks).
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
from scipy.optimize import least_squares

from fctac.pitch.camera import apply_h
from fctac.pitch.model import sample_line_points


class LineRefiner:
    def __init__(self, work_width: int = 640, step_m: float = 1.0, cap_px: float = 12.0,
                 grass_lo=(33, 45, 35), grass_hi=(92, 255, 255)):
        self.ww = work_width
        self.model = sample_line_points(step_m)
        self.cap = cap_px
        self.grass_lo, self.grass_hi = grass_lo, grass_hi
        self._k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def line_mask(self, small: np.ndarray, hsv=None) -> np.ndarray:
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV) if hsv is None else hsv
        white = ((hsv[..., 1] < 70) & (hsv[..., 2] > 165)).astype(np.uint8) * 255
        grass = cv2.inRange(hsv, self.grass_lo, self.grass_hi)
        lines = cv2.bitwise_and(white, cv2.dilate(grass, self._k))
        thick = cv2.morphologyEx(lines, cv2.MORPH_OPEN, self._k)       # white kits are thick
        return cv2.bitwise_and(lines, cv2.bitwise_not(thick))

    def refine_small(self, small: np.ndarray, H0: np.ndarray, mask: np.ndarray) -> tuple[Optional[np.ndarray], float]:
        """H0: pitch -> small-image homography.  Returns refined H (same frame) and inlier ratio."""
        sh, sw = small.shape[:2]
        if mask.sum() < 255 * 80:
            return None, 0.0
        dt = cv2.distanceTransform(255 - mask, cv2.DIST_L2, 3)
        q = np.c_[self.model, np.ones(len(self.model))] @ H0.T
        uv = q[:, :2] / np.where(np.abs(q[:, 2:3]) < 1e-9, 1e-9, q[:, 2:3])
        ok = (q[:, 2] > 0) & (uv[:, 0] >= 2) & (uv[:, 0] < sw - 2) & (uv[:, 1] >= 2) & (uv[:, 1] < sh - 2)
        if ok.sum() < 40:
            return None, 0.0
        base = uv[ok]
        if len(base) > 700:
            base = base[np.linspace(0, len(base) - 1, 700).astype(int)]
        # 8-parameter homography correction in normalised image coordinates (centre
        # origin, half-width units): corrects pan, zoom AND perspective (tilt) drift
        c = np.array([sw / 2, sh / 2])
        n = sw / 2.0
        un = (base - c) / n

        def warp(p, u):
            Hc = np.array([[1 + p[0], p[1], p[2]], [p[3], 1 + p[4], p[5]], [p[6], p[7], 1.0]])
            q = np.c_[u, np.ones(len(u))] @ Hc.T
            return q[:, :2] / q[:, 2:3]

        def resid(p):
            u = warp(p, un) * n + c
            x = np.clip(u[:, 0], 0, sw - 1.001).astype(np.float32).reshape(1, -1)
            y = np.clip(u[:, 1], 0, sh - 1.001).astype(np.float32).reshape(1, -1)
            return np.minimum(cv2.remap(dt, x, y, cv2.INTER_LINEAR).ravel(), self.cap)

        r0 = resid(np.zeros(8))
        sol = least_squares(resid, np.zeros(8), loss="soft_l1", f_scale=2.0, max_nfev=50,
                            x_scale=np.full(8, 0.01), diff_step=1e-3)
        conf = float(np.mean(sol.fun < 2.0))
        if np.mean(sol.fun) > np.mean(r0) or conf < 0.3:
            return None, float(np.mean(r0 < 2.0))
        p = sol.x
        Hc = np.array([[1 + p[0], p[1], p[2]], [p[3], 1 + p[4], p[5]], [p[6], p[7], 1.0]])
        N = np.array([[1 / n, 0, -c[0] / n], [0, 1 / n, -c[1] / n], [0, 0, 1.0]])
        return np.linalg.inv(N) @ Hc @ N @ H0, conf


class LineTracker:
    """Keeps a screen<->pitch homography alive without the radar."""

    def __init__(self, work_width: int = 640, ecc_width: int = 256, ecc_iters: int = 25, hud_rects=()):
        self.ref = LineRefiner(work_width)
        self.ecc_w = ecc_width
        self.ecc_iters = ecc_iters
        self.hud = hud_rects
        self.prev_gray: Optional[np.ndarray] = None
        self.prev_warp = np.eye(3, dtype=np.float32)
        self.conf = 0.0

    def reset(self):
        self.prev_gray = None
        self.prev_warp = np.eye(3, dtype=np.float32)

    def _ecc_mask(self, small_bgr):
        hsv = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2HSV)
        grass = cv2.inRange(hsv, self.ref.grass_lo, self.ref.grass_hi)
        white = ((hsv[..., 1] < 70) & (hsv[..., 2] > 165)).astype(np.uint8) * 255
        m = cv2.bitwise_or(grass, white)
        m = cv2.erode(m, np.ones((3, 3), np.uint8))          # drop player borders
        h, w = m.shape
        for x0, y0, x1, y1 in self.hud:
            m[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)] = 0
        return m

    def update(self, frame: np.ndarray, H_img2pitch: np.ndarray) -> tuple[Optional[np.ndarray], float]:
        """Never raises: a degenerate homography (real footage: cuts, replays) -> (None, 0)."""
        try:
            return self._update(frame, H_img2pitch)
        except (np.linalg.LinAlgError, cv2.error):
            self.conf = 0.0
            self.prev_gray = None
            return None, 0.0

    def _update(self, frame: np.ndarray, H_img2pitch: np.ndarray) -> tuple[Optional[np.ndarray], float]:
        h, w = frame.shape[:2]
        se = self.ecc_w / w
        tiny = cv2.resize(frame, (self.ecc_w, int(round(h * se))), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(tiny, cv2.COLOR_BGR2GRAY)
        H_p2i = np.linalg.inv(H_img2pitch)
        motion_ok = True
        if self.prev_gray is not None and self.prev_gray.shape == gray.shape:
            # a panning perspective camera moves the image by a homography (an affine
            # model drifts metres within a second); warm-start from the last motion
            warp = self.prev_warp.copy()
            try:
                _, warp = cv2.findTransformECC(self.prev_gray, gray, warp, cv2.MOTION_HOMOGRAPHY,
                                               (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, self.ecc_iters, 1e-4),
                                               self._ecc_mask(tiny), 5)
                # warp maps previous-frame pixels to current-frame pixels (tiny scale)
                S = np.diag([se, se, 1.0])
                H_p2i = np.linalg.inv(S) @ warp @ S @ H_p2i
                self.prev_warp = warp
            except cv2.error:
                motion_ok = False
                self.prev_warp = np.eye(3, dtype=np.float32)
        self.prev_gray = gray
        # line refinement on the working image
        s = self.ref.ww / w
        small = cv2.resize(frame, (self.ref.ww, int(round(h * s))), interpolation=cv2.INTER_AREA)
        mask = self.ref.line_mask(small)
        sh, sw = mask.shape
        for x0, y0, x1, y1 in self.hud:
            mask[int(y0 * sh):int(y1 * sh), int(x0 * sw):int(x1 * sw)] = 0
        S = np.diag([s, s, 1.0])
        Hs, lconf = self.ref.refine_small(small, S @ H_p2i, mask)
        if Hs is not None:
            H_p2i = np.linalg.inv(S) @ Hs
            self.conf = min(1.0, 0.5 + 0.6 * lconf)
        else:
            # motion-only propagation drifts: confidence decays
            self.conf *= 0.97 if motion_ok else 0.8
        try:
            H = np.linalg.inv(H_p2i)
        except np.linalg.LinAlgError:
            self.conf = 0.0
            return None, 0.0
        if not np.all(np.isfinite(H)) or abs(H[2, 2]) < 1e-12:
            self.conf = 0.0
            return None, 0.0
        return H / H[2, 2], self.conf
