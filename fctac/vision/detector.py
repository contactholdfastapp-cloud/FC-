"""Baseline main-view detector (classical CV, CPU, a few ms).

Pitch = largest grass region; players = compact non-grass blobs on it; ball
= small compact white blob; controlled player = blob under the controlled
indicator colour.  It is the fallback/bootstrapping detector: it produces
the first labels for the FC 27 dataset and a baseline the learned detector
(fctac.vision.learned) must beat.  Works on a downscaled copy of the frame
(``work_width``); all outputs are in full-resolution screen pixels.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from fctac import types as T


@dataclass
class DetectorConfig:
    work_width: int = 640
    grass_lo: tuple = (33, 45, 35)          # HSV lower bound of pitch grass
    grass_hi: tuple = (92, 255, 255)
    min_h: float = 0.022                    # player blob height, fraction of work height
    max_h: float = 0.30
    indicator_color: tuple = (255, 210, 0)  # BGR of FC's controlled-player marker
    indicator_hue_tol: int = 8
    hud_masks: list = field(default_factory=lambda: [(0.415, 0.795, 0.585, 0.985), (0.0, 0.0, 0.3, 0.12)])
    ball_full_res: bool = True              # refine ball at full resolution around prior
    ball_global: bool = True                # full-frame full-res ball search when there is no prior
                                            # (off for FC 27: the radar gives the ball; saves ~20-40 ms)
    resize: str = "area"                    # area (default: keeps the ball at 1440p) | linear (~4 ms faster, loses ball recall)


class ColorDetector:
    name = "color"

    def __init__(self, cfg: DetectorConfig = DetectorConfig()):
        self.cfg = cfg
        self.ind_hue = int(cv2.cvtColor(np.uint8([[cfg.indicator_color]]), cv2.COLOR_BGR2HSV)[0, 0, 0])
        self._k2 = np.ones((2, 2), np.uint8)
        self._k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self._k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self._kv = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 7))
        self._kbig = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
        self.last_pitch_mask: Optional[np.ndarray] = None
        self.last_scale = 1.0
        self.ball_exclude: np.ndarray = np.zeros((0, 2))   # screen points that look like a ball (pitch spots)

    def _hud_mask(self, h, w) -> np.ndarray:
        m = np.full((h, w), 255, np.uint8)
        for x0, y0, x1, y1 in self.cfg.hud_masks:
            m[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)] = 0
        return m

    def detect(self, frame: np.ndarray, ball_prior: Optional[np.ndarray] = None) -> list[T.Detection]:
        H, W = frame.shape[:2]
        ww = self.cfg.work_width
        s = ww / W
        wh = int(round(H * s))
        interp = cv2.INTER_AREA if self.cfg.resize == "area" else cv2.INTER_LINEAR
        small = cv2.resize(frame, (ww, wh), interpolation=interp) if s != 1 else frame
        self.last_scale = s
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        grass = cv2.inRange(hsv, self.cfg.grass_lo, self.cfg.grass_hi)
        hud = self._hud_mask(wh, ww)
        # pitch region = biggest closed grass area (players inside become part of it)
        closed = cv2.morphologyEx(grass, cv2.MORPH_CLOSE, self._kbig)
        n, lbl, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=4)
        if n <= 1:
            return []
        big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        pitch = np.where(lbl == big, 255, 0).astype(np.uint8)
        cnts, _ = cv2.findContours(pitch, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hull = cv2.convexHull(max(cnts, key=cv2.contourArea))
        pitch = np.zeros_like(pitch)
        cv2.fillConvexPoly(pitch, hull, 255)
        pitch = cv2.bitwise_and(pitch, hud)
        self.last_pitch_mask = pitch

        sat, val = hsv[..., 1], hsv[..., 2]
        white = ((sat < 60) & (val > 175)).astype(np.uint8) * 255
        # pitch lines = thin bright structures: white top-hat keeps them, drops
        # thick white regions (white kits), so only lines are removed from fg
        thin_white = cv2.bitwise_and(white, cv2.bitwise_not(cv2.morphologyEx(white, cv2.MORPH_OPEN, self._k5)))
        # only long thin structures are lines (white kits also have thin parts)
        n, lbl, stats, _ = cv2.connectedComponentsWithStats(thin_white, connectivity=8)
        long_ = np.zeros(n, np.uint8)
        long_[1:] = (np.maximum(stats[1:, cv2.CC_STAT_WIDTH], stats[1:, cv2.CC_STAT_HEIGHT]) > 0.09 * wh)
        thin_white = (long_[lbl] * 255).astype(np.uint8)
        fg = cv2.bitwise_and(cv2.bitwise_not(grass), pitch)
        fg = cv2.bitwise_and(fg, cv2.bitwise_not(cv2.dilate(thin_white, self._k3)))
        # remove noise, then merge body parts vertically
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, self._k2)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, self._kv)

        # controlled indicator mask (excluded from player blobs)
        hue = hsv[..., 0].astype(np.int16)
        dh = np.abs(hue - self.ind_hue)
        dh = np.minimum(dh, 180 - dh)
        ind = ((dh <= self.cfg.indicator_hue_tol) & (sat > 150) & (val > 150)).astype(np.uint8)
        ind = cv2.bitwise_and(ind, ind, mask=hud)

        dets: list[T.Detection] = []
        lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
        n, lbl, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
        min_h, max_h = self.cfg.min_h * wh, self.cfg.max_h * wh
        for i in range(1, n):
            x, y, bw, bh, area = stats[i]
            if bh < min_h or bh > max_h:
                continue
            ar = bh / max(bw, 1)
            if ar < 0.9 or area < 0.25 * bw * bh:
                continue
            if ind[y:y + bh, x:x + bw].mean() > 0.3:      # the marker itself
                continue
            # split side-by-side merged players (width >> expected for height)
            k = int(np.clip(round(bw / (0.55 * bh)), 1, 3)) if ar < 1.4 else 1
            for j in range(k):
                xs0 = x + j * bw / k
                xs1 = x + (j + 1) * bw / k
                sub = (lbl[y:y + bh, int(xs0):int(np.ceil(xs1))] == i)
                rows = np.nonzero(sub.any(1))[0]
                if not len(rows):
                    continue
                top, bot = y + rows[0], y + rows[-1] + 1
                ty0, ty1 = int(top + 0.18 * (bot - top)), int(top + 0.5 * (bot - top))
                torso = lab[ty0:max(ty1, ty0 + 1), int(xs0):int(np.ceil(xs1))]
                tm = (lbl[ty0:max(ty1, ty0 + 1), int(xs0):int(np.ceil(xs1))] == i)
                feat = torso[tm].mean(0) if tm.any() else lab[top:bot, int(xs0):int(np.ceil(xs1))].reshape(-1, 3).mean(0)
                cx = 0.5 * (xs0 + xs1)
                d = T.Detection(cls=T.CLS_PLAYER, x=cx / s, y=bot / s, w=(xs1 - xs0) / s, h=(bot - top) / s,
                                conf=float(min(1.0, area / (0.45 * bw * bh))), feature=feat.astype(np.float32))
                dets.append(d)
        self._mark_controlled(dets, ind, s)
        ball = self._ball(frame, small, white, pitch, fg, s, ball_prior, dets)
        if ball is not None:
            dets.append(ball)
        return dets

    def _mark_controlled(self, dets, ind, s):
        """Exactly one controlled player: best marker blob that sits above a head
        and outside every player box (kit pixels of a similar hue are inside boxes)."""
        if not dets or ind.sum() < 3:
            return
        n, _, stats, cent = cv2.connectedComponentsWithStats(ind, connectivity=8)
        best, bscore = None, 1e9
        for i in range(1, n):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < 3:
                continue
            mx, my = cent[i] / s
            inside = any(abs(d.x - mx) < 0.45 * d.w and d.y - d.h * 0.9 < my < d.y for d in dets)
            if inside:
                continue
            for d in dets:
                top = d.y - d.h
                dy = top - my                    # marker sits above the head
                if -0.15 * d.h < dy < 1.0 * d.h and abs(d.x - mx) < 0.6 * d.w + 3 / s:
                    sc = abs(d.x - mx) / max(d.w, 1) + abs(dy / max(d.h, 1) - 0.25) - 0.02 * area
                    if sc < bscore:
                        best, bscore = d, sc
        if best is not None:
            best.controlled = True
            best.controlled_conf = 1.0

    def _excluded(self, c, width) -> bool:
        if not len(self.ball_exclude):
            return False
        return bool(np.min(np.hypot(self.ball_exclude[:, 0] - c[1], self.ball_exclude[:, 1] - c[2])) < 0.012 * width)

    @staticmethod
    def _on_body(x, y, dets) -> bool:
        """White blob on a player's shirt/shorts (not at the feet) -> not the ball."""
        for d in dets:
            if abs(x - d.x) < 0.5 * d.w and d.y - d.h < y < d.y - 0.22 * d.h:
                return True
        return False

    def _ball(self, frame, small, white, pitch, fg, s, prior, dets=()) -> Optional[T.Detection]:
        cands = []
        if prior is None and self.cfg.ball_full_res and not self.cfg.ball_global:
            return None
        if prior is None and self.cfg.ball_full_res:
            # (re-)acquisition: at working resolution thin line fragments look like a
            # ball; at full resolution they are clearly elongated
            c = self._ball_global_full_res(frame, pitch, s, dets)
            if c is not None and not self._excluded(c, frame.shape[1]):
                sc, x, y, sz = c
                return T.Detection(cls=T.CLS_BALL, x=float(x), y=float(y), w=sz, h=sz, conf=float(min(1.0, sc)))
            return None
        # tracking: working-resolution candidates + full-resolution search around the prior
        wm = cv2.bitwise_and(white, pitch)
        n, lbl, stats, cent = cv2.connectedComponentsWithStats(wm, connectivity=8)
        for i in range(1, n):
            x, y, bw, bh, a = stats[i]
            if a < 1 or a > 30 or max(bw, bh) > 7:
                continue
            sq = min(bw, bh) / max(bw, bh)
            fill = a / float(bw * bh)
            # lines continue beyond the blob: check whiteness of a surrounding ring
            x0, y0 = max(0, x - 3), max(0, y - 3)
            ring = wm[y0:y + bh + 3, x0:x + bw + 3]
            ring_white = (ring.sum() / 255.0 - a) / max(ring.size - a, 1)
            score = sq * fill * (1.0 - min(1.0, 4 * ring_white))
            if score > 0.15:
                cands.append((score, cent[i][0] / s, cent[i][1] / s, max(bw, bh) / s))
        if prior is not None and self.cfg.ball_full_res:
            c = self._ball_full_res(frame, prior)
            if c is not None:
                cands.append(c)
        cands = [c for c in cands if not self._on_body(c[1], c[2], dets)]
        cands = [c for c in cands if not self._excluded(c, frame.shape[1])]
        if not cands:
            return None
        if prior is not None:
            px, py = prior
            cands.sort(key=lambda c: -(c[0] * np.exp(-np.hypot(c[1] - px, c[2] - py) / (60.0 / max(s, 1e-3) * 0.25))))
        else:
            cands.sort(key=lambda c: -c[0])
        sc, x, y, sz = cands[0]
        return T.Detection(cls=T.CLS_BALL, x=float(x), y=float(y), w=sz, h=sz, conf=float(min(1.0, sc)))

    def _ball_global_full_res(self, frame, pitch_small, s, dets):
        H, W = frame.shape[:2]
        pm = cv2.resize(pitch_small, (W, H), interpolation=cv2.INTER_NEAREST)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        m = ((hsv[..., 1] < 60) & (hsv[..., 2] > 185)).astype(np.uint8)
        m = cv2.bitwise_and(m, m, mask=pm)
        n, lbl, stats, cent = cv2.connectedComponentsWithStats(m, connectivity=8)
        scale = W / 1280.0
        best = None
        for i in range(1, n):
            x, y, bw, bh, a = stats[i]
            if a < 4 * scale * scale or max(bw, bh) > 14 * scale or min(bw, bh) < 3 * scale:
                continue
            sq = min(bw, bh) / max(bw, bh)
            fill = a / float(bw * bh)
            if sq < 0.65 or fill < 0.55 or self._on_body(cent[i][0], cent[i][1], dets):
                continue
            # isolated: little white around it (lines continue, the ball does not)
            x0, y0 = max(0, x - 4), max(0, y - 4)
            ring = m[y0:y + bh + 4, x0:x + bw + 4]
            ring_white = (float(ring.sum()) - a) / max(ring.size - a, 1)
            score = sq * fill * (1.0 - min(1.0, 4 * ring_white))
            if score > 0.2 and (best is None or score > best[0]):
                best = (score, cent[i][0], cent[i][1], float(max(bw, bh)))
        return best

    def _ball_full_res(self, frame, prior, half=48):
        H, W = frame.shape[:2]
        px, py = int(prior[0]), int(prior[1])
        x0, y0, x1, y1 = max(0, px - half), max(0, py - half), min(W, px + half), min(H, py + half)
        if x1 - x0 < 8 or y1 - y0 < 8:
            return None
        roi = frame[y0:y1, x0:x1]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        m = ((hsv[..., 1] < 60) & (hsv[..., 2] > 185)).astype(np.uint8)
        n, lbl, stats, cent = cv2.connectedComponentsWithStats(m, connectivity=8)
        best = None
        scale = W / 1280.0
        for i in range(1, n):
            x, y, bw, bh, a = stats[i]
            if a < 3 * scale or max(bw, bh) > 16 * scale:
                continue
            sq = min(bw, bh) / max(bw, bh)
            fill = a / float(bw * bh)
            d = np.hypot(cent[i][0] + x0 - prior[0], cent[i][1] + y0 - prior[1])
            score = sq * fill * np.exp(-d / (half * 0.7))
            if best is None or score > best[0]:
                best = (score, cent[i][0] + x0, cent[i][1] + y0, float(max(bw, bh)))
        return best
