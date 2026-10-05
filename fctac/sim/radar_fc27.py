"""FC 27-style radar renderer with exact labels (training data for the radar CNN).

Draws the radar the way FC 27 does -- a dark semi-transparent pitch panel
over the game, thin white pitch lines, one team as triangles and the other
as circles, highlighted (controlled) players, an orange "+" ball -- on top
of real main-view backgrounds, from simulator player positions.  Styles are
randomised widely (filled / hollow / ringed symbols, any colours, outline
widths, glow, faded radar without panel, no radar, blur, chroma subsampling,
JPEG) so the model learns shapes rather than one kit.  Geometry was measured
on real FC 27 footage (panel, halfway line, centre circle, boxes, goals and
the bar under the radar).

All coordinates are canonical crop pixels (see fctac/vision/radar_fc27.py).
"""
from __future__ import annotations

import glob
import os
from typing import Optional

import cv2
import numpy as np

from fctac.types import PITCH_LENGTH as L, PITCH_WIDTH as W
from fctac.vision.radar_fc27 import CH, CW, PH, PW, PX0, PY0, SHAPE_CIRCLE, SHAPE_TRIANGLE

SS = 4                     # supersampling for thin lines / anti-aliasing
SHIFT = 4                  # cv2 fractional bits
_F = 1 << SHIFT

# measured on the median FC 27 radar (fractions of panel width/height, centre-circle in panel heights)
HALF_V = (0.058, 0.946)
CIRCLE_R = 0.123
BOX_U, BOX_V = 0.130, (0.188, 0.812)
SPOT_U, ARC_R = 0.112, 0.130
SIXYD_U, SIXYD_V, BRACKET = 0.069, (0.287, 0.705), 6.5 / 290.75
GOAL_U, GOAL_V = -0.025, (0.385, 0.615)
BAR_V = 1.044

PALETTE = np.array([   # BGR: kit-like and FC indicator colours
    (215, 235, 245), (245, 245, 245), (190, 215, 235), (230, 205, 170), (235, 215, 235), (70, 25, 60),
    (30, 30, 30), (60, 60, 220), (220, 120, 40), (60, 190, 60), (40, 150, 245), (40, 220, 245),
    (200, 60, 150), (150, 150, 150), (120, 200, 255), (255, 220, 120), (90, 40, 170), (180, 120, 255),
], np.float64)
HIGHLIGHT = np.array([
    (255, 0, 255), (190, 110, 255), (40, 210, 255), (30, 150, 255), (50, 50, 235), (255, 130, 40),
    (255, 230, 60), (70, 230, 70), (210, 60, 160), (255, 255, 255), (0, 120, 255), (150, 70, 255),
], np.float64)


def _p(x, y):
    return int(round(x * SS * _F)), int(round(y * SS * _F))


def _rand_colour(rng) -> np.ndarray:
    if rng.random() < 0.7:
        c = PALETTE[rng.integers(len(PALETTE))] + rng.normal(0, 12, 3)
    else:
        hsv = np.uint8([[[rng.integers(0, 180), rng.integers(0, 256), rng.integers(40, 256)]]])
        c = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0].astype(np.float64)
    return np.clip(c, 0, 255)


class PositionPool:
    """Player/ball positions (raw pitch metres) from the match simulator."""

    def __init__(self, n_runs: int = 16, steps: int = 3000, every: int = 12, seed: int = 0):
        from fctac.sim.match import MatchSim, SimConfig
        from fctac.state.formations import FORMATIONS
        rng = np.random.default_rng(seed)
        forms = list(FORMATIONS)
        P, B, T = [], [], []
        for r in range(n_runs):
            cfg = SimConfig(seed=int(rng.integers(1 << 30)), us_attack_sign=int(rng.choice([-1, 1])),
                            formation_us=str(rng.choice(forms)), formation_them=str(rng.choice(forms)))
            sim = MatchSim(cfg)
            for k in range(steps):
                sim.step()
                if k % every == 0:
                    P.append(sim.pos.copy())
                    B.append(sim.ball.copy())
                    T.append(sim.team.copy())
        self.pos = np.array(P)
        self.ball = np.array(B)
        self.team = np.array(T)

    def sample(self, rng):
        k = int(rng.integers(len(self.pos)))
        return self.pos[k].copy(), self.ball[k].copy(), self.team[k].copy()

    def save(self, path):
        np.savez_compressed(path, pos=self.pos, ball=self.ball, team=self.team)

    @classmethod
    def load(cls, path):
        z = np.load(path)
        o = cls.__new__(cls)
        o.pos, o.ball, o.team = z["pos"], z["ball"], z["team"]
        return o


class BackgroundPool:
    """Random main-view patches from real frames (never the radar area), else synthetic grass."""

    def __init__(self, frames_glob: str = "", max_frames: int = 400, seed: int = 0, cache: int = 150,
                 files: Optional[list] = None):
        files = list(files) if files is not None else (sorted(glob.glob(frames_glob)) if frames_glob else [])
        rng = np.random.default_rng(seed)
        if len(files) > max_frames:
            files = list(rng.choice(files, max_frames, replace=False))
        self.files = files
        self.cache: dict = {}
        self.cache_n = cache

    def _frame(self, rng):
        f = self.files[int(rng.integers(len(self.files)))]
        im = self.cache.get(f)
        if im is None:
            im = cv2.imread(f)
            im = cv2.resize(im, (im.shape[1] // 2, im.shape[0] // 2), interpolation=cv2.INTER_AREA)
            if len(self.cache) >= self.cache_n:
                self.cache.pop(next(iter(self.cache)))
            self.cache[f] = im
        return im

    def sample(self, rng) -> np.ndarray:
        if not self.files:
            return synthetic_grass(rng)
        im = self._frame(rng)
        h, w = im.shape[:2]
        s = rng.uniform(0.35, 0.8)             # crop size relative to a 1080p crop (frames are half size)
        cw, chh = int(CW * s), int(CH * s)
        for _ in range(20):
            x = int(rng.integers(0, w - cw))
            y = int(rng.integers(int(0.15 * h), h - chh))
            # avoid the real radar (bottom centre) and the broadcast webcams (bottom corners)
            if y + chh > 0.76 * h and (x + cw > 0.38 * w and x < 0.62 * w):
                continue
            if y + chh > 0.66 * h and (x < 0.24 * w or x + cw > 0.76 * w):
                continue
            break
        patch = cv2.resize(im[y:y + chh, x:x + cw], (CW, CH), interpolation=cv2.INTER_LINEAR)
        if rng.random() < 0.5:
            patch = patch[:, ::-1]
        return np.ascontiguousarray(patch)


def synthetic_grass(rng) -> np.ndarray:
    base = np.array([40, 110, 80]) * rng.uniform(0.7, 1.2)
    img = np.ones((CH, CW, 3)) * base
    period = rng.uniform(25, 70)
    ang = rng.uniform(-0.4, 0.4)
    yy, xx = np.mgrid[0:CH, 0:CW]
    stripe = ((xx * np.cos(ang) + yy * np.sin(ang)) // period) % 2
    img *= (1 + 0.08 * stripe)[..., None]
    img += rng.normal(0, 4, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def _blend(img: np.ndarray, colour: np.ndarray, alpha: np.ndarray):
    """img float32 HxWx3 in place: img = img*(1-a) + colour*a (colour HxWx3 or 3)."""
    a = alpha[..., None]
    img *= 1.0 - a
    img += colour * a


def _down(mask_ss: np.ndarray) -> np.ndarray:
    return cv2.resize(mask_ss, (CW, CH), interpolation=cv2.INTER_AREA).astype(np.float32) * (1.0 / 255.0)


def _down_premult(col_ss: np.ndarray, mask_ss: np.ndarray):
    """Supersampled colour + coverage -> canonical colour and alpha (premultiplied resampling, uint8 ops)."""
    pre = cv2.multiply(col_ss, cv2.merge([mask_ss, mask_ss, mask_ss]), scale=1.0 / 255.0)
    pre_d = cv2.resize(pre, (CW, CH), interpolation=cv2.INTER_AREA).astype(np.float32)
    m_d = _down(mask_ss)
    col = pre_d / np.maximum(m_d[..., None], 1e-3)
    return np.minimum(col, 255.0), m_d


class Geometry:
    """Panel placement in canonical px (with optional jitter)."""

    def __init__(self, dx=0.0, dy=0.0, s=1.0):
        self.x0 = PX0 + dx - (s - 1) * PW / 2
        self.y0 = PY0 + dy - (s - 1) * PH / 2
        self.w = PW * s
        self.h = PH * s

    def uv(self, u, v):
        return self.x0 + u * self.w, self.y0 + v * self.h

    def pitch(self, xy: np.ndarray) -> np.ndarray:
        return np.stack([self.x0 + xy[:, 0] / L * self.w, self.y0 + (1 - xy[:, 1] / W) * self.h], 1)


def draw_lines(mask: np.ndarray, g: Geometry, th: int):
    def line(u0, v0, u1, v1):
        cv2.line(mask, _p(*g.uv(u0, v0)), _p(*g.uv(u1, v1)), 255, th, cv2.LINE_AA, SHIFT)

    line(0.5, HALF_V[0], 0.5, HALF_V[1])
    cx, cy = g.uv(0.5, 0.5)
    cv2.circle(mask, _p(cx, cy), int(round(CIRCLE_R * g.h * SS * _F)), 255, th, cv2.LINE_AA, SHIFT)
    for side in (0, 1):
        f = (lambda u: u) if side == 0 else (lambda u: 1 - u)
        line(f(0.0), BOX_V[0], f(BOX_U), BOX_V[0])
        line(f(BOX_U), BOX_V[0], f(BOX_U), BOX_V[1])
        line(f(BOX_U), BOX_V[1], f(0.0), BOX_V[1])
        # penalty arc outside the box
        sx, sy = g.uv(f(SPOT_U), 0.5)
        r = ARC_R * g.h
        bx = g.uv(f(BOX_U), 0.5)[0]
        dxb = abs(bx - sx)
        if r > dxb:
            half = np.degrees(np.arccos(dxb / r))
            a0 = 0 if side == 0 else 180
            cv2.ellipse(mask, _p(sx, sy), (int(r * SS * _F), int(r * SS * _F)), 0, a0 - half, a0 + half, 255, th,
                        cv2.LINE_AA, SHIFT)
        # six-yard box corner brackets
        b = BRACKET
        for v, dv in ((SIXYD_V[0], b * g.w / g.h), (SIXYD_V[1], -b * g.w / g.h)):
            u1 = f(SIXYD_U)
            u0 = f(SIXYD_U - b)
            line(u0, v, u1, v)
            line(u1, v, u1, v + dv)


def draw_goals_bar(mask: np.ndarray, g: Geometry, th: int, rng):
    for u in (GOAL_U, 1 - GOAL_U):
        cv2.line(mask, _p(*g.uv(u, GOAL_V[0])), _p(*g.uv(u, GOAL_V[1])), 255, th, cv2.LINE_AA, SHIFT)
    if rng.random() < 0.85:
        y = BAR_V
        cuts = np.sort(rng.uniform(0, 1, int(rng.integers(0, 8))))
        xs = [0.0] + list(cuts) + [1.015]
        for k in range(len(xs) - 1):
            if k % 2 == 0:
                cv2.line(mask, _p(*g.uv(xs[k], y)), _p(*g.uv(xs[k + 1], y)), 255, th, cv2.LINE_AA, SHIFT)


def _tri_pts(cx, cy, t):
    h = t * np.sqrt(3) / 2
    return np.array([_p(cx, cy - h / 2), _p(cx - t / 2, cy + h / 2), _p(cx + t / 2, cy + h / 2)], np.int32)


def draw_symbol(col, mask, shape, cx, cy, size, fill, outline, ow):
    """fill/outline: BGR or None; ow outline width (canonical px)."""
    th = max(1, int(round(ow * SS)))
    if shape == SHAPE_TRIANGLE:
        pts = _tri_pts(cx, cy, size)
        if fill is not None:
            cv2.fillPoly(col, [pts], tuple(float(c) for c in fill), cv2.LINE_AA, SHIFT)
            cv2.fillPoly(mask, [pts], 255, cv2.LINE_AA, SHIFT)
        if outline is not None:
            cv2.polylines(col, [pts], True, tuple(float(c) for c in outline), th, cv2.LINE_AA, SHIFT)
            cv2.polylines(mask, [pts], True, 255, th, cv2.LINE_AA, SHIFT)
    else:
        r = int(round(size / 2 * SS * _F))
        c = _p(cx, cy)
        if fill is not None:
            cv2.circle(col, c, r, tuple(float(v) for v in fill), -1, cv2.LINE_AA, SHIFT)
            cv2.circle(mask, c, r, 255, -1, cv2.LINE_AA, SHIFT)
        if outline is not None:
            cv2.circle(col, c, r, tuple(float(v) for v in outline), th, cv2.LINE_AA, SHIFT)
            cv2.circle(mask, c, r, 255, th, cv2.LINE_AA, SHIFT)


def draw_ball(col, mask, cx, cy, size, th, fill, edge):
    a = size / 2
    t = th / 2
    for (x0, y0, x1, y1) in ((cx - a, cy - t, cx + a, cy + t), (cx - t, cy - a, cx + t, cy + a)):
        pts = np.array([_p(x0 - 0.6, y0 - 0.6), _p(x1 + 0.6, y0 - 0.6), _p(x1 + 0.6, y1 + 0.6), _p(x0 - 0.6, y1 + 0.6)],
                       np.int32)
        cv2.fillPoly(col, [pts], tuple(float(c) for c in edge), cv2.LINE_AA, SHIFT)
        cv2.fillPoly(mask, [pts], 255, cv2.LINE_AA, SHIFT)
    for (x0, y0, x1, y1) in ((cx - a, cy - t, cx + a, cy + t), (cx - t, cy - a, cx + t, cy + a)):
        pts = np.array([_p(x0, y0), _p(x1, y0), _p(x1, y1), _p(x0, y1)], np.int32)
        cv2.fillPoly(col, [pts], tuple(float(c) for c in fill), cv2.LINE_AA, SHIFT)


def team_style(rng) -> dict:
    kind = rng.choice(["filled", "hollow", "filled_outline"], p=[0.3, 0.3, 0.4])
    fill = _rand_colour(rng)
    r = rng.random()
    if r < 0.45:
        outline = np.array([245, 245, 245.0]) + rng.normal(0, 6, 3)
    elif r < 0.65:
        outline = fill * rng.uniform(0.4, 0.8)
    elif r < 0.8:
        outline = np.array([25, 25, 25.0])
    else:
        outline = _rand_colour(rng)
    if kind == "hollow":
        return {"fill": None, "outline": np.clip(fill if rng.random() < 0.6 else outline, 0, 255),
                "ow": rng.uniform(1.1, 2.0)}
    if kind == "filled":
        return {"fill": fill, "outline": None, "ow": 1.0}
    return {"fill": fill, "outline": np.clip(outline, 0, 255), "ow": rng.uniform(0.8, 1.6)}


def highlight_style(rng, base: dict) -> dict:
    hc = HIGHLIGHT[int(rng.integers(len(HIGHLIGHT)))] + rng.normal(0, 10, 3)
    hc = np.clip(hc, 0, 255)
    k = rng.choice(["outline", "fill", "ring"], p=[0.4, 0.35, 0.25])
    if k == "outline":
        return {"fill": base["fill"], "outline": hc, "ow": max(base["ow"], 1.0) * rng.uniform(1.1, 1.6), "glow": hc}
    if k == "fill":
        return {"fill": hc, "outline": base["outline"], "ow": base["ow"], "glow": hc}
    return {"fill": None, "outline": hc, "ow": rng.uniform(1.3, 2.2), "glow": hc}


def realism(img: np.ndarray, rng) -> np.ndarray:
    if rng.random() < 0.8:
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.25, 0.75))
    if rng.random() < 0.7:                                   # 4:2:0 chroma subsampling (video)
        ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        c = cv2.resize(ycc[..., 1:], (CW // 2, CH // 2), interpolation=cv2.INTER_AREA)
        ycc[..., 1:] = cv2.resize(c, (CW, CH), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)
    if rng.random() < 0.5:
        g = rng.uniform(0.85, 1.15)
        b = rng.uniform(-12, 12)
        img = np.clip(img.astype(np.float32) * g + b, 0, 255).astype(np.uint8)
    if rng.random() < 0.4:
        img = np.clip(img.astype(np.float32) + rng.normal(0, rng.uniform(1, 4), img.shape), 0, 255).astype(np.uint8)
    if rng.random() < 0.8:
        q = int(rng.integers(35, 92))
        img = cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])[1], cv2.IMREAD_COLOR)
    return img


def render(rng, positions: PositionPool, backgrounds: BackgroundPool, mode: Optional[str] = None):
    """One sample -> (BGR uint8 CHxCW image, labels dict with canonical px)."""
    mode = mode or rng.choice(["panel", "faded", "transition", "none"], p=[0.68, 0.14, 0.08, 0.10])
    img = backgrounds.sample(rng).astype(np.float32)
    lab = {"uv": np.zeros((0, 2)), "shape": np.zeros(0, int), "highlight": np.zeros(0, bool), "ball": None,
           "mode": mode}
    if mode == "none":
        return realism(np.clip(img, 0, 255).astype(np.uint8), rng), lab
    g = Geometry(rng.normal(0, 1.0), rng.normal(0, 1.0), rng.uniform(0.97, 1.03))
    fade = 1.0 if mode == "panel" else (rng.uniform(0.3, 0.85) if mode == "transition" else 1.0)
    # panel
    if mode in ("panel", "transition"):
        pm = np.zeros((CH * SS, CW * SS), np.uint8)
        rad = rng.uniform(2, 4) * SS
        x0, y0 = g.x0 * SS, g.y0 * SS
        x1, y1 = (g.x0 + g.w) * SS, (g.y0 + g.h) * SS
        cv2.rectangle(pm, (int(x0 + rad), int(y0)), (int(x1 - rad), int(y1)), 255, -1)
        cv2.rectangle(pm, (int(x0), int(y0 + rad)), (int(x1), int(y1 - rad)), 255, -1)
        for cx, cy in ((x0 + rad, y0 + rad), (x1 - rad, y0 + rad), (x0 + rad, y1 - rad), (x1 - rad, y1 - rad)):
            cv2.circle(pm, (int(cx), int(cy)), int(rad), 255, -1, cv2.LINE_AA)
        a = _down(pm) * rng.uniform(0.32, 0.5) * fade
        img *= (1.0 - a)[..., None]
        line_a = rng.uniform(0.28, 0.55) * fade
    else:
        line_a = rng.uniform(0.0, 0.25)
    lines = np.zeros((CH * SS, CW * SS), np.uint8)
    th = max(1, int(round(rng.uniform(0.85, 1.25) * SS)))
    draw_lines(lines, g, th)
    goals = np.zeros_like(lines)
    draw_goals_bar(goals, g, th, rng)
    white = np.array([245, 250, 245], np.float32)
    _blend(img, white, _down(lines) * line_a)
    _blend(img, white, _down(goals) * min(1.0, line_a * rng.uniform(1.3, 1.8)))
    # players
    pos, ball, team = positions.sample(rng)
    if rng.random() < 0.5:
        pos[:, 0], ball[0] = L - pos[:, 0], L - ball[0]
    if rng.random() < 0.5:
        pos[:, 1], ball[1] = W - pos[:, 1], W - ball[1]
    pos += rng.normal(0, 0.6, pos.shape)
    if rng.random() < 0.15:                                  # break formation priors
        k = rng.random(len(pos)) < 0.4
        pos[k] = rng.uniform([-1, -1], [L + 1, W + 1], (int(k.sum()), 2))
    if rng.random() < 0.08:                                  # crowded duel / set-piece cluster
        c = rng.uniform([5, 5], [L - 5, W - 5])
        k = rng.random(len(pos)) < 0.5
        pos[k] = c + rng.normal(0, 3.0, (int(k.sum()), 2))
    shape_of_team = rng.permutation([SHAPE_TRIANGLE, SHAPE_CIRCLE])
    shapes = shape_of_team[team]
    uv = g.pitch(pos)
    styles = {s: team_style(rng) for s in (SHAPE_TRIANGLE, SHAPE_CIRCLE)}
    if rng.random() < 0.3:                                   # both teams same colours (shape is the only cue)
        styles[SHAPE_CIRCLE] = dict(styles[SHAPE_TRIANGLE])
    sizes = {SHAPE_TRIANGLE: rng.uniform(9.0, 12.0), SHAPE_CIRCLE: rng.uniform(8.0, 10.5)}
    hl = np.zeros(len(pos), bool)
    bpos_uv = g.pitch(ball[None])[0]
    for s in (SHAPE_TRIANGLE, SHAPE_CIRCLE):
        idx = np.nonzero(shapes == s)[0]
        n_hl = rng.choice([0, 1, 2], p=[0.3, 0.55, 0.15])
        if n_hl:
            d = np.hypot(*(uv[idx] - bpos_uv).T)
            first = idx[np.argmin(d)] if rng.random() < 0.7 else rng.choice(idx)
            hl[first] = True
            if n_hl == 2:
                hl[rng.choice(idx)] = True
    sym_a = 1.0 if mode == "panel" else (rng.uniform(0.3, 0.65) if mode == "faded" else fade)
    order = rng.permutation(len(pos))
    col = np.zeros((CH * SS, CW * SS, 3), np.uint8)
    msk = np.zeros((CH * SS, CW * SS), np.uint8)
    gcol = np.zeros_like(col)
    gmsk = np.zeros_like(msk)
    hl_styles = {}
    for i in order:
        s = int(shapes[i])
        st = styles[s]
        if hl[i]:
            st = hl_styles.setdefault((s, i), highlight_style(rng, styles[s]))
            if rng.random() < 0.6:
                draw_symbol(gcol, gmsk, s, uv[i, 0], uv[i, 1], sizes[s] * 1.5, st["glow"], None, 1)
        elif styles[s]["fill"] is not None and rng.random() < 0.1:
            draw_symbol(gcol, gmsk, s, uv[i, 0], uv[i, 1], sizes[s] * 1.4, styles[s]["fill"], None, 1)
        draw_symbol(col, msk, s, uv[i, 0], uv[i, 1], sizes[s], st["fill"], st["outline"], st["ow"])
    if gmsk.any():
        gm = cv2.GaussianBlur(gmsk, (0, 0), 1.6 * SS)
        gc, ga = _down_premult(gcol, gm)
        _blend(img, gc, ga * rng.uniform(0.25, 0.6) * sym_a)
    c, a = _down_premult(col, msk)
    _blend(img, c, a * sym_a)
    # ball
    has_ball = rng.random() < 0.95
    if has_ball:
        if rng.random() < 0.05:
            ball = rng.uniform([-3, -3], [L + 3, W + 3])
            bpos_uv = g.pitch(ball[None])[0]
        bc = np.zeros((CH * SS, CW * SS, 3), np.uint8)
        bm = np.zeros((CH * SS, CW * SS), np.uint8)
        fill = np.clip(np.array([20, 185, 250.0]) + rng.normal(0, [10, 18, 6]), 0, 255)
        draw_ball(bc, bm, bpos_uv[0], bpos_uv[1], rng.uniform(9.0, 12.0), rng.uniform(2.4, 3.6), fill,
                  fill * rng.uniform(0.3, 0.6))
        c, a = _down_premult(bc, bm)
        _blend(img, c, a * sym_a)
    out = realism(np.clip(img, 0, 255).astype(np.uint8), rng)
    inside = (uv[:, 0] > -3) & (uv[:, 0] < CW + 3) & (uv[:, 1] > -3) & (uv[:, 1] < CH + 3)
    lab.update(uv=uv[inside], shape=shapes[inside].astype(int), highlight=hl[inside],
               ball=bpos_uv if has_ball and -3 < bpos_uv[0] < CW + 3 and -3 < bpos_uv[1] < CH + 3 else None)
    return out, lab


def default_backgrounds(root: str = "data/real/harvest", groups: Optional[tuple] = None, seed: int = 0,
                        max_frames: int = 400) -> BackgroundPool:
    """Gameplay frames (broadcast camera, mostly grass) from a harvest index; 10 % any frame."""
    import csv
    idx = os.path.join(root, "index.csv")
    if not os.path.exists(idx):
        return BackgroundPool("")
    rows = [r for r in csv.DictReader(open(idx)) if r["full"] == "1" and (groups is None or r["group"] in groups)]
    rng = np.random.default_rng(seed)
    game = [r for r in rows if float(r["green"]) > 0.55]
    other = [r for r in rows if float(r["green"]) <= 0.55]
    pick = list(rng.choice(game, min(len(game), int(max_frames * 0.9)), replace=False)) if game else []
    pick += list(rng.choice(other, min(len(other), max_frames - len(pick)), replace=False)) if other else []
    files = [os.path.join(root, "frames", f"{r['clip']}_{int(r['frame']):05d}.jpg") for r in pick]
    return BackgroundPool(files=files, max_frames=max_frames, seed=seed)
