"""FC 27 radar reader (learned, shape-aware).

FC 27 draws its 2D radar as a dark, semi-transparent pitch panel at the
bottom centre of the screen.  One team is drawn as triangles, the other as
circles; fill and outline colours depend on kits and settings (filled,
hollow, white rings, ...).  Controlled players get a coloured ring or fill
(pink, yellow, red, ...).  The ball is an orange "+".  When the action is
behind the radar, or at set pieces, the panel disappears and the symbols are
drawn faint over the game.

Colour rules cannot cover all of that, so a tiny CNN (``TinyRadarNet``)
reads a canonical crop of the radar and outputs heatmaps for
triangle-team centres, circle-team centres, the ball and highlighted
(controlled) players.  It is trained on FC 27-style renders
(``fctac/sim/radar_fc27.py``) plus real FC 27 radar crops, see
``tools/train_radar.py``.

Geometry: the radar panel is the pitch (goal lines at its left/right edges,
touchlines at its top/bottom edges).  The canonical crop is 320x192 px with
the panel at (14.5, 16)-(305.5, 176), i.e. 1:1 with 1080p.
"""
from __future__ import annotations

import json
import os
from typing import Optional

import cv2
import numpy as np

from fctac.types import PITCH_LENGTH as L, PITCH_WIDTH as W, TEAM_THEM, TEAM_US

CW, CH = 320, 192                       # canonical crop size
PX0, PY0, PW, PH = 14.5, 16.0, 291.0, 160.0
STRIDE = 2                              # heatmap stride
CH_TRI, CH_CIR, CH_BALL, CH_HL = 0, 1, 2, 3
SHAPE_TRIANGLE, SHAPE_CIRCLE = 0, 1


def pitch_to_canon(xy: np.ndarray) -> np.ndarray:
    xy = np.asarray(xy, np.float64).reshape(-1, 2)
    return np.stack([PX0 + xy[:, 0] / L * PW, PY0 + (1.0 - xy[:, 1] / W) * PH], 1)


def canon_to_pitch(uv: np.ndarray) -> np.ndarray:
    uv = np.asarray(uv, np.float64).reshape(-1, 2)
    return np.stack([(uv[:, 0] - PX0) / PW * L, (1.0 - (uv[:, 1] - PY0) / PH) * W], 1)


def canon_affine(panel: tuple, w: int, h: int) -> np.ndarray:
    """2x3 affine mapping frame pixels -> canonical crop pixels."""
    x0, y0, x1, y1 = panel[0] * w, panel[1] * h, panel[2] * w, panel[3] * h
    sx, sy = PW / (x1 - x0), PH / (y1 - y0)
    return np.array([[sx, 0.0, PX0 - x0 * sx], [0.0, sy, PY0 - y0 * sy]])


def canonical_crop(frame: np.ndarray, panel: tuple) -> tuple[np.ndarray, np.ndarray]:
    """Canonical 320x192 radar crop of a full frame and the frame->crop affine."""
    h, w = frame.shape[:2]
    M = canon_affine(panel, w, h)
    sx, sy = M[0, 0], M[1, 1]
    # only warp the ROI (cheap); +2 px border for interpolation
    fx0 = max(0, int(np.floor((0 - M[0, 2]) / sx)) - 2)
    fy0 = max(0, int(np.floor((0 - M[1, 2]) / sy)) - 2)
    fx1 = min(w, int(np.ceil((CW - M[0, 2]) / sx)) + 2)
    fy1 = min(h, int(np.ceil((CH - M[1, 2]) / sy)) + 2)
    roi = frame[fy0:fy1, fx0:fx1]
    Mr = M.copy()
    Mr[0, 2] += fx0 * sx
    Mr[1, 2] += fy0 * sy
    interp = cv2.INTER_AREA if sx < 0.9 else cv2.INTER_LINEAR
    if interp == cv2.INTER_AREA:
        # INTER_AREA is only exact for pure resizes: resize then shift
        rs = cv2.resize(roi, None, fx=sx, fy=sy, interpolation=cv2.INTER_AREA)
        T = np.array([[1.0, 0.0, Mr[0, 2]], [0.0, 1.0, Mr[1, 2]]])
        crop = cv2.warpAffine(rs, T, (CW, CH), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    else:
        crop = cv2.warpAffine(roi, Mr, (CW, CH), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return crop, M


def harvest_crop_to_canonical(crop: np.ndarray, panel_in_crop: tuple) -> np.ndarray:
    """Canonical crop from a stored radar crop whose panel is at ``panel_in_crop`` (px)."""
    x0, y0, x1, y1 = panel_in_crop
    sx, sy = PW / (x1 - x0), PH / (y1 - y0)
    M = np.array([[sx, 0.0, PX0 - x0 * sx], [0.0, sy, PY0 - y0 * sy]])
    return cv2.warpAffine(crop, M, (CW, CH), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


# ----------------------------------------------------------------------------- decoding
def heat_peaks(hm: np.ndarray, thr: float, nms: int = 3) -> np.ndarray:
    """Local maxima above thr with quadratic sub-cell refinement -> (n, 3) [u, v, score] in heatmap cells."""
    if hm.max() < thr:
        return np.zeros((0, 3))
    k = np.ones((nms, nms), np.uint8)
    mx = cv2.dilate(hm, k)
    ys, xs = np.nonzero((hm >= mx) & (hm >= thr))
    out = []
    H, Wd = hm.shape
    for y, x in zip(ys, xs):
        s = hm[y, x]
        dx = dy = 0.0
        if 0 < x < Wd - 1:
            a, b = hm[y, x - 1], hm[y, x + 1]
            den = a - 2 * s + b
            dx = 0.5 * (a - b) / den if den < -1e-6 else 0.0
        if 0 < y < H - 1:
            a, b = hm[y - 1, x], hm[y + 1, x]
            den = a - 2 * s + b
            dy = 0.5 * (a - b) / den if den < -1e-6 else 0.0
        out.append((x + float(np.clip(dx, -0.5, 0.5)), y + float(np.clip(dy, -0.5, 0.5)), float(s)))
    return np.array(out)


def decode(heat: np.ndarray, thr: float = 0.35, ball_thr: float = 0.3, hl_thr: float = 0.35) -> dict:
    """heat: (4, CH/STRIDE, CW/STRIDE) sigmoid heatmaps -> symbols in canonical px."""
    tri = heat_peaks(heat[CH_TRI], thr)
    cir = heat_peaks(heat[CH_CIR], thr)
    # one symbol cannot be both shapes: suppress the weaker of coincident peaks
    if len(tri) and len(cir):
        d = np.hypot(tri[:, None, 0] - cir[None, :, 0], tri[:, None, 1] - cir[None, :, 1])
        kill_t = np.zeros(len(tri), bool)
        kill_c = np.zeros(len(cir), bool)
        for i, j in zip(*np.nonzero(d < 1.0)):
            if tri[i, 2] >= cir[j, 2]:
                kill_c[j] = True
            else:
                kill_t[i] = True
        tri, cir = tri[~kill_t], cir[~kill_c]
    pts = np.concatenate([tri, cir]) if len(tri) + len(cir) else np.zeros((0, 3))
    shape = np.array([SHAPE_TRIANGLE] * len(tri) + [SHAPE_CIRCLE] * len(cir), int)
    hl = np.zeros(len(pts))
    if len(pts):
        hm = heat[CH_HL]
        ix = np.clip(np.round(pts[:, 0]).astype(int), 0, hm.shape[1] - 1)
        iy = np.clip(np.round(pts[:, 1]).astype(int), 0, hm.shape[0] - 1)
        hl = np.maximum.reduce([hm[np.clip(iy + dy, 0, hm.shape[0] - 1), np.clip(ix + dx, 0, hm.shape[1] - 1)]
                                for dy in (-1, 0, 1) for dx in (-1, 0, 1)])
    b = heat_peaks(heat[CH_BALL], ball_thr)
    ball = None
    if len(b):
        k = int(np.argmax(b[:, 2]))
        ball = (np.array([b[k, 0], b[k, 1]]) + 0.5) * STRIDE - 0.5
    uv = (pts[:, :2] + 0.5) * STRIDE - 0.5 if len(pts) else np.zeros((0, 2))
    return {"uv": uv, "score": pts[:, 2] if len(pts) else np.zeros(0), "shape": shape,
            "highlight": hl, "ball_uv": ball, "ball_score": float(b[:, 2].max()) if len(b) else 0.0,
            "hl_thr": hl_thr}


# ----------------------------------------------------------------------------- runtime reader
class FC27RadarReader:
    """Drop-in replacement for ``RadarReader`` on real FC 27 footage.

    Team assignment: the radar shows shapes, not "us"/"them".  ``us_shape``
    ("triangle" / "circle") fixes it; "auto" picks the shape whose player is
    highlighted (the user's controlled player) when only one team shows a
    highlight -- the usual case against the CPU -- and otherwise keeps the
    last decision.  ``swap()`` flips it (live hotkey F7).
    """

    def __init__(self, cfg, model_path: Optional[str] = None, providers=None):
        self.cfg = cfg
        path = model_path or resolve_radar_model(getattr(cfg, "fc27_model", "registry"))
        if not path or not os.path.exists(path):
            raise FileNotFoundError(f"FC 27 radar model not found ({path}); train with tools/train_radar.py")
        from fctac.vision.learned import make_session
        self.sess = make_session(path, providers)
        self.inp = self.sess.get_inputs()[0].name
        meta_p = path + ".json"
        self.meta = json.load(open(meta_p)) if os.path.exists(meta_p) else {}
        self.thr = float(self.meta.get("thr", 0.35))
        self.dtype = np.float16 if "float16" in self.sess.get_inputs()[0].type else np.float32
        us = getattr(cfg, "us_shape", "auto")
        self.us_shape: Optional[int] = {"triangle": SHAPE_TRIANGLE, "circle": SHAPE_CIRCLE}.get(us)
        self.fixed = self.us_shape is not None
        self.votes = np.zeros(2)
        self.generation = 0           # bumps whenever the effective "your team" shape changes
        self.last = None
        # one-off auto-alignment of the panel position (HUD may sit a few px elsewhere)
        self.auto_align = bool(getattr(cfg, "auto_align", True))
        self.align_status = "pending" if self.auto_align else "off"
        self._align_rois: list = []
        self._calls = 0
        self._thread = None

    def effective_us(self) -> int:
        return self.us_shape if self.us_shape is not None else SHAPE_CIRCLE

    def swap(self):
        """F7: undecided -> triangles first; decided -> the other shape.  Locks the choice."""
        self.us_shape = SHAPE_TRIANGLE if self.us_shape is None else 1 - self.us_shape
        self.fixed = True
        self.generation += 1

    def infer(self, crop: np.ndarray) -> np.ndarray:
        x = crop[:, :, ::-1].transpose(2, 0, 1)[None].astype(self.dtype) / 255.0
        return self.sess.run(None, {self.inp: x})[0][0].astype(np.float32)

    def read_symbols(self, frame: np.ndarray) -> dict:
        crop, M = canonical_crop(frame, self.cfg.panel)
        d = decode(self.infer(crop), self.thr)
        d["crop"] = crop
        d["M"] = M
        return d

    @property
    def decided(self) -> bool:
        return self.us_shape is not None

    def _update_team(self, d: dict):
        """Your team = the shape whose controlled-player highlight clearly stands out.

        Evidence accumulates only on frames where one shape's strongest highlight beats the
        other's by a clear margin (vs the CPU only your team is highlighted).  The decision
        needs ~1 s of consistent evidence and is then locked for the session (no flip-flopping;
        F7 overrides).  If both teams keep showing highlights (online 1v1, co-op) it stays
        undecided and the assistant asks for F7 instead of guessing."""
        if self.fixed or self.us_shape is not None:
            return
        top = [float(d["highlight"][d["shape"] == s].max()) if np.any(d["shape"] == s) else 0.0
               for s in (SHAPE_TRIANGLE, SHAPE_CIRCLE)]
        if max(top) >= 0.35 and abs(top[0] - top[1]) >= 0.25:
            self.votes[int(np.argmax(top))] += 1.0
        a, b = self.votes.max(), self.votes.min()
        if a >= 30 and a >= 3.0 * b:
            self.us_shape = int(np.argmax(self.votes))
            self.generation += 1

    def _maybe_align(self, frame: np.ndarray, d: dict):
        """Collect ~10 radar-visible frames 0.5 s apart, then refine the panel in the background."""
        self._calls += 1
        if self.align_status != "pending" or len(d["uv"]) < 15 or self._calls % 15:
            return
        h, w = frame.shape[:2]
        x0, y0, x1, y1 = self.cfg.panel
        mx, my = 0.25 * (x1 - x0), 0.25 * (y1 - y0)
        r = (max(0, int((x0 - mx) * w)), max(0, int((y0 - my) * h)), min(w, int((x1 + mx) * w)), min(h, int((y1 + my) * h)))
        self._align_rois.append((r, frame[r[1]:r[3], r[0]:r[2]].copy(), (h, w)))
        if len(self._align_rois) >= 10:
            self.align_status = "running"
            import threading
            self._thread = threading.Thread(target=self._align, daemon=True, name="radar-align")
            self._thread.start()

    def _align(self):
        try:
            frames = []
            for (x0, y0, x1, y1), roi, (h, w) in self._align_rois:
                f = np.zeros((h, w, 3), np.uint8)
                f[y0:y1, x0:x1] = roi
                frames.append(f)
            panel, sc = estimate_panel(frames, tuple(self.cfg.panel))
            if sc >= 0.3:
                h, w = frames[0].shape[:2]
                shift = max(abs(a - b) * (w if i % 2 == 0 else h) for i, (a, b) in enumerate(zip(panel, self.cfg.panel)))
                self.cfg.panel = tuple(round(v, 5) for v in panel)
                self.align_status = f"aligned (moved {shift:.1f} px, score {sc:.2f})"
            else:
                self.align_status = f"kept configured position (score {sc:.2f})"
        except Exception as e:          # never break the live loop
            self.align_status = f"failed: {e}"
        self._align_rois = []

    def read(self, frame: np.ndarray, ball_prior: Optional[np.ndarray] = None):
        from fctac.vision.radar import RadarResult
        d = self.read_symbols(frame)
        self.last = d
        if self.auto_align:
            self._maybe_align(frame, d)
        h, w = frame.shape[:2]
        x0, y0 = self.cfg.panel[0] * w, self.cfg.panel[1] * h
        roi = (int(x0), int(y0), int(self.cfg.panel[2] * w), int(self.cfg.panel[3] * h))
        n = len(d["uv"])
        if n < max(6, getattr(self.cfg, "min_dots", 10)):
            return RadarResult(ok=False, roi_px=roi)
        self._update_team(d)
        pts = canon_to_pitch(d["uv"])
        ball = canon_to_pitch(d["ball_uv"])[0] if d["ball_uv"] is not None else None
        if self.cfg.flip:
            pts = np.array([L, W]) - pts
            if ball is not None:
                ball = np.array([L, W]) - ball
        teams = np.where(d["shape"] == self.effective_us(), TEAM_US, TEAM_THEM)
        # your team always has a controlled player: take the most highlighted symbol of your
        # team (top-1 is far more precise than a plain threshold); co-op: nearest the ball
        ctrl = None
        ours = np.nonzero(teams == TEAM_US)[0]
        thr = float(getattr(self.cfg, "highlight_thr", 0.25))
        if len(ours):
            cand = ours[d["highlight"][ours] >= max(thr, d["hl_thr"])]
            if len(cand) > 1 and ball is not None:
                ctrl = int(cand[np.argmin(np.hypot(*(pts[cand] - ball).T))])
            else:
                k = int(ours[np.argmax(d["highlight"][ours])])
                ctrl = k if d["highlight"][k] >= thr else None
        return RadarResult(ok=True, points=pts, teams=teams, controlled=ctrl, ball=ball, roi_px=roi)


def resolve_radar_model(spec: str) -> Optional[str]:
    if spec and spec != "registry":
        return spec
    try:
        from fctac.training.registry import Registry
        e = Registry().deployed("radar")
        return e["path"] if e else None
    except Exception:
        return None


# ----------------------------------------------------------------------------- panel auto-alignment
def _line_template(dx: float, dy: float, s: float) -> np.ndarray:
    from fctac.sim.radar_fc27 import SS, Geometry, draw_lines
    g = Geometry(dx, dy, s)
    m = np.zeros((CH * SS, CW * SS), np.uint8)
    draw_lines(m, g, SS)
    return cv2.resize(m, (CW, CH), interpolation=cv2.INTER_AREA).astype(np.float32)


def _lineness(crop: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return cv2.morphologyEx(g, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))


def estimate_panel(frames: list, panel: tuple, search_px: int = 12) -> tuple[tuple, float]:
    """Refine the radar panel rect from a few gameplay frames (radar visible).

    The median canonical crop keeps the static pitch lines and drops moving
    symbols; it is matched (normalised correlation of thin bright structures)
    against the FC 27 line layout under small shifts/scales.  Returns the
    corrected normalised panel and the match score (< 0.3: not found, keep
    the old panel).
    """
    crops = [canonical_crop(f, panel)[0] for f in frames]
    med = np.median(np.stack(crops), 0).astype(np.uint8)
    L_ = _lineness(med)
    L_ = (L_ - L_.mean()) / (L_.std() + 1e-6)
    best = (-1.0, 0.0, 0.0, 1.0)
    for s in (0.94, 0.97, 1.0, 1.03, 1.06):
        for dy in range(-search_px, search_px + 1, 2):
            for dx in range(-search_px, search_px + 1, 2):
                t = _line_template(dx, dy, s)
                t = (t - t.mean()) / (t.std() + 1e-6)
                sc = float((L_ * t).mean())
                if sc > best[0]:
                    best = (sc, dx, dy, s)
    sc, dx, dy, s = best
    for ddx in (-1, 0, 1):                       # 1 px refinement
        for ddy in (-1, 0, 1):
            t = _line_template(dx + ddx, dy + ddy, s)
            t = (t - t.mean()) / (t.std() + 1e-6)
            v = float((L_ * t).mean())
            if v > sc:
                sc, best = v, (v, dx + ddx, dy + ddy, s)
    _, dx, dy, s = best
    h, w = frames[0].shape[:2]
    x0, y0, x1, y1 = panel[0] * w, panel[1] * h, panel[2] * w, panel[3] * h
    kx, ky = (x1 - x0) / PW, (y1 - y0) / PH            # frame px per canonical px
    # canonical panel after the found transform (Geometry semantics)
    cx0 = PX0 + dx - (s - 1) * PW / 2
    cy0 = PY0 + dy - (s - 1) * PH / 2
    nx0 = x0 + (cx0 - PX0) * kx
    ny0 = y0 + (cy0 - PY0) * ky
    nx1 = nx0 + PW * s * kx
    ny1 = ny0 + PH * s * ky
    return (nx0 / w, ny0 / h, nx1 / w, ny1 / h), float(sc)
