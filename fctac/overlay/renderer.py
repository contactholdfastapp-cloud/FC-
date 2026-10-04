"""Minimal recommendation overlay.

Draws into a transparent BGRA canvas whose colour channels are premultiplied
by alpha (what Win32 ``UpdateLayeredWindow`` expects).  The same canvas is
alpha-composited onto video frames in replay mode, so replay shows exactly
what the live overlay would show.

Visual language (kept deliberately small):
  ACCENT  (green-cyan) recommended action
  SECOND  (yellow)     secondary / possible action
  DANGER  (red)        defensive threat
  NEUTRAL (white/grey) informational, ANALYSING
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from fctac.pitch.camera import apply_h
from fctac import types as T

ACCENT = (170, 255, 30)
SECOND = (0, 215, 255)
DANGER = (70, 70, 255)
NEUTRAL = (235, 235, 235)
SHADOW = (0, 0, 0)

FONT = cv2.FONT_HERSHEY_DUPLEX
SH = 4                      # sub-pixel shift for smooth AA drawing
S = 1 << SH


@dataclass
class OverlayConfig:
    show_secondary: bool = False
    show_confidence: bool = False
    show_debug: bool = False
    show_analysing: bool = True
    label_anchor: str = "player"     # "player" (above controlled player) or "top"
    scale: float = 1.0               # UI scale (1.0 tuned for 1280x720; auto x1.5 at 1080p)


def _pt(p) -> tuple[int, int]:
    return int(round(float(p[0]) * S)), int(round(float(p[1]) * S))


def _rgba(c, a=255):
    k = a / 255.0
    return (int(c[0] * k), int(c[1] * k), int(c[2] * k), int(a))


class Canvas:
    """Premultiplied BGRA drawing surface with dirty-rect tracking."""

    def __init__(self, width: int, height: int):
        self.w, self.h = width, height
        self.img = np.zeros((height, width, 4), np.uint8)
        self.dirty: Optional[list] = None

    def clear(self):
        if self.dirty is not None:
            x0, y0, x1, y1 = self.dirty
            self.img[y0:y1, x0:x1] = 0
        self.dirty = None

    def mark(self, x0, y0, x1, y1, pad=8):
        x0, y0 = max(0, int(x0) - pad), max(0, int(y0) - pad)
        x1, y1 = min(self.w, int(x1) + pad), min(self.h, int(y1) + pad)
        if x1 <= x0 or y1 <= y0:
            return
        if self.dirty is None:
            self.dirty = [x0, y0, x1, y1]
        else:
            d = self.dirty
            self.dirty = [min(d[0], x0), min(d[1], y0), max(d[2], x1), max(d[3], y1)]

    def mark_pts(self, pts, pad=10):
        pts = np.asarray(pts, float).reshape(-1, 2)
        self.mark(pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max(), pad)

    # --- primitives (all colours BGR, alpha separately) -----------------------
    def line(self, a, b, color, th, alpha=255, outline=True):
        if outline:
            cv2.line(self.img, _pt(a), _pt(b), _rgba(SHADOW, int(alpha * 0.55)), th + 3, cv2.LINE_AA, SH)
        cv2.line(self.img, _pt(a), _pt(b), _rgba(color, alpha), th, cv2.LINE_AA, SH)
        self.mark_pts([a, b], th + 6)

    def polyline(self, pts, color, th, alpha=255, outline=True, closed=False):
        p = np.array([_pt(q) for q in pts], np.int32)
        if outline:
            cv2.polylines(self.img, [p], closed, _rgba(SHADOW, int(alpha * 0.55)), th + 3, cv2.LINE_AA, SH)
        cv2.polylines(self.img, [p], closed, _rgba(color, alpha), th, cv2.LINE_AA, SH)
        self.mark_pts(pts, th + 6)

    def dashed(self, pts, color, th, alpha=255, dash=10.0, gap=8.0):
        pts = np.asarray(pts, float)
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        cum = np.concatenate([[0], np.cumsum(seg)])
        s = 0.0
        while s < cum[-1]:
            e = min(s + dash, cum[-1])
            ts = np.linspace(s, e, 4)
            q = np.stack([np.interp(ts, cum, pts[:, 0]), np.interp(ts, cum, pts[:, 1])], 1)
            self.polyline(q, color, th, alpha, outline=False)
            s = e + gap

    def arrow_head(self, tip, direction, size, color, alpha=255):
        d = np.asarray(direction, float)
        n = np.linalg.norm(d)
        if n < 1e-6:
            return
        d /= n
        nrm = np.array([-d[1], d[0]])
        tip = np.asarray(tip, float)
        base = tip - d * size
        tri = np.array([tip, base + nrm * size * 0.55, base - nrm * size * 0.55])
        p = np.array([_pt(q) for q in tri], np.int32)
        cv2.fillConvexPoly(self.img, p, _rgba(SHADOW, int(alpha * 0.55)), cv2.LINE_AA, SH)
        shrink = tri.mean(0) + (tri - tri.mean(0)) * 0.78
        cv2.fillConvexPoly(self.img, np.array([_pt(q) for q in shrink], np.int32), _rgba(color, alpha), cv2.LINE_AA, SH)
        self.mark_pts(tri, 4)

    def ring(self, pts, color, th, alpha=255):
        self.polyline(pts, color, th, alpha, outline=True, closed=True)

    def rect(self, x0, y0, x1, y1, color, alpha):
        x0, y0, x1, y1 = (int(max(0, x0)), int(max(0, y0)), int(min(self.w, x1)), int(min(self.h, y1)))
        if x1 <= x0 or y1 <= y0:
            return
        roi = self.img[y0:y1, x0:x1]
        a = alpha / 255.0
        roi[:] = (roi.astype(np.float32) * (1 - a) + np.array(_rgba(color, alpha), np.float32)).astype(np.uint8)
        self.mark(x0, y0, x1, y1, 0)

    def text(self, s, org, scale, color, th=1, alpha=255, outline=True):
        x, y = int(org[0]), int(org[1])
        if outline:
            cv2.putText(self.img, s, (x, y), FONT, scale, _rgba(SHADOW, int(alpha * 0.7)), th + 3, cv2.LINE_AA)
        cv2.putText(self.img, s, (x, y), FONT, scale, _rgba(color, alpha), th, cv2.LINE_AA)
        (tw, tht), base = cv2.getTextSize(s, FONT, scale, th + 3)
        self.mark(x, y - tht, x + tw, y + base, 4)
        return tw

    # --- compositing ---------------------------------------------------------
    def composite_onto(self, frame: np.ndarray) -> np.ndarray:
        if self.dirty is None:
            return frame
        x0, y0, x1, y1 = self.dirty
        src = self.img[y0:y1, x0:x1]
        dst = frame[y0:y1, x0:x1]
        a = src[..., 3:4].astype(np.uint16)
        out = (dst.astype(np.uint16) * (255 - a) + 127) // 255 + src[..., :3]
        frame[y0:y1, x0:x1] = np.minimum(out, 255).astype(np.uint8)
        return frame


def _label_size(parts, scale, th):
    w = 0
    for kind, val in parts:
        if kind == "text":
            (tw, _), _ = cv2.getTextSize(val, FONT, scale, th)
            w += tw
        else:
            w += int(26 * scale)
        w += int(8 * scale)
    return w - int(8 * scale)


class OverlayRenderer:
    def __init__(self, width: int, height: int, cfg: OverlayConfig = OverlayConfig()):
        self.cfg = cfg
        self.canvas = Canvas(width, height)
        self.k = cfg.scale * (height / 720.0)

    # --- projection helpers ---------------------------------------------------
    @staticmethod
    def _H_att2img(state: T.GameState) -> Optional[np.ndarray]:
        if state is None or state.H_img2pitch is None:
            return None
        Hraw = np.linalg.inv(state.H_img2pitch)
        if state.attack_sign > 0:
            return Hraw
        F = np.array([[-1, 0, T.PITCH_LENGTH], [0, -1, T.PITCH_WIDTH], [0, 0, 1.0]])
        return Hraw @ F

    def _ground_ring(self, H, center, radius_m, n=28):
        a = np.linspace(0, 2 * np.pi, n, endpoint=False)
        pts = np.asarray(center, float) + radius_m * np.stack([np.cos(a), np.sin(a)], 1)
        return apply_h(H, pts)

    def _player_screen(self, H, p: T.PlayerState):
        if p.screen is not None:
            return np.asarray(p.screen, float)
        return apply_h(H, p.pos)

    # --- main -------------------------------------------------------------------
    def render(self, fa: T.FrameAnalysis, debug_lines: Optional[list] = None) -> Canvas:
        c = self.canvas
        c.clear()
        rec = fa.recommendation
        st = fa.state
        H = self._H_att2img(st)
        if rec.status == T.STATUS_ACTIVE and rec.action is not None and H is not None and st.controlled is not None:
            if self.cfg.show_secondary and rec.alternatives:
                self._draw_action(st, H, rec.alternatives[0], secondary=True)
            self._draw_action(st, H, rec.action, secondary=False)
            self._draw_label(st, H, rec)
        elif rec.status == T.STATUS_ANALYSING and self.cfg.show_analysing:
            k = self.k
            c.text("ANALYSING", (c.w // 2 - int(60 * k), int(40 * k)), 0.6 * k, NEUTRAL, 1, 170)
        if self.cfg.show_debug and debug_lines:
            self._draw_debug(debug_lines)
        return c

    def _draw_action(self, st, H, a: T.Action, secondary: bool):
        c = self.canvas
        k = self.k
        col = SECOND if secondary else ACCENT
        alpha = 150 if secondary else 255
        th = max(1, int((2 if secondary else 4) * k))
        me = st.controlled
        p0 = self._player_screen(H, me)
        tgt_player = st.player(a.target_id)
        if a.kind in (T.PASS, T.LOB, T.CROSS, T.THROUGH):
            if a.kind == T.THROUGH and a.target_point is not None:
                end = apply_h(H, a.target_point)
            elif tgt_player is not None:
                end = self._player_screen(H, tgt_player)
            elif a.target_point is not None:
                end = apply_h(H, a.target_point)
            else:
                return
            lofted = a.kind in (T.LOB, T.CROSS)
            self._arrow(p0, end, col, th, alpha, curve=0.18 if lofted else (0.08 if a.kind == T.THROUGH else 0.0))
            if tgt_player is not None and not secondary:
                ring = self._ground_ring(H, tgt_player.pos, 1.1)
                c.ring(ring, col, max(1, int(2 * k)), 220)
                if a.kind == T.THROUGH and a.target_point is not None:
                    rp = self._player_screen(H, tgt_player)
                    c.dashed([rp, end], NEUTRAL, max(1, int(1.5 * k)), 170, dash=8 * k, gap=6 * k)
            if a.kind == T.THROUGH and a.target_point is not None and not secondary:
                ring = self._ground_ring(H, a.target_point, 1.4)
                c.ring(ring, col, max(1, int(2 * k)), 255)
        elif a.kind == T.SHOOT and a.target_point is not None:
            end = apply_h(H, a.target_point)
            self._arrow(p0, end, col, th, alpha, curve=0.0)
            r = max(4, int(7 * k))
            c.ring([end + r * np.array([np.cos(t), np.sin(t)]) for t in np.linspace(0, 6.28, 16, endpoint=False)],
                   col, max(1, int(2 * k)), 255)
        elif a.kind == T.DRIBBLE and a.target_point is not None:
            end = apply_h(H, a.target_point)
            self._arrow(p0, end, col, max(1, int(3 * k)), alpha, curve=0.0, dashed=True)
        elif a.kind in (T.PRESS, T.JOCKEY) and tgt_player is not None:
            end = self._player_screen(H, tgt_player)
            self._arrow(p0, end, col, th, alpha, curve=0.0, dashed=a.kind == T.JOCKEY)
            c.ring(self._ground_ring(H, tgt_player.pos, 1.2), DANGER, max(1, int(2 * k)), 230)
        elif a.kind == T.SWITCH and tgt_player is not None:
            c.ring(self._ground_ring(H, tgt_player.pos, 1.3), col, max(2, int(3 * k)), 255)
        elif a.kind == T.COVER and tgt_player is not None:
            c.ring(self._ground_ring(H, tgt_player.pos, 1.3), DANGER, max(1, int(2 * k)), 230)
            if a.target_point is not None:
                end = apply_h(H, a.target_point)
                self._arrow(p0, end, col, max(1, int(3 * k)), alpha, curve=0.0, dashed=True)

    def _arrow(self, p0, p1, col, th, alpha, curve=0.0, dashed=False):
        p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
        d = p1 - p0
        n = np.linalg.norm(d)
        if n < 6:
            return
        k = self.k
        start = p0 + d / n * min(14 * k, n * 0.2)
        head = max(10.0, 16 * k)
        nrm = np.array([-d[1], d[0]]) / n
        ctrl = (start + p1) / 2 + nrm * curve * n
        ts = np.linspace(0, 1, 24)[:, None]
        pts = (1 - ts) ** 2 * start + 2 * (1 - ts) * ts * ctrl + ts ** 2 * p1
        # stop the shaft at the arrow head base
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        cum = np.concatenate([[0], np.cumsum(seg)])
        keep = cum <= max(cum[-1] - head * 0.8, 1)
        shaft = pts[keep]
        if dashed:
            self.canvas.dashed(shaft, col, th, alpha, dash=12 * k, gap=8 * k)
        else:
            self.canvas.polyline(shaft, col, th, alpha)
        self.canvas.arrow_head(p1, p1 - pts[-3], head, col, alpha)

    # --- label -----------------------------------------------------------------
    def _label_parts(self, st, H, a: T.Action):
        parts = []
        verb = a.kind
        if a.kind == T.SHOOT:
            parts.append(("text", "SHOOT"))
            me = self._player_screen(H, st.controlled)
            if a.target_point is not None:
                d = apply_h(H, a.target_point) - me
                parts.append(("arrow", float(np.arctan2(d[1], d[0]))))
            return parts
        parts.append(("text", verb))
        if a.kind in (T.PASS, T.THROUGH, T.LOB, T.CROSS, T.SWITCH) and a.target_label:
            parts.append(("arrow", 0.0))
            parts.append(("text", a.target_label))
        elif a.kind == T.COVER and a.target_label:
            parts.append(("text", a.target_label))
        return parts

    def _draw_label(self, st, H, rec: T.Recommendation):
        c = self.canvas
        k = self.k
        a = rec.action
        scale, th = 0.75 * k, max(1, int(2 * k))
        parts = self._label_parts(st, H, a)
        w = _label_size(parts, scale, th)
        hgt = int(30 * k)
        if self.cfg.label_anchor == "top":
            x0, y0 = c.w // 2 - w // 2, int(18 * k)
        else:
            me = self._player_screen(H, st.controlled)
            x0 = int(me[0] - w / 2)
            y0 = int(me[1] - 120 * k)
        x0 = int(np.clip(x0, 6, c.w - w - 18))
        y0 = int(np.clip(y0, 6, c.h - hgt - 40 * k))
        pad = int(9 * k)
        c.rect(x0 - pad, y0, x0 + w + pad, y0 + hgt, (10, 10, 10), 150)
        bar = ACCENT if a.kind not in T.DEFENCE_KINDS or a.kind == T.SWITCH else ACCENT
        c.rect(x0 - pad, y0 + hgt - max(2, int(3 * k)), x0 + w + pad, y0 + hgt, bar, 255)
        x = x0
        base = y0 + int(22 * k)
        for kind, val in parts:
            if kind == "text":
                x += c.text(val, (x, base), scale, NEUTRAL, th, 255, outline=False)
            else:
                cx, cy = x + 13 * k, y0 + hgt * 0.45
                d = np.array([np.cos(val), np.sin(val)])
                c.line((cx - d[0] * 10 * k, cy - d[1] * 10 * k), (cx + d[0] * 4 * k, cy + d[1] * 4 * k),
                       ACCENT, max(2, int(3 * k)), outline=False)
                c.arrow_head((cx + d[0] * 11 * k, cy + d[1] * 11 * k), d, 11 * k, ACCENT)
                x += int(26 * k)
            x += int(8 * k)
        sub = []
        if a.kind == T.SHOOT:
            for key in ("placement", "power", "shot_type"):
                if a.detail.get(key):
                    sub.append(str(a.detail[key]))
        if self.cfg.show_confidence:
            sub.append(rec.confidence_level)
        if sub:
            c.text("  ".join(sub), (x0, y0 + hgt + int(18 * k)), 0.5 * k, NEUTRAL, 1, 230)

    def _draw_debug(self, lines):
        c = self.canvas
        k = self.k
        lh = int(17 * k)
        w = int(330 * k)
        x0, y0 = c.w - w - int(10 * k), int(10 * k)
        c.rect(x0, y0, x0 + w, y0 + lh * len(lines) + int(10 * k), (15, 15, 15), 170)
        for i, ln in enumerate(lines):
            c.text(ln, (x0 + int(8 * k), y0 + lh * (i + 1)), 0.42 * k, NEUTRAL, 1, 255, outline=False)
