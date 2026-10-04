"""Render synthetic broadcast-style frames (with a FC-style radar) from sim state."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from fctac.pitch import model as pm
from fctac.pitch.camera import CameraParams
from fctac.types import PITCH_LENGTH as L, PITCH_WIDTH as W

# BGR kit palette: (shirt, shorts)
KITS = {
    "red": ((40, 40, 205), (240, 240, 240)),
    "blue": ((190, 80, 25), (245, 245, 245)),
    "white": ((235, 235, 235), (40, 40, 40)),
    "yellow": ((20, 215, 240), (120, 40, 20)),
    "black": ((30, 30, 30), (30, 30, 30)),
    "skyblue": ((235, 190, 120), (250, 250, 250)),
    "purple": ((140, 40, 120), (240, 240, 240)),
}
GK_KITS = [((0, 140, 255), (20, 20, 20)), ((150, 220, 20), (20, 20, 20)), ((200, 0, 200), (10, 10, 10))]
REF_KIT = ((15, 15, 15), (15, 15, 15))
SKIN = [(140, 170, 220), (95, 130, 185), (60, 85, 120), (120, 160, 205)]
INDICATOR_BGR = (255, 210, 0)      # controlled-player marker (cyan-ish blue)
RADAR_RECT = (0.415, 0.795, 0.585, 0.985)   # normalised x0, y0, x1, y1


@dataclass
class RenderStyle:
    kit_us: str = "red"
    kit_them: str = "blue"
    noise: float = 3.0
    radar: bool = True
    hud: bool = True


class CameraController:
    """Broadcast camera that follows the ball with lag and gentle zoom."""

    def __init__(self, width=1280, height=720, seed=0, base: CameraParams | None = None):
        rng = np.random.default_rng(seed + 1000)
        self.base = base or CameraParams(cx=52.5, cy=rng.uniform(-42, -34), cz=rng.uniform(18, 24),
                                         focal=1.45 * width, width=width, height=height)
        self.lx = L / 2
        self.zoom_phase = rng.uniform(0, 6.28)
        self.cam = CameraParams(**self.base.to_dict())

    def update(self, ball_xy, ball_v, t, dt) -> CameraParams:
        target = float(np.clip(ball_xy[0] + 0.5 * ball_v[0], 16, L - 16))
        a = 1 - np.exp(-dt / 0.55)
        self.lx += a * (target - self.lx)
        c = self.cam
        c.focal = self.base.focal * (1.0 + 0.07 * np.sin(0.21 * t + self.zoom_phase))
        ly = 30.0 + 0.15 * (ball_xy[1] - W / 2)
        c.look_at(np.array([self.lx, ly, 0.0]))
        return CameraParams(**c.to_dict())


class Renderer:
    PPM = 10.0                      # texture pixels per metre
    EXT = (-25.0, -18.0, 130.0, 90.0)

    def __init__(self, width=1280, height=720, style: RenderStyle = RenderStyle(), seed=0):
        self.w, self.h = width, height
        self.style = style
        self.rng = np.random.default_rng(seed + 7)
        self.tex = self._make_texture()
        x0, y0 = self.EXT[0], self.EXT[1]
        self.T = np.array([[1 / self.PPM, 0, x0], [0, 1 / self.PPM, y0], [0, 0, 1.0]])
        self.skin = [SKIN[i % len(SKIN)] for i in self.rng.permutation(23)]
        gk = self.rng.permutation(len(GK_KITS))
        self.gk_kits = (GK_KITS[gk[0]], GK_KITS[gk[1]])
        self.stand = np.array([42, 38, 46], np.uint8)
        # pre-generated sensor/compression-like noise, applied with saturating ops
        self._noise = []
        self._noise_i = 0
        for _ in range(6 if style.noise > 0 else 0):
            n = self.rng.normal(0, style.noise, (height, width, 3))
            self._noise.append((np.clip(n, 0, 255).astype(np.uint8), np.clip(-n, 0, 255).astype(np.uint8)))

    # --- static texture ------------------------------------------------------
    def _make_texture(self) -> np.ndarray:
        x0, y0, x1, y1 = self.EXT
        tw, th = int((x1 - x0) * self.PPM), int((y1 - y0) * self.PPM)
        tex = np.zeros((th, tw, 3), np.uint8)
        # stands: dark noisy crowd
        crowd = self.rng.integers(0, 60, (th, tw, 1)).astype(np.int16)
        tex[:] = np.clip(np.array([38, 36, 44], np.int16) + crowd - 30, 0, 255).astype(np.uint8)
        # grass area incl. run-off
        def to_px(x, y):
            return int(round((x - x0) * self.PPM)), int(round((y - y0) * self.PPM))
        gx0, gy0 = to_px(-6, -5)
        gx1, gy1 = to_px(L + 6, W + 5)
        base = np.array([62, 138, 66], np.int16)
        light = np.array([72, 156, 78], np.int16)
        stripe_w = L / 20
        xs = (np.arange(tw) / self.PPM + x0)
        stripe = ((np.floor(xs / stripe_w)).astype(int) % 2 == 0)
        grass = np.where(stripe[None, :, None], light, base).astype(np.int16)
        grass = np.repeat(grass, th, axis=0)
        n = cv2.GaussianBlur(self.rng.normal(0, 6, (th, tw)).astype(np.float32), (0, 0), 3)
        grass = np.clip(grass + n[..., None].astype(np.int16), 0, 255).astype(np.uint8)
        tex[gy0:gy1, gx0:gx1] = grass[gy0:gy1, gx0:gx1]
        # advertising boards (flat band)
        for (bx0, by0, bx1, by1) in ((-6, -5, L + 6, -4), (-6, W + 4, L + 6, W + 5)):
            p0, p1 = to_px(bx0, by0), to_px(bx1, by1)
            tex[p0[1]:p1[1], p0[0]:p1[0]] = (180, 120, 30)
        # lines
        lc = (232, 236, 236)
        thick = 2
        sh = 4
        def pt(p):
            return (int(round((p[0] - x0) * self.PPM * (1 << sh))), int(round((p[1] - y0) * self.PPM * (1 << sh))))
        for a, b in pm.line_segments():
            cv2.line(tex, pt(a), pt(b), lc, thick, cv2.LINE_AA, sh)
        for poly in pm.arcs():
            pts = np.array([pt(p) for p in poly], np.int32)
            cv2.polylines(tex, [pts], False, lc, thick, cv2.LINE_AA, sh)
        cv2.circle(tex, pt((L / 2, W / 2)), int(0.3 * self.PPM * (1 << sh)), lc, -1, cv2.LINE_AA, sh)
        for sx in (pm.PEN_SPOT, L - pm.PEN_SPOT):
            cv2.circle(tex, pt((sx, W / 2)), int(0.25 * self.PPM * (1 << sh)), lc, -1, cv2.LINE_AA, sh)
        return tex

    # --- per frame -------------------------------------------------------------
    def render(self, snap: dict, cam: CameraParams) -> np.ndarray:
        H = cam.H_pitch2img()
        img = cv2.warpPerspective(self.tex, H @ self.T, (self.w, self.h), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=self.stand.tolist())
        self._mask_horizon(img, H)
        self._draw_goals(img, cam)
        self._draw_people(img, snap, cam)
        if self.style.radar:
            draw_radar(img, snap, self._kit(0)[0], self._kit(1)[0])
        if self.style.hud:
            self._draw_hud(img, snap)
        if self.style.noise > 0:
            pos, neg = self._noise[self._noise_i % len(self._noise)]
            self._noise_i += 1
            cv2.add(img, pos, dst=img)
            cv2.subtract(img, neg, dst=img)
        return img

    def _kit(self, team):
        return KITS[self.style.kit_us if team == 0 else self.style.kit_them]

    def _mask_horizon(self, img, H):
        l = np.cross(H[:, 0], H[:, 1])
        ref = H @ np.array([L / 2, W / 2, 1.0])
        s = np.sign(l @ ref)
        # evaluate on the 4 corners to skip the common case quickly
        cs = np.array([[0, 0, 1], [self.w, 0, 1], [0, self.h, 1], [self.w, self.h, 1]], float)
        if np.all(np.sign(cs @ l) == s):
            return
        yy, xx = np.mgrid[0:self.h, 0:self.w]
        side = (l[0] * xx + l[1] * yy + l[2]) * s
        img[side <= 1e-6 * np.abs(l).max()] = self.stand

    def _draw_goals(self, img, cam):
        for gx in (0.0, L):
            post_y = (W / 2 - pm.GOAL_WIDTH / 2, W / 2 + pm.GOAL_WIDTH / 2)
            back = gx - 2.0 if gx == 0 else gx + 2.0
            pts3 = []
            for y in post_y:
                pts3 += [(gx, y, 0), (gx, y, pm.GOAL_HEIGHT)]
            P, z = cam.project(np.array(pts3))
            if np.any(z <= 0):
                continue
            P = P.astype(np.int32)
            Pb, _ = cam.project(np.array([(back, post_y[0], 0), (back, post_y[1], 0),
                                          (back, post_y[0], pm.GOAL_HEIGHT * 0.8), (back, post_y[1], pm.GOAL_HEIGHT * 0.8)]))
            Pb = Pb.astype(np.int32)
            net = (205, 205, 205)
            cv2.line(img, tuple(P[1]), tuple(Pb[2]), net, 1, cv2.LINE_AA)
            cv2.line(img, tuple(P[3]), tuple(Pb[3]), net, 1, cv2.LINE_AA)
            cv2.line(img, tuple(Pb[2]), tuple(Pb[3]), net, 1, cv2.LINE_AA)
            cv2.line(img, tuple(Pb[0]), tuple(Pb[2]), net, 1, cv2.LINE_AA)
            cv2.line(img, tuple(Pb[1]), tuple(Pb[3]), net, 1, cv2.LINE_AA)
            for a, b in ((0, 1), (2, 3), (1, 3)):
                cv2.line(img, tuple(P[a]), tuple(P[b]), (250, 250, 250), 3, cv2.LINE_AA)

    def _draw_people(self, img, snap, cam):
        people = []
        for p in snap["players"]:
            people.append((p["x"], p["y"], p["team"], p["role"] == "GK", p["controlled"], p["id"], p["vx"], p["vy"]))
        r = snap.get("referee")
        if r:
            people.append((r["x"], r["y"], 2, False, False, 22, 0.0, 0.0))
        P = np.array([[q[0], q[1], 0.0] for q in people] + [[q[0], q[1], 1.85] for q in people])
        uv, z = cam.project(P)
        n = len(people)
        order = np.argsort(uv[:n, 1])
        t = snap["t"]
        b = snap["ball"]
        ball_uv, ball_z = cam.project(np.array([[b["x"], b["y"], b["z"]], [b["x"], b["y"], 0.0]]))
        ball_drawn = False
        for k in order:
            x, y, team, gk, ctrl, pid, vx, vy = people[k]
            if z[k] <= 0:
                continue
            if not ball_drawn and ball_uv[1, 1] < uv[k, 1]:
                self._draw_ball(img, ball_uv, ball_z, cam)
                ball_drawn = True
            fu, fv = uv[k]
            hv = uv[n + k, 1]
            hgt = fv - hv
            if hgt < 4 or fu < -50 or fu > self.w + 50 or fv < -20 or hv > self.h + 20:
                continue
            if team == 2:
                shirt, shorts = REF_KIT
            elif gk:
                shirt, shorts = self.gk_kits[team]
            else:
                shirt, shorts = self._kit(team)
            self._draw_person(img, fu, fv, hgt, shirt, shorts, self.skin[pid % 23],
                              np.hypot(vx, vy), t + pid * 0.37)
            if ctrl:
                cx, cy = fu, fv - hgt * 1.22
                s = max(5.0, 0.16 * hgt)
                tri = np.array([[cx - s, cy - s], [cx + s, cy - s], [cx, cy + 0.2 * s]], np.int32)
                cv2.fillConvexPoly(img, tri, INDICATOR_BGR, cv2.LINE_AA)
                cv2.polylines(img, [tri], True, (40, 40, 40), 1, cv2.LINE_AA)
        if not ball_drawn:
            self._draw_ball(img, ball_uv, ball_z, cam)

    def _draw_person(self, img, fu, fv, h, shirt, shorts, skin, speed, phase):
        sh = 4
        S = 1 << sh
        def P(x, y):
            return (int(round(x * S)), int(round(y * S)))
        # shadow
        cv2.ellipse(img, P(fu + 0.12 * h, fv), (int(0.22 * h * S), int(0.06 * h * S)), 0, 0, 360,
                    (40, 92, 44), -1, cv2.LINE_AA, sh)
        swing = 0.10 * h * np.sin(phase * 9.0) * min(1.0, speed / 4.0)
        hip_v = fv - 0.48 * h
        lw = max(1, int(0.07 * h))
        cv2.line(img, P(fu - 0.05 * h, hip_v), P(fu - 0.05 * h + swing, fv), skin, lw, cv2.LINE_AA, sh)
        cv2.line(img, P(fu + 0.05 * h, hip_v), P(fu + 0.05 * h - swing, fv), skin, lw, cv2.LINE_AA, sh)
        # socks
        cv2.line(img, P(fu - 0.05 * h + swing * 0.6, fv - 0.18 * h), P(fu - 0.05 * h + swing, fv), shirt, lw, cv2.LINE_AA, sh)
        cv2.line(img, P(fu + 0.05 * h - swing * 0.6, fv - 0.18 * h), P(fu + 0.05 * h - swing, fv), shirt, lw, cv2.LINE_AA, sh)
        # shorts
        cv2.ellipse(img, P(fu, hip_v + 0.02 * h), (int(0.13 * h * S), int(0.08 * h * S)), 0, 0, 360, shorts, -1, cv2.LINE_AA, sh)
        # torso + arms
        cv2.ellipse(img, P(fu, fv - 0.68 * h), (int(0.15 * h * S), int(0.2 * h * S)), 0, 0, 360, shirt, -1, cv2.LINE_AA, sh)
        cv2.line(img, P(fu - 0.14 * h, fv - 0.8 * h), P(fu - 0.2 * h - swing * 0.5, fv - 0.55 * h), skin, lw, cv2.LINE_AA, sh)
        cv2.line(img, P(fu + 0.14 * h, fv - 0.8 * h), P(fu + 0.2 * h + swing * 0.5, fv - 0.55 * h), skin, lw, cv2.LINE_AA, sh)
        # head
        cv2.circle(img, P(fu, fv - 0.93 * h), int(0.075 * h * S), skin, -1, cv2.LINE_AA, sh)

    def _draw_ball(self, img, ball_uv, ball_z, cam):
        if ball_z[0] <= 0:
            return
        r = max(2.0, 0.11 * cam.focal / ball_z[0])
        sh = 4
        S = 1 << sh
        su, sv = ball_uv[1]
        cv2.ellipse(img, (int(su * S), int(sv * S)), (int(r * 1.1 * S), int(r * 0.45 * S)), 0, 0, 360,
                    (40, 92, 44), -1, cv2.LINE_AA, sh)
        u, v = ball_uv[0]
        cv2.circle(img, (int(u * S), int(v * S)), int(r * S), (248, 248, 248), -1, cv2.LINE_AA, sh)
        cv2.circle(img, (int(u * S), int(v * S)), int(r * S), (90, 90, 90), 1, cv2.LINE_AA, sh)

    def _draw_hud(self, img, snap):
        x, y = int(0.03 * self.w), int(0.035 * self.h)
        cv2.rectangle(img, (x, y), (x + 230, y + 30), (25, 25, 25), -1)
        cv2.rectangle(img, (x, y), (x + 8, y + 30), self._kit(0)[0], -1)
        cv2.rectangle(img, (x + 222, y), (x + 230, y + 30), self._kit(1)[0], -1)
        m, s = divmod(int(snap["t"] * 6), 60)   # accelerated match clock like FC
        txt = f"USR {snap['score'][0]} - {snap['score'][1]} CPU  {m:02d}:{s:02d}"
        cv2.putText(img, txt, (x + 14, y + 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA)


def radar_rect_px(width: int, height: int, rect=RADAR_RECT) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = rect
    return int(x0 * width), int(y0 * height), int(x1 * width), int(y1 * height)


def draw_radar(img, snap, col_us, col_them, rect=RADAR_RECT):
    h, w = img.shape[:2]
    x0, y0, x1, y1 = radar_rect_px(w, h, rect)
    roi = img[y0:y1, x0:x1]
    roi[:] = (roi.astype(np.float32) * 0.35 + np.array([20, 20, 20]) * 0.65).astype(np.uint8)
    rw, rh = x1 - x0, y1 - y0

    def P(x, y):
        return (int(round(x0 + x / L * rw)), int(round(y0 + (1 - y / W) * rh)))

    lc = (150, 150, 150)
    cv2.rectangle(img, P(0, W), P(L, 0), lc, 1)
    cv2.line(img, P(L / 2, 0), P(L / 2, W), lc, 1)
    for gx, s in ((0, 1), (L, -1)):
        cv2.rectangle(img, P(gx, (W + pm.PEN_WIDTH) / 2), P(gx + s * pm.PEN_DEPTH, (W - pm.PEN_WIDTH) / 2), lc, 1)
    k = h / 720.0                       # HUD scales with resolution, like the game's
    r_dot, r_ring, r_ball = max(2, round(3 * k)), max(4, round(5 * k)), max(1, round(2 * k))
    th = max(1, round(k))
    for p in snap["players"]:
        c = col_us if p["team"] == 0 else col_them
        q = P(p["x"], p["y"])
        cv2.circle(img, q, r_dot, c, -1, cv2.LINE_AA)
        if p["controlled"]:
            cv2.circle(img, q, r_ring, (0, 230, 255), th, cv2.LINE_AA)
    b = snap["ball"]
    cv2.circle(img, P(b["x"], b["y"]), r_ball, (255, 255, 255), -1, cv2.LINE_AA)
