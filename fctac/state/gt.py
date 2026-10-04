"""Build GameState directly from synthetic ground truth (oracle perception).

Used to (a) develop/evaluate tactics independently of vision errors and
(b) measure how much perception error costs (oracle vs. vision pipeline).
"""
from __future__ import annotations

import json
from typing import Optional

import numpy as np

from fctac.pitch.camera import CameraParams
from fctac import types as T


def load_gt(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def state_from_gt(snap: dict, cam: Optional[CameraParams] = None, rng: Optional[np.random.Generator] = None,
                  pos_noise: float = 0.0, visible_only: bool = False) -> T.GameState:
    sign = int(snap.get("us_attack_sign", 1))
    if cam is None and "camera" in snap:
        cam = CameraParams(**snap["camera"])
    H = cam.H_img2pitch() if cam is not None else None
    gs = T.GameState(frame=int(snap.get("video_frame", snap["frame"])), t=float(snap["t"]), players=[], ball=None,
                     attack_sign=sign, H_img2pitch=H, calib_conf=1.0 if H is not None else 0.0)
    raw = np.array([[p["x"], p["y"]] for p in snap["players"]])
    if pos_noise > 0 and rng is not None:
        raw = raw + rng.normal(0, pos_noise, raw.shape)
    screen = None
    if cam is not None:
        uv, z = cam.project(np.concatenate([raw, np.zeros((len(raw), 1))], 1))
        screen = uv
    for k, p in enumerate(snap["players"]):
        if visible_only and screen is not None:
            u, v = screen[k]
            if not (0 <= u < cam.width and 0 <= v < cam.height):
                continue
        team = T.TEAM_US if p["team"] == 0 else T.TEAM_THEM
        pos = gs.to_attack(raw[k])
        vel = gs.vel_to_raw(np.array([p["vx"], p["vy"]]))
        ps = T.PlayerState(id=p["id"], team=team, pos=pos, vel=vel, acc=np.zeros(2),
                           controlled=bool(p["controlled"]), role=p["role"],
                           screen=None if screen is None else screen[k].copy())
        gs.players.append(ps)
        if ps.controlled:
            gs.controlled_id = ps.id
    b = snap["ball"]
    bpos = gs.to_attack(np.array([b["x"], b["y"]]))
    bvel = gs.vel_to_raw(np.array([b["vx"], b["vy"]]))
    owner = b.get("owner")
    owner_team = T.TEAM_UNKNOWN
    if owner is not None:
        owner_team = T.TEAM_US if snap["players"][owner]["team"] == 0 else T.TEAM_THEM
    bscreen = None
    if cam is not None:
        bscreen = cam.project(np.array([[b["x"], b["y"], b.get("z", 0.0)]]))[0][0]
    gs.ball = T.BallState(pos=bpos, vel=bvel, confidence=1.0, owner_id=owner, owner_team=owner_team,
                          screen=bscreen, height=float(b.get("z", 0.0)))
    poss = snap.get("possession", -1)
    gs.possession = T.TEAM_US if poss == 0 else (T.TEAM_THEM if poss == 1 else T.TEAM_UNKNOWN)
    return gs
