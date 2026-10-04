"""Pinhole broadcast-camera model.

FC's default broadcast cameras sit high behind the near touchline and mostly
pan (yaw), tilt and zoom while following the ball.  Parameterising the
camera this way (instead of a free 8-DoF homography) makes calibration far
more stable: frame-to-frame tracking only refines a few physical parameters.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np


def apply_h(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a 3x3 homography to (N,2) points."""
    pts = np.asarray(pts, dtype=np.float64)
    single = pts.ndim == 1
    p = np.atleast_2d(pts)
    q = p @ H[:, :2].T + H[:, 2]
    w = q[:, 2:3]
    w = np.where(np.abs(w) < 1e-12, 1e-12, w)
    out = q[:, :2] / w
    return out[0] if single else out


@dataclass
class CameraParams:
    cx: float = 52.5        # camera position (raw pitch metres)
    cy: float = -40.0
    cz: float = 22.0
    yaw: float = 0.0        # radians, + pans towards +x
    tilt: float = 0.30      # radians, + looks down
    focal: float = 1400.0   # pixels (for the given image width)
    width: int = 1280
    height: int = 720

    # --- geometry --------------------------------------------------------------
    def rotation(self) -> np.ndarray:
        cy, sy = np.cos(self.yaw), np.sin(self.yaw)
        ct, st = np.cos(self.tilt), np.sin(self.tilt)
        fwd = np.array([sy * ct, cy * ct, -st])
        right = np.array([cy, -sy, 0.0])
        down = np.cross(fwd, right)
        return np.stack([right, down, fwd])

    def K(self) -> np.ndarray:
        return np.array([[self.focal, 0, self.width / 2.0],
                         [0, self.focal, self.height / 2.0],
                         [0, 0, 1.0]])

    def position(self) -> np.ndarray:
        return np.array([self.cx, self.cy, self.cz])

    def project(self, P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Project (N,3) world points; returns (N,2) pixels and (N,) depth."""
        P = np.atleast_2d(np.asarray(P, dtype=np.float64))
        R = self.rotation()
        Xc = (P - self.position()) @ R.T
        z = Xc[:, 2]
        zs = np.where(np.abs(z) < 1e-9, 1e-9, z)
        u = self.focal * Xc[:, 0] / zs + self.width / 2.0
        v = self.focal * Xc[:, 1] / zs + self.height / 2.0
        return np.stack([u, v], 1), z

    def H_pitch2img(self) -> np.ndarray:
        R = self.rotation()
        t = -R @ self.position()
        M = np.stack([R[:, 0], R[:, 1], t], 1)
        H = self.K() @ M
        return H / H[2, 2]

    def H_img2pitch(self) -> np.ndarray:
        H = np.linalg.inv(self.H_pitch2img())
        return H / H[2, 2]

    def look_at(self, target: np.ndarray) -> "CameraParams":
        d = np.asarray(target, float) - self.position()
        self.yaw = float(np.arctan2(d[0], d[1]))
        self.tilt = float(np.arctan2(-d[2], np.hypot(d[0], d[1])))
        return self

    def as_vector(self) -> np.ndarray:
        return np.array([self.cx, self.cy, self.cz, self.yaw, self.tilt, self.focal])

    @classmethod
    def from_vector(cls, v, width: int, height: int) -> "CameraParams":
        return cls(cx=v[0], cy=v[1], cz=v[2], yaw=v[3], tilt=v[4], focal=v[5],
                   width=width, height=height)

    def to_dict(self) -> dict:
        return {k: (float(v) if isinstance(v, float) else v) for k, v in asdict(self).items()}

    def scaled(self, width: int, height: int) -> "CameraParams":
        s = width / self.width
        return CameraParams(self.cx, self.cy, self.cz, self.yaw, self.tilt,
                            self.focal * s, width, height)
