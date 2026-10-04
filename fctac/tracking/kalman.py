"""2D constant-velocity Kalman filter (pitch metres), tiny and allocation-free."""
from __future__ import annotations

import numpy as np


class CVKalman:
    __slots__ = ("x", "P", "q")

    def __init__(self, pos, vel=None, pos_var=1.0, vel_var=9.0, q_acc=3.0):
        self.x = np.array([pos[0], pos[1], 0.0, 0.0] if vel is None else [pos[0], pos[1], vel[0], vel[1]], float)
        self.P = np.diag([pos_var, pos_var, vel_var, vel_var]).astype(float)
        self.q = q_acc                  # std of unmodelled acceleration (m/s^2)

    def predict(self, dt: float):
        if dt <= 0:
            return
        F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], float)
        g = np.array([0.5 * dt * dt, 0.5 * dt * dt, dt, dt])
        q2 = self.q * self.q
        Q = np.zeros((4, 4))
        # independent x / y white-acceleration noise
        for a, b in ((0, 2), (1, 3)):
            Q[a, a] = g[a] ** 2 * q2
            Q[a, b] = Q[b, a] = g[a] * g[b] * q2
            Q[b, b] = g[b] ** 2 * q2
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update(self, z, r_var: float):
        """Position measurement z (2,) with isotropic variance r_var."""
        Hx = self.x[:2]
        S = self.P[:2, :2] + np.eye(2) * r_var
        K = self.P[:, :2] @ np.linalg.inv(S)
        self.x = self.x + K @ (np.asarray(z, float) - Hx)
        self.P = (np.eye(4) - K @ np.hstack([np.eye(2), np.zeros((2, 2))])) @ self.P

    @property
    def pos(self) -> np.ndarray:
        return self.x[:2]

    @property
    def vel(self) -> np.ndarray:
        return self.x[2:]

    def pos_std(self) -> float:
        return float(np.sqrt(max(self.P[0, 0], self.P[1, 1])))
