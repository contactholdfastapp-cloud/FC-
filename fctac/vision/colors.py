"""Colour utilities: fast Lab conversion and online team-colour prototypes."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np


def bgr_to_lab(bgr) -> np.ndarray:
    a = np.asarray(bgr, np.uint8).reshape(-1, 1, 3)
    return cv2.cvtColor(a, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)


def lab_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.asarray(a, np.float32) - np.asarray(b, np.float32), axis=-1)


@dataclass
class TeamPrototypes:
    """Running colour prototypes (Lab) for our team, opponent, referee/others.

    Initialised from config (or auto-clustered), then slowly adapted with
    confidently classified samples so lighting changes are absorbed.
    """
    us: Optional[np.ndarray] = None
    them: Optional[np.ndarray] = None
    others: list = field(default_factory=list)       # referee, goalkeepers
    rate: float = 0.02
    max_dist: float = 38.0

    def ready(self) -> bool:
        return self.us is not None and self.them is not None

    def classify(self, lab: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(N,3) Lab -> team ids (0 us, 1 them, 2 other, -1 unknown), confidence."""
        n = len(lab)
        if not self.ready() or n == 0:
            return np.full(n, -1), np.zeros(n)
        protos = [self.us, self.them] + list(self.others)
        D = np.stack([lab_dist(lab, p) for p in protos], 1)
        k = np.argmin(D, 1)
        d1 = D[np.arange(n), k]
        Ds = np.sort(D, 1)
        d2 = Ds[:, 1] if D.shape[1] > 1 else d1 + 50
        team = np.where(k >= 2, 2, k)
        conf = np.clip((d2 - d1) / (d2 + 1e-3), 0, 1) * np.clip(1.5 - d1 / self.max_dist, 0, 1)
        team = np.where(d1 > self.max_dist * 1.5, -1, team)
        return team, conf

    def adapt(self, lab: np.ndarray, team: np.ndarray, conf: np.ndarray, min_conf: float = 0.5):
        for t, attr in ((0, "us"), (1, "them")):
            m = (team == t) & (conf >= min_conf)
            if m.sum() >= 2:
                cur = getattr(self, attr)
                setattr(self, attr, (1 - self.rate) * cur + self.rate * lab[m].mean(0))

    @staticmethod
    def cluster(samples_lab: np.ndarray, k: int = 3) -> np.ndarray:
        """k-means on colour samples; returns centres sorted by cluster size (desc)."""
        z = samples_lab.astype(np.float32)
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
        _, labels, centres = cv2.kmeans(z, k, None, crit, 4, cv2.KMEANS_PP_CENTERS)
        counts = np.bincount(labels.ravel(), minlength=k)
        return centres[np.argsort(-counts)]
