"""Formation fitting and role labels (ST, RW, LCB, ...) with temporal voting."""
from __future__ import annotations

from collections import Counter, defaultdict, deque

import numpy as np
from scipy.optimize import linear_sum_assignment

from fctac.state.formations import FORMATIONS, template


def fit_formation(pos: np.ndarray, names=None) -> tuple[str, list[str], float]:
    """Assign attack-aligned positions (N<=11,2) to the best formation template.

    The template is shifted/stretched to the team's block (centroid + spread)
    so it works in any phase (deep block, high press).  Returns
    (formation, roles per input row, mean cost in metres).
    """
    names = names or list(FORMATIONS)
    best = (None, None, np.inf)
    if len(pos) < 4:
        return "", [""] * len(pos), np.inf
    for name in names:
        roles, tpl = template(name)
        # goalkeeper anchored to own goal; outfield template matched to outfield block
        out_t = tpl[1:]
        gk_idx = int(np.argmin(pos[:, 0]))
        others = [i for i in range(len(pos)) if i != gk_idx]
        P = pos[others]
        if len(P) == 0:
            continue
        sx = max(np.std(P[:, 0]), 3.0) / max(np.std(out_t[:, 0]), 1e-3)
        sy = max(np.std(P[:, 1]), 3.0) / max(np.std(out_t[:, 1]), 1e-3)
        scale = np.clip(np.array([sx, sy]), 0.5, 1.6)
        T_ = (out_t - out_t.mean(0)) * scale + P.mean(0)
        D = np.linalg.norm(P[:, None, :] - T_[None, :, :], axis=2)
        a, b = linear_sum_assignment(D)
        cost = float(D[a, b].mean())
        if cost < best[2]:
            out = [""] * len(pos)
            out[gk_idx] = "GK" if pos[gk_idx, 0] < 25 else roles[1 + 0]
            for i, j in zip(a, b):
                out[others[i]] = roles[1 + j]
            best = (name, out, cost)
    return best


class RoleTracker:
    """Smooths per-track role labels and the formation over time."""

    def __init__(self, window: int = 25, every_n: int = 6):
        self.votes = defaultdict(lambda: deque(maxlen=window))
        self.form_votes = deque(maxlen=window)
        self.every_n = every_n
        self.n = 0
        self.formation = ""

    def update(self, ids: list[int], pos: np.ndarray) -> dict[int, str]:
        self.n += 1
        if self.n % self.every_n == 1 and len(ids) >= 6:
            form, roles, _ = fit_formation(pos)
            if form:
                self.form_votes.append(form)
                self.formation = Counter(self.form_votes).most_common(1)[0][0]
                for i, r in zip(ids, roles):
                    self.votes[i].append(r)
        return {i: Counter(self.votes[i]).most_common(1)[0][0] for i in ids if self.votes[i]}
