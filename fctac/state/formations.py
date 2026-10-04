"""Formation templates in attack-aligned metres (x from own goal line).

y follows the attack-aligned frame: facing +x, larger y is the LEFT side.
Positions describe a neutral mid-block; callers shift/scale them.
"""
from __future__ import annotations

import numpy as np

_BACK4 = [("GK", 5, 34), ("LB", 30, 58), ("LCB", 25, 42), ("RCB", 25, 26), ("RB", 30, 10)]

FORMATIONS: dict[str, list[tuple[str, float, float]]] = {
    "4-3-3": _BACK4 + [("LCM", 45, 46), ("CDM", 38, 34), ("RCM", 45, 22),
                       ("LW", 62, 58), ("ST", 66, 34), ("RW", 62, 10)],
    "4-4-2": _BACK4 + [("LM", 48, 58), ("LCM", 44, 41), ("RCM", 44, 27), ("RM", 48, 10),
                       ("LS", 63, 40), ("RS", 63, 28)],
    "4-2-3-1": _BACK4 + [("LDM", 40, 42), ("RDM", 40, 26), ("LAM", 55, 56), ("CAM", 56, 34),
                         ("RAM", 55, 12), ("ST", 66, 34)],
    "3-5-2": [("GK", 5, 34), ("LCB", 26, 48), ("CB", 24, 34), ("RCB", 26, 20),
              ("LWB", 44, 60), ("LCM", 45, 44), ("CDM", 39, 34), ("RCM", 45, 24),
              ("RWB", 44, 8), ("LS", 63, 40), ("RS", 63, 28)],
    "4-1-2-1-2": _BACK4 + [("CDM", 38, 34), ("LCM", 46, 46), ("RCM", 46, 22), ("CAM", 55, 34),
                           ("LS", 64, 40), ("RS", 64, 28)],
}

ATTACKING_ROLES = {"ST", "LS", "RS", "LW", "RW", "CAM", "LAM", "RAM", "LM", "RM"}
DEFENSIVE_ROLES = {"GK", "LB", "RB", "LCB", "RCB", "CB", "LWB", "RWB"}


def template(name: str) -> tuple[list[str], np.ndarray]:
    slots = FORMATIONS[name]
    return [s[0] for s in slots], np.array([[s[1], s[2]] for s in slots], dtype=np.float64)
