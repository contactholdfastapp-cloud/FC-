"""Human-action event format (shared by GT conversion, auto-labelling, viewer).

One event per line (jsonl):
  {"frame": 120, "end_frame": 151, "t": 4.0, "kind": "PASS", "team": 0,
   "from_id": 7, "to_id": 9, "target_label": "ST", "outcome": "complete",
   "target": [x, y]  # attack-aligned metres if known
  }
``frame`` is the video frame where the action started.
"""
from __future__ import annotations

import json

import numpy as np

from fctac import types as T

SIM_KIND = {"pass": T.PASS, "through": T.THROUGH, "lob": T.LOB, "cross": T.CROSS, "shot": T.SHOOT,
            "tackle": "TACKLE"}
SUCCESS = {"complete", "goal", "won"}


def events_from_gt(gt: list[dict]) -> list[dict]:
    started: dict[tuple, dict] = {}
    out = []
    for vf, snap in enumerate(gt):
        sign = snap.get("us_attack_sign", 1)
        roles = {p["id"]: p["role"] for p in snap["players"]}
        for e in snap.get("events", []):
            key = (e["kind"], e["t"], e["from_id"])
            tgt = np.array(e["target"], float)
            if sign < 0:
                tgt = T.PITCH_SIZE - tgt
            started[key] = {
                "frame": vf, "t": snap["t"], "kind": SIM_KIND.get(e["kind"], e["kind"].upper()),
                "team": e["team"], "from_id": e["from_id"], "to_id": e["to_id"],
                "target_label": roles.get(e["to_id"], "") if e["to_id"] is not None else "",
                "target": tgt.tolist(), "outcome": "pending", "end_frame": -1,
            }
        for e in snap.get("resolved", []):
            key = (e["kind"], e["t"], e["from_id"])
            ev = started.pop(key, None)
            if ev is None:
                continue
            ev["outcome"] = e["outcome"]
            ev["end_frame"] = vf
            ev["receiver_id"] = e.get("receiver_id")
            out.append(ev)
    out.extend(started.values())
    out.sort(key=lambda e: e["frame"])
    return out


def save_events(path: str, events: list[dict]):
    with open(path, "w") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")


def load_events(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(x) for x in f if x.strip()]


def describe(e: dict) -> str:
    k = e["kind"]
    lab = e.get("target_label") or ""
    s = f"{k} -> {lab}" if lab and k in T.PASS_KINDS else k
    return f"{s} ({e.get('outcome', '?')})"
