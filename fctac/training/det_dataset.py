"""Detection dataset: per-image JSON labels, leakage-safe splits, statistics.

Label file ``<image>.json``::

  {"image": "f_000123.jpg", "width": 2560, "height": 1440,
   "source": "session01.mp4", "frame": 123, "status": "auto" | "verified",
   "objects": [{"cls": "player", "x": 812.5, "y": 640.0, "h": 96.0,
                "team": 0, "controlled": false},
               {"cls": "ball", "x": 900.1, "y": 655.2, "h": 9.0}]}

Players are anchored at the foot point (what the pitch projection needs),
the ball at its centre.  ``team``: 0 = user's team, 1 = opponent, 2 = other
(referee/unknown kit), -1 = unlabelled.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
from collections import Counter

CLASSES = ("player", "ball")


def label_path(image_path: str) -> str:
    return image_path + ".json"


def load_label(image_path: str) -> dict:
    with open(label_path(image_path)) as f:
        return json.load(f)


def save_label(image_path: str, label: dict):
    with open(label_path(image_path), "w") as f:
        json.dump(label, f, indent=0)


def list_images(root: str) -> list[str]:
    out = []
    for ext in ("jpg", "png"):
        out += glob.glob(os.path.join(root, "**", f"*.{ext}"), recursive=True)
    return sorted(p for p in out if os.path.exists(label_path(p)))


def segment_key(label: dict, segment_s: float = 30.0, fps: float = 30.0) -> str:
    """Frames of the same source within one segment are near-duplicates: they
    must land in the same split."""
    seg = int(label.get("frame", 0) / (segment_s * fps))
    return f"{label.get('source', '?')}#{seg}"


def split_of(key: str, val: float = 0.15, test: float = 0.15) -> str:
    h = int(hashlib.sha1(key.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    if h < test:
        return "test"
    if h < test + val:
        return "val"
    return "train"


def make_splits(root: str, val: float = 0.15, test: float = 0.15, verified_only: bool = False) -> dict:
    splits = {"train": [], "val": [], "test": []}
    for p in list_images(root):
        lab = load_label(p)
        if verified_only and lab.get("status") != "verified":
            continue
        splits[split_of(segment_key(lab), val, test)].append(p)
    for k, v in splits.items():
        with open(os.path.join(root, f"{k}.txt"), "w") as f:
            f.write("\n".join(v))
    return splits


def stats(paths: list[str]) -> dict:
    c = Counter()
    teams = Counter()
    status = Counter()
    sources = set()
    for p in paths:
        lab = load_label(p)
        status[lab.get("status", "?")] += 1
        sources.add(lab.get("source", "?"))
        for o in lab["objects"]:
            c[o["cls"]] += 1
            if o["cls"] == "player":
                teams[o.get("team", -1)] += 1
                c["controlled"] += int(bool(o.get("controlled")))
    return {"images": len(paths), "sources": len(sources), "objects": dict(c), "teams": dict(teams),
            "status": dict(status)}


def validate(paths: list[str]) -> list[str]:
    """Return a list of problems (empty = clean)."""
    problems = []
    for p in paths:
        try:
            lab = load_label(p)
        except Exception as e:
            problems.append(f"{p}: unreadable label ({e})")
            continue
        w, h = lab.get("width", 0), lab.get("height", 0)
        n_ctrl = 0
        n_ball = 0
        for o in lab.get("objects", []):
            if o.get("cls") not in CLASSES + ("referee",):
                problems.append(f"{p}: unknown class {o.get('cls')}")
            if not (0 <= o.get("x", -1) < w and 0 <= o.get("y", -1) < h):
                problems.append(f"{p}: object outside image {o}")
            n_ctrl += int(bool(o.get("controlled")))
            n_ball += int(o.get("cls") == "ball")
        if n_ctrl > 1:
            problems.append(f"{p}: {n_ctrl} controlled players")
        if n_ball > 1:
            problems.append(f"{p}: {n_ball} balls")
    return problems
