"""Versioned model registry: never deploy a worse model.

models/registry.json::

  {"detector": [{"version": "detector_v001", "path": "models/detector_v001.onnx",
                 "metrics": {...}, "primary": "f1", "higher_is_better": true,
                 "created": "...", "data": "...", "deployed": true}, ...],
   "ranker": [...], "predictor": [...]}

``register`` copies the artefact into models/ under its version name, stores
its benchmark, and marks it deployed only if its primary metric beats the
currently deployed version (ties keep the current one).  ``deployed(kind)``
returns the entry the runtime should load.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import shutil
from typing import Optional

DEFAULT_DIR = "models"


class Registry:
    def __init__(self, root: str = DEFAULT_DIR):
        self.root = root
        self.path = os.path.join(root, "registry.json")
        os.makedirs(root, exist_ok=True)
        self.data = json.load(open(self.path)) if os.path.exists(self.path) else {}

    def _save(self):
        with open(self.path, "w") as f:
            json.dump(self.data, f, indent=1)

    def entries(self, kind: str) -> list:
        return self.data.get(kind, [])

    def deployed(self, kind: str) -> Optional[dict]:
        return next((e for e in self.entries(kind) if e.get("deployed")), None)

    def redeploy(self, kind: str, primary: str, higher_is_better: bool = True) -> Optional[dict]:
        """Re-decide which entry is deployed under a (new) primary metric."""
        lst = [e for e in self.entries(kind) if e["metrics"].get(primary) is not None]
        if not lst:
            return None
        best = max(lst, key=lambda e: e["metrics"][primary] if higher_is_better else -e["metrics"][primary])
        for e in self.entries(kind):
            e["deployed"] = e is best
            e["primary"], e["higher_is_better"] = primary, higher_is_better
        self._save()
        return best

    def register(self, kind: str, artefact: str, metrics: dict, primary: str, higher_is_better: bool = True,
                 data: str = "", extra_files: tuple = (".json",), force: bool = False,
                 better: "Optional[bool]" = None) -> dict:
        """``better``: the caller compared new vs deployed on the *same* evaluation data
        (stored metrics may come from different data); None = compare stored metrics."""
        lst = self.data.setdefault(kind, [])
        version = f"{kind}_v{len(lst) + 1:03d}"
        ext = os.path.splitext(artefact)[1]
        dst = os.path.join(self.root, version + ext)
        shutil.copyfile(artefact, dst)
        for e in extra_files:
            if os.path.exists(artefact + e):
                shutil.copyfile(artefact + e, dst + e)
        cur = self.deployed(kind)
        if better is not None:
            better = bool(better) or cur is None or force
        else:
            better = cur is None or force
            if cur is not None and not force:
                a, b = metrics.get(primary), cur["metrics"].get(primary)
                if a is not None and b is not None:
                    better = a > b if higher_is_better else a < b
        entry = {"version": version, "path": dst, "metrics": metrics, "primary": primary,
                 "higher_is_better": higher_is_better, "data": data,
                 "created": _dt.datetime.now().isoformat(timespec="seconds"), "deployed": bool(better)}
        if better:
            for e in lst:
                e["deployed"] = False
        lst.append(entry)
        self._save()
        return entry
