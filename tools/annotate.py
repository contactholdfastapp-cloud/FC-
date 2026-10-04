"""Review / correct detection labels (OpenCV GUI).

    python tools/annotate.py data/datasets/fc27            # all images under the folder
    python tools/annotate.py data/datasets/fc27 --only-auto

Mouse: LEFT add player at foot point | SHIFT+LEFT add ball | RIGHT delete nearest
Keys : T cycle team of nearest (us/them/other) | C set nearest as controlled
       +/- adjust height of nearest player box
       V mark verified + next   N next   P previous   S save   Z zoom 2x around cursor
       ESC save & quit
Colours: green = us, red = them, grey = other/unknown, yellow ring = controlled,
white = ball.  Labels are saved next to each image (<image>.json).
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fctac.training.det_dataset import list_images, load_label, save_label, stats, validate  # noqa: E402

TEAM_COL = {0: (80, 220, 80), 1: (60, 60, 230), 2: (170, 170, 170), -1: (170, 170, 170)}


class Annotator:
    def __init__(self, paths: list[str], max_w: int = 1600):
        self.paths = paths
        self.i = 0
        self.max_w = max_w
        self.cursor = (0, 0)
        self.zoom = False
        self.dirty = False
        self.load()

    def load(self):
        self.img = cv2.imread(self.paths[self.i])
        self.lab = load_label(self.paths[self.i])
        h, w = self.img.shape[:2]
        self.s = min(1.0, self.max_w / w)
        self.dirty = False

    def save(self):
        save_label(self.paths[self.i], self.lab)
        self.dirty = False

    def nearest(self, x, y, cls=None):
        best, bd = None, 1e9
        for o in self.lab["objects"]:
            if cls and o["cls"] != cls:
                continue
            d = np.hypot(o["x"] - x, o["y"] - y)
            if d < bd:
                best, bd = o, d
        return best if bd < 60 else None

    def on_mouse(self, ev, x, y, flags, _):
        X, Y = x / self.s, y / self.s
        self.cursor = (X, Y)
        if ev == cv2.EVENT_LBUTTONDOWN:
            if flags & cv2.EVENT_FLAG_SHIFTKEY:
                self.lab["objects"] = [o for o in self.lab["objects"] if o["cls"] != "ball"]
                self.lab["objects"].append({"cls": "ball", "x": X, "y": Y, "h": 8.0})
            else:
                hs = [o["h"] for o in self.lab["objects"] if o["cls"] == "player" and o.get("h")]
                self.lab["objects"].append({"cls": "player", "x": X, "y": Y, "h": float(np.median(hs)) if hs else 60.0,
                                            "team": -1, "controlled": False})
            self.dirty = True
        elif ev == cv2.EVENT_RBUTTONDOWN:
            o = self.nearest(X, Y)
            if o is not None:
                self.lab["objects"].remove(o)
                self.dirty = True

    def key(self, k) -> bool:
        X, Y = self.cursor
        ch = chr(k & 0xFF).lower() if k != -1 else ""
        if k == 27:
            if self.dirty:
                self.save()
            return False
        if ch == "t":
            o = self.nearest(X, Y, "player")
            if o is not None:
                o["team"] = {0: 1, 1: 2, 2: 0, -1: 0}[o.get("team", -1)]
                self.dirty = True
        elif ch == "c":
            o = self.nearest(X, Y, "player")
            if o is not None:
                for q in self.lab["objects"]:
                    q["controlled"] = False
                o["controlled"] = True
                o["team"] = 0
                self.dirty = True
        elif ch in "+=-":
            o = self.nearest(X, Y, "player")
            if o is not None:
                o["h"] = max(8.0, o.get("h", 60.0) * (1.1 if ch in "+=" else 0.9))
                self.dirty = True
        elif ch == "s":
            self.save()
        elif ch == "z":
            self.zoom = not self.zoom
        elif ch in "vnp":
            if ch == "v":
                self.lab["status"] = "verified"
                self.dirty = True
            if self.dirty:
                self.save()
            self.i = max(0, min(len(self.paths) - 1, self.i + (-1 if ch == "p" else 1)))
            self.load()
        return True

    def render(self):
        img = self.img.copy()
        for o in self.lab["objects"]:
            x, y = int(o["x"]), int(o["y"])
            if o["cls"] == "ball":
                cv2.circle(img, (x, y), 10, (255, 255, 255), 2)
                continue
            col = TEAM_COL.get(o.get("team", -1), (170, 170, 170))
            h = int(o.get("h", 60))
            w = max(6, h // 3)
            cv2.rectangle(img, (x - w // 2, y - h), (x + w // 2, y), col, 2)
            cv2.circle(img, (x, y), 4, col, -1)
            if o.get("controlled"):
                cv2.circle(img, (x, y - h - 12), 9, (0, 230, 255), 3)
        if self.zoom:
            X, Y = map(int, self.cursor)
            r = 120
            x0, y0 = max(0, X - r), max(0, Y - r)
            crop = img[y0:Y + r, x0:X + r]
            if crop.size:
                big = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
                img[: big.shape[0], : big.shape[1]] = big[: img.shape[0], : img.shape[1]]
        disp = cv2.resize(img, None, fx=self.s, fy=self.s, interpolation=cv2.INTER_AREA) if self.s < 1 else img
        st = self.lab.get("status", "?")
        txt = f"[{self.i + 1}/{len(self.paths)}] {os.path.basename(self.paths[self.i])}  status={st}{'  *' if self.dirty else ''}"
        cv2.putText(disp, txt, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(disp, txt, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        return disp


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root")
    ap.add_argument("--only-auto", action="store_true")
    ap.add_argument("--check", action="store_true", help="validate labels and print statistics, no GUI")
    a = ap.parse_args()
    paths = list_images(a.root)
    if a.only_auto:
        paths = [p for p in paths if load_label(p).get("status") != "verified"]
    if a.check or not paths:
        print(stats(paths))
        probs = validate(paths)
        print(f"{len(probs)} problems")
        for p in probs[:50]:
            print(" ", p)
        return
    an = Annotator(paths)
    win = "FC27 annotate"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win, an.on_mouse)
    while True:
        cv2.imshow(win, an.render())
        if not an.key(cv2.waitKeyEx(30)):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
