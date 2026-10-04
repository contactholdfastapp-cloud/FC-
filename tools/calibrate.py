"""One-time setup on YOUR FC 27 footage: radar position, optional manual
pitch calibration, attack direction.  Writes a calib JSON used by replay/live.

    python tools/calibrate.py --video data/recordings/session01.mp4 --frame 300 --out configs/my_calib.json

Steps (keys shown in the window):
  1. RADAR   drag a rectangle tightly around the radar's pitch outline, ENTER
             -> the detected dots are drawn live so you can verify team colours
  2. PITCH   (optional) press a number to choose a landmark, click it on the
             frame; >= 4 landmarks -> manual homography.  ENTER to finish / skip
  3. DIR     press L if your team attacks towards the LEFT goal on screen in
             this half, R for right, A for automatic
Then the file is saved.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fctac.pitch.model import LANDMARKS  # noqa: E402
from fctac.vision.radar import RadarConfig, RadarReader  # noqa: E402

LM_KEYS = list(LANDMARKS)


def put(img, txt, y, col=(255, 255, 255)):
    cv2.putText(img, txt, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, txt, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 1, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--frame", type=int, default=100)
    ap.add_argument("--out", default="configs/my_calib.json")
    a = ap.parse_args()
    cap = cv2.VideoCapture(a.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, a.frame)
    ok, frame = cap.read()
    if not ok:
        raise SystemExit("cannot read frame")
    H, W = frame.shape[:2]
    s = min(1.0, 1600 / W)
    win = "FC27 calibrate"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    state = {"drag": None, "rect": None, "pts": {}, "lm": 0, "mouse": (0, 0)}

    def on_mouse(ev, x, y, flags, _):
        X, Y = x / s, y / s
        state["mouse"] = (X, Y)
        if state.get("step") == "radar":
            if ev == cv2.EVENT_LBUTTONDOWN:
                state["drag"] = (X, Y)
            elif ev == cv2.EVENT_LBUTTONUP and state["drag"] is not None:
                x0, y0 = state["drag"]
                state["rect"] = (min(x0, X) / W, min(y0, Y) / H, max(x0, X) / W, max(y0, Y) / H)
                state["drag"] = None
        elif state.get("step") == "pitch" and ev == cv2.EVENT_LBUTTONDOWN:
            state["pts"][LM_KEYS[state["lm"]]] = (X, Y)

    cv2.setMouseCallback(win, on_mouse)
    out = {"width": W, "height": H, "source": os.path.basename(a.video)}

    # 1) radar
    state["step"] = "radar"
    reader = None
    while True:
        img = frame.copy()
        if state["rect"] is not None:
            x0, y0, x1, y1 = state["rect"]
            cv2.rectangle(img, (int(x0 * W), int(y0 * H)), (int(x1 * W), int(y1 * H)), (0, 255, 255), 2)
            reader = RadarReader(RadarConfig(rect=state["rect"]))
            res = None
            for _ in range(3):          # auto colours need a couple of reads
                res = reader.read(frame)
            if res is not None and res.ok:
                for p, t in zip(res.points, res.teams):
                    u = x0 * W + p[0] / 105.0 * (x1 - x0) * W
                    v = y0 * H + (1 - p[1] / 68.0) * (y1 - y0) * H
                    cv2.circle(img, (int(u), int(v)), 6, (80, 220, 80) if t == 0 else (60, 60, 230), 2)
                put(img, f"radar OK: {len(res.points)} players (green = your team), controlled={'yes' if res.controlled is not None else 'no'}", 60, (80, 255, 80))
            else:
                put(img, "radar not read yet: adjust rectangle (or set colours in config)", 60, (80, 80, 255))
        put(img, "STEP 1: drag a rectangle around the radar pitch, ENTER to accept", 30)
        cv2.imshow(win, cv2.resize(img, None, fx=s, fy=s) if s < 1 else img)
        k = cv2.waitKey(30) & 0xFF
        if k in (13, 10) and state["rect"] is not None:
            out["radar_rect"] = [round(v, 4) for v in state["rect"]]
            break
        if k == 27:
            return

    # 2) manual pitch landmarks
    state["step"] = "pitch"
    while True:
        img = frame.copy()
        for name, (x, y) in state["pts"].items():
            cv2.circle(img, (int(x), int(y)), 6, (0, 255, 255), 2)
            cv2.putText(img, name, (int(x) + 8, int(y) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        put(img, f"STEP 2 (optional): landmark [{state['lm']}] {LM_KEYS[state['lm']]}  -- [ / ] change, click to place,"
                 f" ENTER done ({len(state['pts'])} placed)", 30)
        cv2.imshow(win, cv2.resize(img, None, fx=s, fy=s) if s < 1 else img)
        k = cv2.waitKey(30) & 0xFF
        if k == ord("]"):
            state["lm"] = (state["lm"] + 1) % len(LM_KEYS)
        elif k == ord("["):
            state["lm"] = (state["lm"] - 1) % len(LM_KEYS)
        elif k in (13, 10):
            break
        elif k == 27:
            return
    if len(state["pts"]) >= 4:
        src = np.array([state["pts"][n] for n in state["pts"]], np.float64)
        dst = np.array([LANDMARKS[n] for n in state["pts"]], np.float64)
        Hm, _ = cv2.findHomography(src, dst, 0)
        if Hm is not None:
            out["H_img2pitch"] = (Hm / Hm[2, 2]).tolist()
            err = np.linalg.norm(cv2.perspectiveTransform(src[None], Hm)[0] - dst, axis=1)
            print(f"manual homography: mean landmark error {err.mean():.2f} m")

    # 3) attack direction
    state["step"] = "dir"
    while True:
        img = frame.copy()
        put(img, "STEP 3: your team attacks towards the  L)eft  /  R)ight  goal (this half)?   A) automatic", 30)
        cv2.imshow(win, cv2.resize(img, None, fx=s, fy=s) if s < 1 else img)
        k = cv2.waitKey(30) & 0xFF
        if k in (ord("l"), ord("r"), ord("a")):
            if k != ord("a"):
                out["attack_sign"] = -1 if k == ord("l") else 1
            break
        if k == 27:
            return
    cv2.destroyAllWindows()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print("saved", a.out, json.dumps(out)[:300])


if __name__ == "__main__":
    main()
