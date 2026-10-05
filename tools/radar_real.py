"""Run the FC 27 radar model on real harvested radar crops.

    python tools/radar_real.py eval   --model runs/radar/radar.onnx --groups m3 --sheet runs/radar/qa_m3.jpg
    python tools/radar_real.py pseudo --model runs/radar/radar.onnx --groups m1,m2 --out data/radar/real_pseudo

eval   -- label-free consistency metrics on real footage (there is no ground
          truth): how often exactly 11 triangles + 11 circles are read, ball
          visibility, and frame-to-frame persistence (a symbol read at t is
          found again 0.2 s later within 6 px, same shape).  Optional QA sheet.
pseudo -- writes confident real crops as training labels (canonical crops +
          labels.jsonl).  A crop is used only if its team counts are
          plausible and its symbols persist in the neighbouring crops;
          uncertain detections become "ignore" regions.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fctac.training.radar_model import match_points, onnx_predictor  # noqa: E402
from fctac.vision.radar import RadarConfig, radar_crop_rect  # noqa: E402
from fctac.vision.radar_fc27 import CH, CW, decode, harvest_crop_to_canonical  # noqa: E402


def load_crops(root: str, groups, margin: float = 0.10, step: int = 1):
    rows = [r for r in csv.DictReader(open(os.path.join(root, "index.csv")))
            if not groups or r["group"] in groups]
    rows.sort(key=lambda r: (r["clip"], int(r["frame"])))
    panel = RadarConfig().panel
    out = []
    for r in rows[::step]:
        w, h = int(r["w"]), int(r["h"])
        cx0, cy0, _, _ = radar_crop_rect(panel, w, h, margin)
        pin = (panel[0] * w - cx0, panel[1] * h - cy0, panel[2] * w - cx0, panel[3] * h - cy0)
        r["path"] = os.path.join(root, "radar", f"{r['clip']}_{int(r['frame']):05d}.jpg")
        r["panel_in_crop"] = pin
        out.append(r)
    return out


def run_color_baseline(rows: list):
    """The pre-FC 27 colour/dot radar reader on the same crops (baseline)."""
    from fctac.vision.radar import RadarReader
    from fctac.vision.radar_fc27 import pitch_to_canon
    panel = RadarConfig().panel
    reader = RadarReader(RadarConfig(rect=panel, pitch_rect=(0.0, 0.0, 1.0, 1.0)))
    res = []
    for r in rows:
        w, h = int(r["w"]), int(r["h"])
        crop = cv2.imread(r["path"])
        frame = np.zeros((h, w, 3), np.uint8)
        cx0, cy0, _, _ = radar_crop_rect(panel, w, h, 0.10)
        frame[cy0:cy0 + crop.shape[0], cx0:cx0 + crop.shape[1]] = crop
        rr = reader.read(frame)
        canon = harvest_crop_to_canonical(crop, r["panel_in_crop"])
        if rr.ok and len(rr.points):
            uv = pitch_to_canon(rr.points)
            shape = np.asarray(rr.teams, int).clip(0, 1)
            hl = np.zeros(len(uv))
            if rr.controlled is not None:
                hl[rr.controlled] = 1.0
        else:
            uv, shape, hl = np.zeros((0, 2)), np.zeros(0, int), np.zeros(0)
        ball = pitch_to_canon(rr.ball[None])[0] if rr.ok and rr.ball is not None else None
        d = {"uv": uv, "score": np.ones(len(uv)), "shape": shape, "highlight": hl, "hl_thr": 0.5,
             "ball_uv": ball, "ball_score": 1.0 if ball is not None else 0.0}
        d["low"] = d
        res.append((r, canon, d))
    return res


def run(model: str, rows: list, thr: float):
    if model == "color":
        return run_color_baseline(rows)
    pred = onnx_predictor(model, ["CPUExecutionProvider"])
    res = []
    for r in rows:
        crop = harvest_crop_to_canonical(cv2.imread(r["path"]), r["panel_in_crop"])
        heat = pred(crop)
        d = decode(heat, thr)
        d["low"] = decode(heat, 0.12)                      # weak peaks (ignore regions for pseudo-labels)
        res.append((r, crop, d))
    return res


def persistence(prev, cur, tol=6.0):
    if len(prev["uv"]) == 0 or len(cur["uv"]) == 0:
        return None
    m = match_points(prev["uv"], cur["uv"], tol)
    same = sum(1 for i, j, _ in m if prev["shape"][i] == cur["shape"][j])
    return same / len(prev["uv"])


def metrics(res: list) -> dict:
    stats = defaultdict(list)
    prev_by_clip = {}
    for r, crop, d in res:
        panel = float(r["radar_line"]) > 20
        n_t = int((d["shape"] == 0).sum())
        n_c = int((d["shape"] == 1).sum())
        key = "panel" if panel else "no_panel"
        stats[key + "_n"].append(n_t + n_c)
        stats[key + "_read"].append(n_t + n_c >= 15)
        if panel:
            stats["exact_11_11"].append(n_t == 11 and n_c == 11)
            stats["teams_le_11"].append(n_t <= 11 and n_c <= 11)
            stats["n_20_22"].append(20 <= n_t + n_c <= 22)
            stats["ball"].append(d["ball_uv"] is not None)
            stats["hl_any"].append(bool((d["highlight"] >= d["hl_thr"]).any()))
        p = prev_by_clip.get(r["clip"])
        if p is not None and int(r["frame"]) - int(p[0]["frame"]) <= 12 and panel and float(p[0]["radar_line"]) > 20:
            v = persistence(p[2], d)
            if v is not None:
                stats["persist"].append(v)
        prev_by_clip[r["clip"]] = (r, crop, d)
    out = {"crops": len(res), "panel_crops": len(stats["panel_n"])}
    for k, v in stats.items():
        out[k] = round(float(np.mean(v)), 4) if v else None
    return out


def real_score(m: dict) -> float:
    """Label-free composite on radar-visible crops (higher = better): plausible
    player count, no team above 11, symbols persist 0.2 s later, ball seen."""
    g = lambda k: float(m.get(k) or 0.0)   # noqa: E731
    return 0.35 * g("n_20_22") + 0.25 * g("teams_le_11") + 0.25 * g("persist") + 0.15 * g("ball")


def draw(crop, d, scale=3):
    im = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    for (u, v), s, hl in zip(d["uv"], d["shape"], d["highlight"]):
        c = (int(u * scale + 1), int(v * scale + 1))
        col = (255, 255, 0) if s == 0 else (255, 0, 255)
        if s == 0:
            cv2.drawMarker(im, c, col, cv2.MARKER_TRIANGLE_UP, 22, 2)
        else:
            cv2.circle(im, c, 12, col, 2)
        if hl >= d["hl_thr"]:
            cv2.circle(im, c, 18, (0, 0, 255), 2)
    if d["ball_uv"] is not None:
        u, v = d["ball_uv"]
        cv2.drawMarker(im, (int(u * scale + 1), int(v * scale + 1)), (0, 255, 255), cv2.MARKER_TILTED_CROSS, 26, 2)
    n_t, n_c = int((d["shape"] == 0).sum()), int((d["shape"] == 1).sum())
    cv2.putText(im, f"tri {n_t} cir {n_c} ball {'y' if d['ball_uv'] is not None else 'n'}", (6, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return im


def sheet(res, path, n=12, seed=0, only_panel=True, cols=3):
    rng = np.random.default_rng(seed)
    cand = [x for x in res if (float(x[0]["radar_line"]) > 20) == only_panel]
    if not cand:
        return
    pick = [cand[i] for i in rng.choice(len(cand), min(n, len(cand)), replace=False)]
    ims = []
    for r, crop, d in pick:
        im = draw(crop, d)
        cv2.putText(im, f"{r['clip']}_{r['frame']}", (6, im.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1)
        ims.append(im)
    while len(ims) % cols:
        ims.append(np.zeros_like(ims[0]))
    rows = [np.hstack(ims[i:i + cols]) for i in range(0, len(ims), cols)]
    cv2.imwrite(path, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])


def pseudo(res: list, out: str, min_persist: float = 0.85):
    os.makedirs(os.path.join(out, "img"), exist_ok=True)
    by_clip = defaultdict(list)
    for x in res:
        by_clip[x[0]["clip"]].append(x)
    n = 0
    with open(os.path.join(out, "labels.jsonl"), "w") as f:
        for clip, xs in by_clip.items():
            for k, (r, crop, d) in enumerate(xs):
                n_t, n_c = int((d["shape"] == 0).sum()), int((d["shape"] == 1).sum())
                nb = [xs[j][2] for j in (k - 1, k + 1) if 0 <= j < len(xs)
                      and abs(int(xs[j][0]["frame"]) - int(r["frame"])) <= 12]
                if n_t + n_c < 15 or n_t > 11 or n_c > 11 or len(nb) < 2:
                    continue
                pers = [persistence(d, e) for e in nb]
                if any(p is None or p < min_persist for p in pers):
                    continue
                # weak peaks that are not confident detections -> ignore regions
                ign = []
                low = d["low"]
                if len(low["uv"]):
                    m = match_points(low["uv"], d["uv"], 2.5)
                    matched = {i for i, _, _ in m}
                    ign = [[round(float(u), 1), round(float(v), 1), 7.0] for i, (u, v) in enumerate(low["uv"])
                           if i not in matched]
                b = d["ball_uv"]
                if b is None and low["ball_uv"] is not None:
                    ign.append([round(float(low["ball_uv"][0]), 1), round(float(low["ball_uv"][1]), 1), 7.0])
                name = f"{clip}_{int(r['frame']):05d}.jpg"
                cv2.imwrite(os.path.join(out, "img", name), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
                players = [[round(float(u), 2), round(float(v), 2), int(s), int(h >= d["hl_thr"])]
                           for (u, v), s, h in zip(d["uv"], d["shape"], d["highlight"])]
                f.write(json.dumps({"id": n, "file": name, "mode": "real", "players": players,
                                    "ball": None if b is None else [round(float(b[0]), 2), round(float(b[1]), 2)],
                                    "ignore": ign, "group": r["group"]}) + "\n")
                n += 1
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["eval", "pseudo"])
    ap.add_argument("--model", required=True, help='ONNX radar model, or "color" for the colour-reader baseline')
    ap.add_argument("--root", default="data/real/harvest")
    ap.add_argument("--groups", default="m3")
    ap.add_argument("--thr", type=float, default=0.35)
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--sheet", default="")
    ap.add_argument("--report", default="")
    ap.add_argument("--out", default="data/radar/real_pseudo")
    a = ap.parse_args()
    groups = tuple(g for g in a.groups.split(",") if g)
    rows = load_crops(a.root, groups, step=a.step)
    res = run(a.model, rows, a.thr)
    m = metrics(res)
    m["real_score"] = round(real_score(m), 4)
    print(json.dumps(m))
    if a.report:
        json.dump(m, open(a.report, "w"), indent=1)
    if a.sheet:
        sheet(res, a.sheet)
        sheet(res, a.sheet.replace(".jpg", "_nopanel.jpg"), only_panel=False, n=9)
    if a.cmd == "pseudo":
        print("pseudo-labelled crops:", pseudo(res, a.out))


if __name__ == "__main__":
    main()
