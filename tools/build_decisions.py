"""Build decision-window datasets (and validate the auto-labeller).

    # real recordings: vision pipeline -> states -> auto-labels
    python tools/build_decisions.py --videos data/recordings/session01.mp4 --out data/decisions

    # synthetic clips: oracle or vision states, validated against true events
    python tools/build_decisions.py --videos data/synthetic/clip01.mp4 --source vision --out data/decisions

    # large synthetic corpus straight from the simulator (no rendering)
    python tools/build_decisions.py --sim-minutes 60 --seeds 1 2 3 4 --out data/decisions
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fctac import types as T  # noqa: E402
from fctac.training.decisions import DecisionLabeler, write_examples  # noqa: E402
from fctac.training.events import events_from_gt  # noqa: E402


def validate_against_gt(examples: list, gt_events: list, tol_frames: int = 12) -> dict:
    """How well do auto-labels match the simulator's true actions (team 0)?"""
    human = [e for e in gt_events if e["team"] == 0 and e["kind"] in T.PASS_KINDS + (T.SHOOT,)]
    labelled = [x for x in examples if x["human"]["kind"] in T.PASS_KINDS + (T.SHOOT,)]
    used = set()
    hit, kind_ok, recv_ok, recv_n, out_ok = 0, 0, 0, 0, 0
    for e in human:
        best = None
        for j, x in enumerate(labelled):
            if j in used:
                continue
            df = x["frame"] - e["frame"]
            if -tol_frames <= df <= 4 and (best is None or abs(df) < abs(best[1])):
                best = (j, df)
        if best is None:
            continue
        used.add(best[0])
        x = labelled[best[0]]
        hit += 1
        fam = lambda k: "shot" if k == T.SHOOT else "pass"   # noqa: E731
        kind_ok += int(fam(x["human"]["kind"]) == fam(e["kind"]))
        if e["kind"] != T.SHOOT and e["outcome"] in ("complete", "intercepted"):
            out_ok += int(x["outcome"]["success"] == (e["outcome"] == "complete"))
        if e["kind"] != T.SHOOT and e["outcome"] == "complete":
            recv_n += 1
            # receiver ids differ between GT and tracks: compare receiver positions
            r = x["outcome"].get("receiver_id")
            recv_ok += int(r is not None and r == x["human"]["target_id"])
    return {"gt_actions": len(human), "auto_actions": len(labelled),
            "recall": round(hit / max(len(human), 1), 3), "precision": round(hit / max(len(labelled), 1), 3),
            "family_acc": round(kind_ok / max(hit, 1), 3), "outcome_acc": round(out_ok / max(hit, 1), 3),
            "receiver_consistency": round(recv_ok / max(recv_n, 1), 3)}


def from_video(path: str, source: str, config: str = "") -> tuple[list, dict]:
    from fctac.capture.video import VideoSource
    base = os.path.splitext(path)[0]
    src = VideoSource(path)
    gt = None
    if os.path.exists(base + ".gt.jsonl"):
        from fctac.state.gt import load_gt
        gt = load_gt(base + ".gt.jsonl")
    if source == "oracle":
        from fctac.pipeline import OracleAnalyzer
        an = OracleAnalyzer(gt)
    else:
        from fctac.vision.analyzer import VisionAnalyzer
        an = VisionAnalyzer.from_files(config)
    lab = DecisionLabeler(os.path.basename(path))
    ex = []
    for i in range(len(src)):
        frame = src.get(i) if source != "oracle" else None
        if source != "oracle" and frame is None:
            break
        fa = an.process(frame, i, i / src.fps)
        ex += lab.push(fa.state)
    ex += lab.flush()
    report = validate_against_gt(ex, events_from_gt(gt)) if gt is not None else {}
    return ex, report


def from_sim(minutes: float, seed: int) -> tuple[list, dict]:
    from fctac.sim.match import MatchSim, SimConfig
    from fctac.state.gt import state_from_gt
    sim = MatchSim(SimConfig(seed=seed, us_attack_sign=1 if seed % 2 else -1))
    lab = DecisionLabeler(f"sim_seed{seed}")
    ex = []
    snaps = []
    n = int(minutes * 60 * sim.cfg.fps)
    for k in range(n):
        sim.step()
        s = sim.snapshot()
        s["video_frame"] = k
        snaps.append({"events": s["events"], "resolved": s["resolved"], "t": s["t"],
                      "us_attack_sign": s["us_attack_sign"], "players": s["players"]})
        ex += lab.push(state_from_gt(s))
    ex += lab.flush()
    return ex, validate_against_gt(ex, events_from_gt(snaps))


def summarize(ex: list) -> dict:
    kinds = Counter(x["human"]["kind"] for x in ex)
    res = Counter(x["outcome"]["result"] for x in ex)
    matched = sum(1 for x in ex if x["human"]["index"] >= 0)
    return {"examples": len(ex), "kinds": dict(kinds), "results": dict(res),
            "matched_to_candidate": round(matched / max(len(ex), 1), 3),
            "mean_candidates": round(float(np.mean([len(x["candidates"]) for x in ex])), 1) if ex else 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="*", default=[])
    ap.add_argument("--source", default="vision", choices=("vision", "oracle"))
    ap.add_argument("--sim-minutes", type=float, default=0)
    ap.add_argument("--seeds", nargs="*", type=int, default=[1])
    ap.add_argument("--out", default="data/decisions")
    ap.add_argument("--config", default="")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for v in a.videos:
        ex, rep = from_video(v, a.source, a.config)
        name = os.path.splitext(os.path.basename(v))[0] + f".{a.source}"
        write_examples(os.path.join(a.out, name + ".jsonl"), ex)
        print(name, json.dumps(summarize(ex)), "\n  vs GT:", json.dumps(rep))
    if a.sim_minutes > 0:
        for seed in a.seeds:
            ex, rep = from_sim(a.sim_minutes, seed)
            name = f"sim_seed{seed}"
            write_examples(os.path.join(a.out, name + ".jsonl"), ex)
            print(name, json.dumps(summarize(ex)), "\n  vs GT:", json.dumps(rep))


if __name__ == "__main__":
    main()
