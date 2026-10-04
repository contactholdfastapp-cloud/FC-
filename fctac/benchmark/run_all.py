"""Phase 14: one repeatable benchmark run -> benchmarks/<timestamp>.json + .md

    python -m fctac.benchmark.run_all --clips data/synthetic/clip01 data/synthetic/clip1440
    python -m fctac.benchmark.run_all --clips data/recordings/annotated01 --labels data/datasets/fc27

Sections (each skipped when its inputs are missing):
  vision      detector precision/recall/foot error/ball/controlled/latency per clip (needs GT)
  labels      baseline vs deployed learned detector on labelled frames (val split)
  pipeline    end-to-end state accuracy + per-stage latency per clip (needs GT)
  live        capture->overlay latency, processed fps, dropped frames (video source)
  tactics     ranker report (runs/ranker/report.json) and counterfactual results if present
  prediction  predictor report (runs/predictor/report.json) if present
  machine     tools/inspect_machine.py summary
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import subprocess
import sys


def _git_rev() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    except Exception:
        return "?"


def run(clips: list[str], labels: str = "", frames: int = 600, live_seconds: float = 15.0, config: str = "") -> dict:
    rep: dict = {"created": dt.datetime.now().isoformat(timespec="seconds"), "git": _git_rev(),
                 "platform": platform.platform(), "python": sys.version.split()[0], "config": config}
    from fctac.benchmark.pipeline_eval import evaluate
    from fctac.benchmark.vision_eval import evaluate_detector
    from fctac.config import load_config
    from fctac.state.gt import load_gt
    from fctac.vision.analyzer import make_detector
    cfg = load_config(config or None)
    rep["vision"], rep["pipeline"], rep["live"] = {}, {}, {}
    for c in clips:
        name = os.path.basename(c)
        if os.path.exists(c + ".gt.jsonl"):
            gt = load_gt(c + ".gt.jsonl")
            rep["vision"][name] = evaluate_detector(make_detector(cfg), c + ".mp4", gt, frames)
            r = evaluate(c, frames, config=config)
            rep["pipeline"][name] = r
        if live_seconds > 0:
            from fctac.capture.base import VideoFileSource
            from fctac.live import LiveRuntime, NullPresenter
            from fctac.vision.analyzer import VisionAnalyzer
            cfg2 = load_config(config or None)
            rt = LiveRuntime(cfg2, VideoFileSource(c + ".mp4"), NullPresenter(), VisionAnalyzer(cfg2))
            rep["live"][name] = rt.run(live_seconds)
    if labels and os.path.exists(os.path.join(labels, "val.txt")):
        from fctac.training.det_eval import evaluate_on_labels, evaluate_onnx_on_labels
        from fctac.training.registry import Registry
        from fctac.vision.detector import ColorDetector
        paths = [p for p in open(os.path.join(labels, "val.txt")).read().split() if p]
        rep["labels"] = {"baseline_color": evaluate_on_labels(ColorDetector(cfg.detector), paths)}
        dep = Registry().deployed("detector")
        if dep:
            rep["labels"][dep["version"]] = evaluate_onnx_on_labels(dep["path"], paths)
    for key, path in (("tactics", "runs/ranker/report.json"), ("counterfactual", "runs/ranker/counterfactual.json"),
                      ("prediction", "runs/predictor/report.json")):
        if os.path.exists(path):
            rep[key] = json.load(open(path))
    if os.path.exists("models/registry.json"):
        reg = json.load(open("models/registry.json"))
        rep["models"] = {k: [{"version": e["version"], "deployed": e["deployed"], "primary": e["primary"],
                              e["primary"]: e["metrics"].get(e["primary"])} for e in v] for k, v in reg.items()}
    return rep


def to_markdown(rep: dict) -> str:
    L = [f"# Benchmark {rep['created']} (git {rep['git']})", "", f"Platform: {rep['platform']}", ""]
    if rep.get("pipeline"):
        L += ["## Pipeline (state reconstruction vs ground truth)", "",
              "| clip | pos err m | team | controlled | ball err m | calib err m | possession | phase | total ms (mean/p95) |",
              "|---|---|---|---|---|---|---|---|---|"]
        for k, r in rep["pipeline"].items():
            t = r["latency_ms"].get("total", {})
            L.append(f"| {k} | {r['player_pos_err_m']} | {r['team_acc']} | {r['controlled_acc']} | {r['ball_err_m_median']} | "
                     f"{r['calib_err_m_median']} | {r.get('possession_acc')} | {r.get('phase_acc')} | {t.get('mean')}/{t.get('p95')} |")
        L.append("")
    if rep.get("vision"):
        L += ["## Detector (main view)", "", "| clip | P | R | foot err px | ball R | controlled | ms |", "|---|---|---|---|---|---|---|"]
        for k, r in rep["vision"].items():
            L.append(f"| {k} | {r['player_precision']} | {r['player_recall']} | {r['foot_err_px_mean']} | {r['ball_recall']} | "
                     f"{r['controlled_acc']} | {r['ms_mean']} |")
        L.append("")
    if rep.get("live"):
        L += ["## Live loop (video file as capture source)", "", "| clip | processed | dropped | e2e ms mean | e2e p95 | analysis ms |",
              "|---|---|---|---|---|---|"]
        for k, r in rep["live"].items():
            e = r["latency_ms"].get("e2e", {})
            an = r["latency_ms"].get("analysis", {})
            L.append(f"| {k} | {r['frames_processed']} | {r['frames_dropped']} | {e.get('mean')} | {e.get('p95')} | {an.get('mean')} |")
        L.append("")
    if rep.get("labels"):
        L += ["## Detectors on labelled frames (val split)", "", "| model | F1 | P | R | ball R | controlled | foot px | ms |",
              "|---|---|---|---|---|---|---|---|"]
        for k, r in rep["labels"].items():
            L.append(f"| {k} | {r['f1']} | {r['player_precision']} | {r['player_recall']} | {r['ball_recall']} | "
                     f"{r['controlled_acc']} | {r['foot_err_px']} | {r['ms_mean']} |")
        L.append("")
    if rep.get("prediction"):
        L += ["## Movement prediction (ADE, metres)", ""]
        for k, v in rep["prediction"].items():
            if isinstance(v, dict):
                L.append(f"- {k}: " + ", ".join(f"{h}={e}" for h, e in v.items()))
        L.append("")
    if rep.get("models"):
        L += ["## Model registry", "", "```", json.dumps(rep["models"], indent=1), "```", ""]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="*", default=[])
    ap.add_argument("--labels", default="")
    ap.add_argument("--frames", type=int, default=600)
    ap.add_argument("--live-seconds", type=float, default=15)
    ap.add_argument("--config", default="")
    ap.add_argument("--out", default="benchmarks")
    a = ap.parse_args()
    rep = run(a.clips, a.labels, a.frames, a.live_seconds, a.config)
    os.makedirs(a.out, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(os.path.join(a.out, f"{stamp}.json"), "w") as f:
        json.dump(rep, f, indent=1)
    md = to_markdown(rep)
    with open(os.path.join(a.out, f"{stamp}.md"), "w") as f:
        f.write(md)
    print(md)


if __name__ == "__main__":
    main()
