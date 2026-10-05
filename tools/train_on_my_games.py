"""Improve the assistant's models with YOUR OWN FC 27 recordings (one command).

    python tools/train_on_my_games.py data\\recordings\\game1.mp4 data\\recordings\\game2.mp4
    (or double-click scripts\\train_my_games.bat and drop the videos on it)

What it does, all locally:
  1. harvests frames + radar crops from the recordings
  2. radar: auto-labels your radar crops with the current radar model (only
     confident, stable crops), fine-tunes the model on them (+ generated
     FC 27-style radars), compares new vs current on a held-out recording
  3. players: auto-labels players in your frames from radar + calibration,
     fine-tunes the player detector, compares new vs current on the held-out
     recording (the colour detector is the baseline)
  4. a model is switched on only if it measured better on the same held-out
     footage (models/registry.json keeps every version)

Use at least 2 recordings (the last one is held out for the comparison);
10+ minutes of normal gameplay with the 2D radar on works well.  With a
CUDA GPU the training steps take minutes; on CPU expect ~1 hour.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
PY = sys.executable


def run(args: list, **kw):
    print(">>", " ".join(str(a) for a in args), flush=True)
    r = subprocess.run([PY] + [str(a) for a in args], cwd=ROOT, **kw)
    if r.returncode:
        raise SystemExit(f"step failed ({r.returncode}): {' '.join(map(str, args))}")


def radar_score(m: dict) -> float:
    from tools.radar_real import real_score
    return real_score(m)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--work", default="data/my_games")
    ap.add_argument("--radar-epochs", type=int, default=3)
    ap.add_argument("--det-epochs", type=int, default=15)
    ap.add_argument("--synth", type=int, default=12000, help="generated FC 27-style radars to mix in")
    ap.add_argument("--webcams", default="none", choices=["none", "livestream"])
    ap.add_argument("--skip-radar", action="store_true")
    ap.add_argument("--skip-detector", action="store_true")
    a = ap.parse_args()
    vids = [os.path.abspath(v) for v in a.videos]
    if len(vids) < 2:
        raise SystemExit("give at least 2 recordings: the last one is kept aside to compare old vs new models")
    work = os.path.join(ROOT, a.work)
    harvest = os.path.join(work, "harvest")
    names = [os.path.splitext(os.path.basename(v))[0] for v in vids]
    held = names[-1]
    from fctac.training.registry import Registry
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    summary = {}

    # 1. harvest (file name -> group, so the held-out recording stays separate)
    if not os.path.exists(os.path.join(harvest, "index.csv")):
        run(["tools/harvest_frames.py", *vids, "--out", harvest])

    # 2. radar
    if not a.skip_radar:
        reg = Registry()
        cur = reg.deployed("radar")
        if cur is None:
            raise SystemExit("no radar model deployed (models/registry.json)")
        train_groups = ",".join(n for n in names if n != held)
        pseudo = os.path.join(work, "radar_pseudo")
        shutil.rmtree(pseudo, ignore_errors=True)
        run(["tools/radar_real.py", "pseudo", "--model", cur["path"], "--root", harvest, "--groups", train_groups,
             "--out", pseudo])
        synth = os.path.join(ROOT, "data", "radar", "synth_my")
        if a.synth and not os.path.exists(os.path.join(synth, "labels.jsonl")):
            run(["tools/make_radar_data.py", "--out", synth, "--n", a.synth, "--bg-root", harvest])
        out = os.path.join(ROOT, "runs", "radar", "my_games")
        init = cur["path"] + ".pt"
        args = ["tools/train_radar.py", "--synth", synth, "--real", pseudo, "--epochs", a.radar_epochs,
                "--out", out, "--lr", "1e-3", "--val", "500"]
        if os.path.exists(init):
            args += ["--init", init]
        run(args)
        rep_old = os.path.join(work, "radar_eval_current.json")
        rep_new = os.path.join(work, "radar_eval_new.json")
        run(["tools/radar_real.py", "eval", "--model", cur["path"], "--root", harvest, "--groups", held,
             "--report", rep_old])
        run(["tools/radar_real.py", "eval", "--model", os.path.join(out, "radar.onnx"), "--root", harvest,
             "--groups", held, "--report", rep_new, "--sheet", os.path.join(work, "radar_qa.jpg")])
        mo, mn = json.load(open(rep_old)), json.load(open(rep_new))
        so, sn = radar_score(mo), radar_score(mn)
        metrics = {**{f"real_{k}": v for k, v in mn.items() if isinstance(v, (int, float))}, "radar_score": round(sn, 4)}
        e = Registry().register("radar", os.path.join(out, "radar.onnx"), metrics, primary="radar_score",
                                data=",".join(vids), extra_files=(".json", ".pt"), better=sn > so)
        summary["radar"] = {"current": round(so, 4), "new": round(sn, 4), "version": e["version"],
                            "switched_on": e["deployed"]}
        print("radar:", summary["radar"], flush=True)

    # 3. player detector
    if not a.skip_detector:
        det_root = os.path.join(work, "det")
        for v, n in zip(vids, names):
            d = os.path.join(det_root, n)
            if not os.path.exists(d) or not os.listdir(d):
                run(["tools/pseudo_label_view.py", "--video", v, "--out", d, "--source", n, "--webcams", a.webcams])
        train_dirs = [os.path.join(det_root, n) for n in names if n != held]
        val_dir = os.path.join(det_root, held)
        reg = Registry()
        cur = reg.deployed("detector")
        args = ["tools/train_detector.py", "--train-dirs", *train_dirs, "--val-dirs", val_dir, "--epochs",
                a.det_epochs, "--device", device, "--name", "my_games", "--no-register"]
        if cur and os.path.exists(cur["path"] + ".pt"):
            args += ["--init", cur["path"] + ".pt", "--lr", "1e-3"]
        run(args)
        from fctac.config import AppConfig
        from fctac.training.det_dataset import list_images
        from fctac.training.det_eval import evaluate_on_labels, evaluate_onnx_on_labels
        from fctac.vision.detector import ColorDetector
        val = list_images(val_dir)
        new_path = os.path.join(ROOT, "runs", "detector", "my_games", "detector.onnx")
        mn = evaluate_onnx_on_labels(new_path, val)
        mc = evaluate_on_labels(ColorDetector(AppConfig().detector), val)
        mo = evaluate_onnx_on_labels(cur["path"], val) if cur else None
        best_old = max(mc["det_score"], mo["det_score"] if mo else 0.0)
        shutil.copyfile(os.path.join(ROOT, "runs", "detector", "my_games", "best.pt"), new_path + ".pt")
        e = Registry().register("detector", new_path, mn, primary="det_score", data=",".join(vids),
                                extra_files=(".json", ".pt"), better=mn["det_score"] > best_old)
        summary["detector"] = {"colour_detector": mc["det_score"], "current_learned": mo["det_score"] if mo else None,
                               "new": mn["det_score"], "version": e["version"], "switched_on": e["deployed"]}
        if e["deployed"] and mn["det_score"] > mc["det_score"]:
            summary["detector"]["note"] = 'set "runtime": {"detector": "onnx"} in configs/fc27_1440p.json to use it'
        print("detector:", summary["detector"], flush=True)

    with open(os.path.join(work, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print("\nDONE.", json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
