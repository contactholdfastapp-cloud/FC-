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

With 2+ recordings the last one is held out for the comparison; with a
single recording its last quarter is held out.  A full match (10+ minutes of
gameplay, 2D radar on) works well.  With a CUDA GPU the training steps take
minutes; on CPU expect an hour or more.
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


def _safe_name(path: str) -> str:
    import re
    return re.sub(r"[^A-Za-z0-9_-]+", "_", os.path.splitext(os.path.basename(path))[0])[:60] or "game"


def _split_single_harvest(harvest: str, name: str, held: str, frac: float):
    """One recording: its last `frac` (by frame index) becomes the held-out group."""
    import csv
    p = os.path.join(harvest, "index.csv")
    rows = list(csv.DictReader(open(p)))
    # split by *live play* (radar panel visible), so menus/replays at the end don't eat the test set
    live = sorted(int(r["frame"]) for r in rows if r["group"] == name and float(r["radar_line"]) > 20)
    frames = live or sorted(int(r["frame"]) for r in rows if r["group"] == name)
    if not frames:
        return
    cut = frames[int(len(frames) * (1.0 - frac))]
    for r in rows:
        if r["group"] == name and int(r["frame"]) >= cut:
            r["group"] = held
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _split_single_labels(src: str, dst: str, frac: float):
    from fctac.training.det_dataset import list_images, load_label
    imgs = list_images(src)
    if not imgs:
        return
    fr = {p: int(load_label(p).get("frame", 0)) for p in imgs}
    ks = sorted(fr.values())
    cut = ks[int(len(ks) * (1.0 - frac))]
    os.makedirs(dst, exist_ok=True)
    for p, k in fr.items():
        if k >= cut:
            for q in (p, p + ".json"):
                shutil.move(q, os.path.join(dst, os.path.basename(q)))


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
    ap.add_argument("--held-fraction", type=float, default=0.25, help="single recording: share held out (its end)")
    ap.add_argument("--dry-run", action="store_true", help="train and compare, but do not touch models/registry.json")
    a = ap.parse_args()
    vids = [os.path.abspath(v) for v in a.videos]
    for v in vids:
        if not os.path.exists(v):
            raise SystemExit(f"video not found: {v}")
    work = os.path.join(ROOT, a.work)
    harvest = os.path.join(work, "harvest")
    names = [_safe_name(v) for v in vids]
    if len(set(names)) != len(names):
        raise SystemExit("two recordings have the same file name - rename one")
    single = len(vids) == 1
    held = names[-1] + "_held" if single else names[-1]
    train_names = names if single else names[:-1]
    from fctac.training.registry import Registry
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    summary = {}

    # 1. harvest (file name -> group, so the held-out recording stays separate)
    if not os.path.exists(os.path.join(harvest, "index.csv")):
        links = os.path.join(work, "videos")
        os.makedirs(links, exist_ok=True)
        named = []
        for v, n in zip(vids, names):              # group = cleaned file name
            dst = os.path.join(links, n + os.path.splitext(v)[1].lower())
            if not os.path.exists(dst):
                try:
                    os.link(v, dst)                # same drive: no copy
                except OSError:
                    shutil.copyfile(v, dst)
            named.append(dst)
        run(["tools/harvest_frames.py", *named, "--out", harvest])
        if single:
            _split_single_harvest(harvest, names[0], held, a.held_fraction)

    # 2. radar
    if not a.skip_radar:
        reg = Registry()
        cur = reg.deployed("radar")
        if cur is None:
            raise SystemExit("no radar model deployed (models/registry.json)")
        train_groups = ",".join(train_names)
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
        if not mn.get("panel_crops"):
            print("not enough held-out radar footage to compare -> keeping the current radar model", flush=True)
            sn = so = 0.0
        metrics = {**{f"real_{k}": v for k, v in mn.items() if isinstance(v, (int, float))}, "radar_score": round(sn, 4)}
        if a.dry_run:
            summary["radar"] = {"current": round(so, 4), "new": round(sn, 4), "would_switch_on": sn > so}
        else:
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
        if single and not os.path.exists(os.path.join(det_root, held)):
            _split_single_labels(os.path.join(det_root, names[0]), os.path.join(det_root, held), a.held_fraction)
        train_dirs = [os.path.join(det_root, n) for n in train_names]
        val_dir = os.path.join(det_root, held)
        reg = Registry()
        cur = reg.deployed("detector")
        args = ["tools/train_detector.py", "--train-dirs", *train_dirs, "--val-dirs", val_dir, "--epochs",
                a.det_epochs, "--device", device, "--name", "my_games", "--no-register"]
        if cur and os.path.exists(cur["path"] + ".pt"):
            args += ["--init", cur["path"] + ".pt", "--lr", "1e-3"]
        if device == "cpu":
            args += ["--crop", "192", "320"]          # ~4x faster without a GPU
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
        summary["detector"] = {"colour_detector": mc["det_score"], "current_learned": mo["det_score"] if mo else None,
                               "new": mn["det_score"], "labelled_frames_train": sum(len(list_images(d)) for d in train_dirs),
                               "labelled_frames_test": len(val)}
        if a.dry_run:
            summary["detector"]["would_switch_on"] = mn["det_score"] > best_old
        else:
            e = Registry().register("detector", new_path, mn, primary="det_score", data=",".join(vids),
                                    extra_files=(".json", ".pt"), better=mn["det_score"] > best_old)
            summary["detector"].update(version=e["version"], switched_on=e["deployed"])
        print("detector:", summary["detector"], flush=True)

    verdict = []
    for k, label in (("radar", "radar reader"), ("detector", "player detector")):
        v = summary.get(k)
        if v is None:
            continue
        on = v.get("switched_on", v.get("would_switch_on"))
        verdict.append(f"{label}: {'NEW model is better -> switched on' if on else 'current model kept (new one was not better)'}")
    summary["verdict"] = verdict
    with open(os.path.join(work, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print("\nDONE.", json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
