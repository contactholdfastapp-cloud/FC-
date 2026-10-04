"""Measure FC 27's frame rate with and without the assistant (Windows).

Uses Intel PresentMon (download PresentMon-*.exe from GitHub and pass
--presentmon path).  PresentMon reads Windows ETW present events; it does not
touch the game.  Run as administrator.

    python tools/game_fps.py --presentmon C:\\tools\\PresentMon.exe --seconds 60 --label baseline
    (start: python -m fctac.live)
    python tools/game_fps.py --presentmon C:\\tools\\PresentMon.exe --seconds 60 --label assistant

Then compare benchmarks/game_fps_*.json (avg fps, 1% low, frame-time p99).
GPU utilisation / VRAM are sampled alongside when nvidia-smi or pynvml exist.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
import threading
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fctac.benchmark.sysmon import SysMonitor  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--presentmon", required=True)
    ap.add_argument("--process", default="FC27.exe", help="game executable name as shown in Task Manager")
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--label", default="run")
    a = ap.parse_args()
    out_csv = os.path.join(tempfile.gettempdir(), f"presentmon_{a.label}.csv")
    mon = SysMonitor(period_s=1.0)
    samples = []
    stop = threading.Event()

    def poll():
        while not stop.is_set():
            samples.append(mon.sample())
            time.sleep(1.0)
    th = threading.Thread(target=poll, daemon=True)
    th.start()
    cmd = [a.presentmon, "--process_name", a.process, "--output_file", out_csv, "--timed", str(int(a.seconds)),
           "--terminate_after_timed", "--stop_existing_session"]
    subprocess.run(cmd, check=False)
    stop.set()
    ft = []
    with open(out_csv, newline="") as f:
        for row in csv.DictReader(f):
            v = row.get("MsBetweenPresents") or row.get("FrameTime")
            if v:
                ft.append(float(v))
    ft = np.array(ft)
    rep = {"label": a.label, "frames": int(len(ft)),
           "avg_fps": round(1000.0 / ft.mean(), 1) if len(ft) else None,
           "low1_fps": round(1000.0 / np.percentile(ft, 99), 1) if len(ft) else None,
           "frametime_ms_p99": round(float(np.percentile(ft, 99)), 2) if len(ft) else None,
           "gpu_pct_mean": round(float(np.mean([s["gpu_pct"] for s in samples if "gpu_pct" in s])), 1)
           if any("gpu_pct" in s for s in samples) else None,
           "vram_mb_max": max((s["vram_used_mb"] for s in samples if "vram_used_mb" in s), default=None),
           "cpu_pct_mean": round(float(np.mean([s["cpu_pct"] for s in samples if "cpu_pct" in s])), 1)
           if any("cpu_pct" in s for s in samples) else None}
    os.makedirs("benchmarks", exist_ok=True)
    path = os.path.join("benchmarks", f"game_fps_{a.label}.json")
    json.dump(rep, open(path, "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
