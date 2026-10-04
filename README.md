# FCTAC — local real-time tactical vision for FC 27

A local Windows assistant that watches visible FC 27 gameplay and shows one
clear recommendation (`PASS → ST`, `THROUGH → RW`, `SHOOT ↗ FAR POST`,
`SWITCH → RCB`, `PRESS`, …) as a transparent overlay.

* **No cloud, no LLM at runtime.** Pixels → classical CV + small local models →
  physics/value-based tactics → overlay. Everything runs on your PC.
* **Read-only.** It uses standard screen capture and an ordinary overlay window.
  It never touches game memory or files, never sends input, and never hides
  from anti-cheat. See [docs/SAFETY.md](docs/SAFETY.md). EA's rules decide
  whether using it is allowed; replay mode on your own recordings is the
  low-risk way to use it.
* **Fresh over complete.** The live path has no queues: newest frame in,
  stale frames dropped, latency measured per frame.

Architecture: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). Dependencies and
the reason for each: [docs/DEPENDENCIES.md](docs/DEPENDENCIES.md).

## Status — be clear about what is validated

| part | state |
|---|---|
| Replay viewer + overlay renderer | working, tested |
| Radar reader, colour detector, calibration, tracker, game state | working, measured on **synthetic** FC-style footage; needs tuning on real FC 27 footage (HUD colours, radar position) |
| Learned detector (ONNX) | training, export and inference pipeline working; current weights are trained on synthetic frames only — **retrain on your annotated FC 27 frames** |
| Tactics: candidates, pass/through-ball physics, shooting, defending, stabiliser | working (heuristic baseline) |
| Decision auto-labelling, learned success/ranking model, learned movement predictor | pipelines working and benchmarked on simulator data; must be retrained on your recordings |
| Live capture (WGC/DXGI) + Win32 transparent overlay | written against the Windows APIs; **not yet run on Windows** (the development machine is a Linux container) — first thing to test on your PC |

## Quick start (Windows, Python 3.10+)

`scripts\setup_windows.bat` does all of this; `scripts\replay.bat` and
`scripts\live.bat` are shortcuts. Manually:

```bat
python -m venv .venv && .venv\Scripts\activate
pip install -e .[dev]
pip install onnxruntime-gpu        :: or onnxruntime-directml (any GPU) / onnxruntime (CPU)
pip install windows-capture dxcam  :: live capture backends

python tools\inspect_machine.py            :: Phase 0: what this machine is, recommended runtime
python -m pytest -q                        :: test suite (~20 s)
```

### 1. Try it on synthetic footage (no game needed)

```bat
python tools\make_synthetic.py --seconds 60 --out data\synthetic\clip01
python -m fctac.replay.viewer --video data\synthetic\clip01.mp4            :: vision pipeline
python -m fctac.replay.viewer --video data\synthetic\clip01.mp4 --oracle   :: perfect-state tactics
python tools\overlay_demo.py --video data\synthetic\clip01.mp4             :: real transparent overlay over a window
```

Replay keys: `SPACE` play/pause, `A/D` or arrows step, `W/S` speed (0.1×–2×),
`Q/E` ±5 s, `O` overlay, `G` debug panel, `T` tracking view (persistent ids,
roles, teams, ball owner), `C` confidence, `2` secondary action, `H` help. The panel under the video shows the timeline with
recommendation changes (ticks) and the human's actions (triangles, red =
failed), the current recommendation, alternatives, and the next human action
with its outcome.

### 2. Your FC 27 footage (replay mode first)

1. FC 27 in **borderless/windowed** mode at 2560×1440, radar (2D) visible.
2. Record: `python tools\record.py --out data\recordings\s01 --minutes 10`
   (or OBS/ShadowPlay at 1440p60, high bitrate).
3. Set up once: `python tools\calibrate.py --video data\recordings\s01.mp4 --out configs\my_calib.json`
   (drag the radar box; optionally click pitch landmarks; set attack direction).
4. Replay: `python -m fctac.replay.viewer --video data\recordings\s01.mp4 --config configs\fc27_1440p.json --calib configs\my_calib.json`

### 3. Live

```bat
python -m fctac.live --config configs\fc27_1440p.json --calib configs\my_calib.json
:: F8 overlay on/off, F9 debug panel, F10 quit.   --preview shows a debug window instead
```

## Improving the models with your own games

```bat
python tools\extract_frames.py data\recordings\s01.mp4 --out data\datasets\fc27 --every 15
python tools\annotate.py data\datasets\fc27            :: correct the pre-annotations, V = verified
python tools\train_detector.py --data data\datasets\fc27 --epochs 60 --device cuda
python tools\bench_models.py --model models\detector_v00N.onnx --data data\datasets\fc27   :: FP32/FP16/INT8 x providers

python tools\build_decisions.py --videos data\recordings\s01.mp4 --out data\decisions
python -m fctac.training.ranker --data data\decisions --glob "*.vision.jsonl" --test-clips s01.mp4 --register
python -m fctac.training.predictor --register

python -m fctac.benchmark.run_all --clips data\recordings\annotated01 --labels data\datasets\fc27
python tools\game_fps.py --presentmon C:\tools\PresentMon.exe --label baseline   :: then again with the assistant
```

Every trained model goes into `models/registry.json` with its benchmark, and is
deployed only if it beats the current one (`runtime.ranker` /
`runtime.predictor` = `registry`, `runtime.detector` = `onnx`).

## Measured results so far (development container, CPU only, synthetic footage)

Development machine: Linux container, 4 vCPU Xeon @ 2.1 GHz, 15.7 GB RAM,
**no GPU**. Your PC will differ — rerun `python -m fctac.benchmark.run_all`
there. Everything below is on synthetic FC-style footage with ground truth,
**not real FC 27**. Latency rows are from an idle machine; full report:
[benchmarks/20261004_135257.md](benchmarks/20261004_135257.md).

| stage | result |
|---|---|
| **Live loop, 1440p** (video file as capture, `configs/fc27_1440p.json`) | **20.5 ms mean / 43 ms p95** frame-available → overlay ready; 449/451 frames processed at 30 fps (720p: 18.1 / 39 ms) |
| Full state at 1440p vs ground truth | player position 0.23 m, team 100 %, controlled player 95 %, ball 0.29 m, calibration 0.53 m, possession 85 %, 16.8 ms/frame mean |
| Radar reader (720p / 1440p) | 98 % player recall, 0.25 / 0.20 m error, ~100 % team, 92–96 % controlled player, 3.3 / 4.9 ms |
| Colour detector vs learned detector v003 (synthetic val) | F1 0.91 vs 0.95, foot error 5.1 vs 1.2 px, controlled player 0.58 vs 0.93, 12.2 vs 9.2 ms (CPU, ONNX Runtime) |
| Calibration: radar↔view registration / pitch-line tracking (no radar) | 0.46–0.53 m / 0.21 m median (line tracking from one seed, 15 s) |
| Overlay render at 1440p | 1.4 ms mean (2.3 p95); 3.1 ms with debug panel |
| Decision auto-labelling (perfect state / vision state) | action recall 1.00 / 0.62, precision 0.99 / 0.83 |
| Learned success model vs physics (held-out sim match) | log-loss 0.25 vs 1.88, ECE 0.013 vs 0.49, AUC 0.88 vs 0.81 |
| Ranking — counterfactual simulator test (313 paired decisions, recommendation actually executed, 3 rollouts each) | learned EV ranker **+0.0129 ± 0.0036** goal-units/decision over the heuristic (3.5σ); heuristic is still below the simulator's own policy ([docs/results](docs/results)) |
| Movement prediction (held-out sim match) | learned MLP 0.95 m vs damped-velocity 1.22 m mean error (1.5 s: 2.36 vs 2.95 m) |
| `detect_every=2` (radar + tracker bridge the gaps) | same accuracy, ~25 % less per-frame cost → default at 1440p |

Learned models in `models/` (registry: detector_v003, ranker_v001,
predictor_v001) were trained on synthetic/simulator data to prove the
pipeline. The default config therefore keeps the colour detector, the
heuristic ranker and the kinematic predictor. Switch to the learned models with
`--ranker registry`, `--predictor registry` and `runtime.detector = "onnx"`
after retraining on your FC 27 recordings.

Raw result files: `docs/results/`; new runs of `fctac.benchmark.run_all` go to `benchmarks/`. Known limitations are listed in
[docs/LIMITATIONS.md](docs/LIMITATIONS.md).
