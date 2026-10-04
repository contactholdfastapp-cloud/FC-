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
`Q/E` ±5 s, `O` overlay, `G` debug panel, `C` confidence, `2` secondary
action, `H` help. The panel under the video shows the timeline with
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
**no GPU**. Your PC will differ — rerun the benchmarks there. Numbers below are
on synthetic FC-style footage with ground truth, not real FC 27.

| stage | result |
|---|---|
| Radar reader (720p / 1440p) | 98 % player recall, 0.25 / 0.20 m error, ~100 % team, 92–96 % controlled player, 3.3 / 4.9 ms |
| Colour detector (720p) | precision 0.90, recall 0.87–0.89, foot error ~5 px, ~10 ms |
| Learned detector v002 (synthetic val) | F1 0.965 (baseline 0.91), foot error 1.2 px (baseline 5.1), ball recall 0.34 |
| Calibration (radar↔view) | median 0.46 m pitch error, 1.5 ms |
| Full state (clip01) | player position 0.27 m, team 100 %, controlled 96 %, ball 0.24 m, possession 88 %, ~19 ms/frame total |
| Live loop (video as capture, VM under load) | e2e 40 ms mean / 64 ms p95 from frame available to overlay presented; stale frames dropped |
| Decision auto-labelling (perfect state / vision state) | action recall 1.00 / 0.62, precision 0.99 / 0.83 |
| Learned success model vs physics | log-loss 0.25 vs 1.64, ECE 0.02 vs 0.34, AUC 0.87 vs 0.83 |

See `benchmarks/` for full reports. Known limitations are listed in
[docs/LIMITATIONS.md](docs/LIMITATIONS.md).
