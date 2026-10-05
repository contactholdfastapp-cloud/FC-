# Architecture

```
FC 27 window pixels
  │  capture thread (WGC window capture | DXGI | MSS | video file)
  ▼
LatestFrameBuffer  ── single slot, newest frame wins, stale frames dropped (counted)
  │
  ▼  analysis loop (one frame at a time, always the newest)
┌─────────────────────────────────────────────────────────────────────────┐
│ radar reader      FC 27 radar -> 22 players (triangles vs circles),     │ ~4 ms CPU 1 thread
│                   ball "+", controlled highlight  (TinyRadarNet CNN)  │ ~1 ms GPU
│ main-view detector colour baseline | ONNX point detector                │ ~10 ms CPU / ~1-3 ms GPU
│ team colours      learned online from radar-registered detections       │ <0.2 ms
│ calibration       homography from radar <-> detection registration      │ ~1.5 ms
│ tracker           Kalman tracks in pitch space, fusing radar + view     │ ~1.5 ms
│ state builder     attack direction, roles/formation, possession, touches│ ~0.7 ms
│ movement predict  damped velocity (baseline) | learned MLP              │ <0.2 ms
│ tactics           candidates -> physics race model -> EV ranking        │ ~1-3 ms
│ stabiliser        hysteresis, min lifetime, lane-closed invalidation    │
└─────────────────────────────────────────────────────────────────────────┘
  │
  ▼
overlay renderer (premultiplied BGRA, dirty-rect) -> Win32 layered window / preview
```

## Key design decisions

**Radar first.** FC's radar is an orthographic top-down map of every player
and the ball. Reading it costs a few milliseconds of CPU, needs no perspective
calibration, covers off-screen players and gives team labels for free. It is
the backbone of the game state. The main-view detector adds precise screen
positions (for drawing) and refines positions near the ball.

**FC 27 radar = shapes, not colours.** On real FC 27 footage one team is
drawn as triangles and the other as circles, with match-dependent fills and
outlines, so a small CNN reads a canonical 320×192 crop of the radar
(`fctac/vision/radar_fc27.py`). It was trained on FC 27-style renders on real
backgrounds plus self-labelled real radar crops. The panel position is
refined automatically at start-up, and which team is yours comes from the
highlighted player (F7 swaps it). See [REAL_FC27.md](REAL_FC27.md).

**Calibration by registration.** The screen↔pitch homography is estimated each
frame by matching projected detections to radar dots (Hungarian + RANSAC
homography), seeded by a coarse-to-fine grid search over broadcast-camera
poses. FC 27's camera is a dolly that slides along the touchline behind the
ball (measured on real footage), so the search is seeded at the radar ball. It works for any camera style and reports its own confidence. Decisions
need pitch coordinates (from the radar), not the homography. The homography is
only needed to draw arrows, so when it is unreliable only the label is shown.

**Pitch coordinates.** Raw metres (x 0..105 along the touchline, as seen on
screen; y 0..68 from near to far touchline). The tactical layer uses
attack-aligned metres (x = 0 own goal, 105 opponent goal); normalised = /size.

**Candidates, not coordinates.** Tactics generates explicit candidates (PASS,
THROUGH, LOB, CROSS to every teammate; SHOOT; DRIBBLE; HOLD; and SWITCH, PRESS,
JOCKEY, COVER in defence). Each candidate is evaluated with a vectorised
pitch-control style race model: does the ball reach each point of its path
before any opponent; does the receiver reach the end point first. Through
balls search lead times and aim at the **predicted receiving point**, not the
receiver's current position. Every candidate is scored in expected-goal units:
`EV = p_success * value_if_success - (1 - p_success) * turnover_cost`.

**Learned models must earn their place.** Learned components (detector,
success model, policy prior, movement predictor) are versioned in
`models/registry.json` with their benchmark, and deployed only if they beat the
current/baseline model on held-out data.

**Freshness over completeness.** No queues anywhere in the live path. Latency
is measured per frame: frame availability → analysis → overlay present.

## Package map

| package | role |
|---|---|
| `fctac/capture` | latest-frame buffer, Windows capture backends, video sources |
| `fctac/vision` | radar reader, colour detector, ONNX detector, analyzer (full pipeline) |
| `fctac/pitch` | pitch model, camera model, calibration |
| `fctac/tracking` | Kalman filter, multi-object tracker |
| `fctac/state` | GameState builder, formations/roles, oracle state from GT |
| `fctac/prediction` | kinematic and learned movement prediction |
| `fctac/tactics` | value model, physics, candidates, shooting, defending, engine, stabiliser, learned ranker |
| `fctac/overlay` | renderer (premultiplied BGRA) and Win32 layered window |
| `fctac/replay` | replay viewer |
| `fctac/training` | datasets (detection, decisions), training scripts, registry |
| `fctac/benchmark` | evaluations, counterfactual simulator test, system monitor, run_all |
| `fctac/sim` | synthetic match simulator and renderer (development ground truth) |
| `tools/` | inspect_machine, record, calibrate, extract_frames, annotate, train_detector, build_decisions, bench_models, game_fps, overlay_demo, make_synthetic |

## Self-improvement loop

```
record (tools/record.py or OBS, 1440p60)
 → calibrate once (tools/calibrate.py)
 → analyse + pre-annotate (tools/extract_frames.py)  → correct (tools/annotate.py)
 → train detector (tools/train_detector.py)          → registry (deployed only if better)
 → auto-label decisions (tools/build_decisions.py)   → train ranker (python -m fctac.training.ranker)
 → train predictor (python -m fctac.training.predictor)
 → benchmark (python -m fctac.benchmark.run_all)      → versioned report in benchmarks/
```
