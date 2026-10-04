# Known limitations and upgrade paths

Honest list of what is not done or not validated yet, and the interface/path
to fix each.

## Not validated on the real target

* **No real FC 27 footage was available during development.** Everything was
  measured on a synthetic FC-style renderer with ground truth
  (`fctac/sim`). FC 27's real HUD (radar position, dot colours, controlled
  markers, ball icon) and real broadcast camera will differ. Path: record a
  session, run `tools/calibrate.py`, check the replay viewer with `G` (debug),
  annotate ~300–1000 frames, retrain the detector, rerun
  `fctac.benchmark.run_all`.
* **Windows-only code was not executed** (development ran in a Linux
  container): `fctac/overlay/win32.py` (layered overlay), `fctac/capture/windows.py`
  (WGC/DXGI/MSS), hotkeys, `tools/inspect_machine.py` Windows branches,
  `tools/game_fps.py`. They use documented APIs but need a first run on your PC:
  `python tools\overlay_demo.py --video ...`, then `python -m fctac.live --preview`.
* **Game FPS impact, GPU utilisation and VRAM were not measured** (no GPU,
  no game here). Use `tools/game_fps.py` (PresentMon) with and without the
  assistant; the live debug panel shows CPU/GPU/VRAM when psutil/pynvml or
  nvidia-smi are available.
* Latency numbers in the README are from a 4-vCPU VM, often under load from
  training jobs, with no GPU. Expect different numbers on your machine.

## Perception

* **Radar colours.** Auto-detection struggles when a team's radar colour is
  close to the ball's (white) or to the controlled marker, or has low contrast
  with the radar background. Set `radar.color_us`/`color_them` explicitly in
  the config in that case. On the white-kit synthetic clip, possession accuracy
  dropped to ~0.69 (vs ~0.88).
* **Ball height is unknown.** Lofted balls are tracked from the radar; the
  main-view ball is only fused when it agrees with the radar.
* **Tight duels.** ID switches happen when two players overlap on the radar
  (~15–25 per minute on synthetic data, mostly in duels). Possession and phase
  are about 87–88 % correct on the clean synthetic clip.
* **Exclusive fullscreen** hides any overlay. Use borderless/windowed.
* The learned detector's ball and controlled-player heads are weak on the
  current synthetic training set (rare classes). The radar provides both, so
  the pipeline does not depend on them.

## Tactics and learning

* Physics constants (pass speed/deceleration, player top speed, reaction time)
  are generic football values, not fitted to FC 27. Fit them from your
  decision datasets. The constants live in `PhysicsConfig`.
* Learned models in this repo were trained on **simulator** data. They show
  the method works (the learned success model is far better calibrated than
  physics; the learned EV ranker beats the heuristic in counterfactual
  simulator rollouts) but they encode the simulator's football, not FC 27's.
  Retrain on your recordings.
* Real recordings only show outcomes of the actions the human chose.
  Rankings learned from them carry selection bias. The simulator
  counterfactual benchmark shows that the proxy metrics can mislead. Use the
  learned ranker with care and keep comparing against the heuristic
  (`runtime.ranker`).
* Defensive recommendations (SWITCH/PRESS/JOCKEY/COVER) are heuristic only.
* Shot placement/power/type are heuristic (`fctac/tactics/shooting.py`).
* Score, clock and formation reading from the HUD are not implemented. The
  formation is inferred from positions.
