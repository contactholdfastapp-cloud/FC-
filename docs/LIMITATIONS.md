# Known limitations and upgrade paths

Honest list of what is not done or not validated yet, and the interface/path
to fix each.

## Not validated on the real target

* **Real FC 27 footage so far = one official EA livestream** (3 matches, 2v2
  co-op, 1080p60, broadcast webcams; YouTube downloads are blocked from the
  development server). The radar reader, calibration and player detector are
  now trained or measured on it (match 3 held out), see
  [REAL_FC27.md](REAL_FC27.md). Not yet seen: **your 1440p screen**, single
  player vs CPU, online 1v1, other camera settings, night or snow matches.
  `scripts/train_my_games.bat` fine-tunes on your own recordings.
* Real-footage metrics are **label-free** (consistency: plausible player
  counts, symbols persisting between frames, radar↔view registration error)
  plus visual checks. There is no hand-labelled FC 27 ground truth yet. The
  synthetic numbers are exact but come from rendered data.
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

* **FC 27 radar.** Teams are read by shape (triangles vs circles), so kit
  colours don't matter. Controlled-player highlights are found about half the
  time (synthetic recall ~0.55), and the tracker smooths over the gaps.
  Overlapping symbols in tight duels can merge (exactly 11 + 11 symbols are
  read in ~29 % of real radar frames, 20–22 in ~84 %). When the radar is
  faded (action behind it, set pieces) it is read less reliably.
* **Which team is yours** is decided from the highlighted player when only
  one team shows a highlight (vs CPU). In online 1v1 or co-op both teams show
  highlights; press F7 if the assistant advises the wrong team, or set
  `radar.us_shape` to `triangle`/`circle`.
* **Old colour radar reader** (`radar.mode = "color"`) is only for the
  synthetic renderer. It fails on the real FC 27 radar.
* **Ball height is unknown.** Lofted balls are tracked from the radar; the
  main-view ball is only fused when it agrees with the radar.
* **Tight duels.** ID switches happen when two players overlap on the radar
  (~15–25 per minute on synthetic data, mostly in duels). Possession and phase
  are about 85–88 % correct on the clean synthetic clips (720p and 1440p).
* **Without the radar**, calibration is kept alive by pitch-line tracking
  (0.2 m over 15 s from one seed) and players are tracked from the main view
  (~1 m error, visible players only). Ball tracking in this mode is not yet
  robust: it can lose lofted balls and lock onto white markings. A plausibility
  guard (a slow ball with nobody within 8 m) then switches to ANALYSING instead
  of advising. Keep the radar on.
* **Exclusive fullscreen** hides any overlay. Use borderless/windowed.
* The learned detector's ball head is weak on the current synthetic training
  set (recall ~0.33; the controlled-player head reaches 0.93 since it targets
  the marker above the head). The radar provides the ball, so the pipeline does
  not depend on it.

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
