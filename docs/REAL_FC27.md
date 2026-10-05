# Training on real FC 27 footage

## Where the footage came from

* YouTube refuses video downloads from this development server ("confirm
  you're not a bot"). That block was **not** worked around.
* EA's own Twitch channel (`twitch.tv/easportsfc`) hosts the official
  **"EA SPORTS FC 27 | First Gameplay Livestream"** VOD. It was downloaded at
  source quality (1080p60) through Twitch's public HLS playlist. The 3 matches
  downloaded are 36 minutes of 2v2 co-op match footage, of which about 12
  minutes are live play with the radar on screen. The rest is replays,
  close-ups, set-piece cameras and studio cuts. It is not committed (`data/real/` is gitignored).
* EA's other FC 27 streams so far (FC Direct developer update and Q&A) only
  show studio cameras and untextured test builds. They were scanned and not
  used. The FC Pro tournament VODs on the same channel are **FC 26**, so they
  were not used either.
* Leakage-safe split: matches 1 and 2 are for training, **match 3 is held out**
  for every number below.

`tools/harvest_frames.py` decodes the videos and keeps every 30th frame
(4,495 frames) and every 12th radar crop (13,436 crops), with a gameplay
index (`index.csv`).

## What the real FC 27 HUD looks like (measured)

* **Radar panel**: a dark, semi-transparent rectangle (the main view shows
  through at about 60 %). In normalised screen coordinates it covers
  x 0.4234–0.5750 and y 0.8009–0.9491 (813–1104 × 865–1025 px at 1080p). The
  panel *is* the pitch: goal lines at its left/right edges, touchlines at its
  top/bottom edges. The pitch lines are drawn stylised (the centre circle is
  a true circle on screen). Goals are short white bars just outside the panel.
  A white bar with dark segments sits under the radar.
* **Teams are shapes**: one team is drawn as **triangles**, the other as
  **circles**. Colours and fill depend on the match: cream triangles vs dark
  circles with a white ring, hollow peach triangles vs filled pale circles,
  light-blue triangles vs hollow white rings. So colour rules cannot work in
  general, but shape always does.
* **Controlled players** (every human in co-op) get a coloured outline, fill
  or ring (pink, magenta, yellow, red, orange).
* **The ball** is an orange-yellow "+".
* The radar **fades out** (no panel, faint symbols) when the action is behind
  it and at set pieces with close cameras. It disappears during replays and
  cut-scenes.
* Main view: small coloured triangles above the heads of human-controlled
  players. Player name and stamina panels sit at the bottom left and right.

The old colour/dot radar reader (built for the synthetic renderer) fails on
this. On held-out match 3 it reads exactly 11 + 11 players in 0.2 % of radar
frames, mixes the teams, and takes white rings for the ball.

## FC 27 radar reader (learned)

* `fctac/sim/radar_fc27.py` renders FC 27-style radars from simulator
  positions on top of **real main-view backgrounds** from matches 1–2. Styles
  are randomised: filled, hollow and ringed symbols, any colours, highlights,
  glow, faded radar, no radar, blur, video chroma subsampling, JPEG. Every
  render has exact labels; 40,000 were generated.
* `TinyRadarNet` (27 k parameters) reads a canonical 320×192 radar crop. It
  outputs heatmaps for triangle centres, circle centres, the ball and
  highlighted players.
* The model trained on renders labels the real match 1–2 radars. Only
  confident, temporally stable crops are kept, with uncertain spots masked
  out. It is then fine-tuned on renders plus these real crops.
* Panel auto-alignment: at start-up the reader matches the pitch-line layout
  in the first radar frames. This corrects a HUD that sits a few pixels
  elsewhere (tested: shifts of up to 9 px recovered exactly, ±5 % scale to
  within 2 px; works on 1440p).
* "Your team": with `radar.us_shape = "auto"` the shape whose player is
  highlighted is yours (the normal case against the CPU). If both teams show
  highlights (online 1v1, co-op), the last decision is kept. **F7** swaps it
  live. The debug panel (F9) shows the current choice.

### Results

Synthetic validation (1,500 rendered radars with exact labels, all styles
including faded and no-radar):

| | v1 (renders only) |
|---|---|
| player precision / recall | 0.946 / 0.951 |
| team shape correct | 99.4 % |
| position error | 0.50 px = **0.18 m** |
| ball recall / precision | 0.988 / 0.995 |
| highlighted (controlled) precision / recall | 0.51 / 0.58 |
| time per radar read (ONNX, CPU, 1 thread) | ~4 ms (~1 ms on a GPU) |

**Real held-out match 3** (1,336 frames with the radar panel). There are no
hand labels, so these are consistency checks: a correct reader sees 22
players, never more than 11 per team, the same symbols 0.2 s later, and the
ball.

| | old colour reader | FC 27 reader v1 |
|---|---|---|
| exactly 11 + 11 players | 0.2 % | **28.7 %** |
| 20–22 players | 41.8 % | **84.1 %** |
| no team above 11 | 41.0 % | **88.2 %** |
| symbols persist 0.2 s later | 88.9 % | **96.0 %** |
| ball seen | 85.5 % (often a white ring, not the ball) | **98.4 %** |
| combined consistency score | 0.60 | **0.90** |

Visual check of random match-3 frames (`runs/radar/qa_v1_m3.jpg`): shapes
are right on almost every symbol, and the ball marker sits on the "+". The
remaining errors are a rare false symbol on a player walking behind the radar
and merged symbols in tight clusters.

**v2 (deployed)**: v1 fine-tuned on renders plus the 3,290 real match 1–2
crops it labelled itself. Synthetic validation is unchanged (F1 0.947, shape
99.4 %, ball 98.8 %). On held-out match 3 the consistency score goes from
0.902 to **0.905**: 20–22 players in 85.3 % (from 84.1 %), exactly 11 + 11 in
29.5 % (from 28.7 %), ball 98.1 % (from 98.4 %). That is a small gain, within
noise. Self-training on one livestream cannot teach much that is new. Your
own games (different kits, radar colours, 1440p) are where real-data
fine-tuning should pay off (`scripts/train_my_games.bat`).

Registry: `radar_v001` (renders only) and `radar_v002` (deployed), with their
metrics in `models/registry.json`.

## Camera and calibration on real FC 27

* FC 27's broadcast camera (default "EA SPORTS GameCam" as used on the
  stream) is a **dolly camera**. It slides along the touchline about 2–3 m
  behind the ball, about 50 m back from the near touchline and about 27 m up.
  It looks straight across (yaw ≈ 0) with tilt ≈ 0.34 rad and focal length
  ≈ 1.65 × image width. This was fitted to radar-registered homographies; the
  reprojection RMS is 0.1–5 px.
* The previous initial search assumed a fixed camera that pans and locked on
  in only 2 of 19 real gameplay frames. The dolly search seeded at the radar
  ball locks on in **19 of 19**, in 8 ms instead of 32 ms
  (`calib.init_mode = "dolly"` in `configs/fc27_1440p.json`).
* The radar is linear in pitch metres. Real pitch-line pixels mapped through
  the radar registration land within 0.4–0.9 m of their true positions:
  halfway line 52.9 m (true 52.5), penalty boxes 16.4 / 89.4 m
  (16.5 / 88.5), box sides 13.4 / 54.1 m (13.8 / 54.2).
* Two crashes that only real footage triggered were fixed (stale matches
  after a failed registration; a singular homography in the line tracker).
  If line tracking holds a confident but wrong lock, the calibrator now
  re-acquires from the radar.

## Players in the main view

Main-view players were labelled automatically on real frames:

* Frames where the radar panel is visible are run through the full pipeline.
* When the radar registration is fresh (at least 7 inliers, under 0.9 m),
  every detection that lands on a radar player becomes a labelled foot point.
* Radar players the colour detector missed (e.g. a black kit on dark grass)
  become labels at their projected position, but only inside the area the
  registration covers and only where the image shows a person. Everything else
  uncertain is an *ignore* region: the referee (not on the radar), players
  outside the registered area, and a ball not confirmed by both radar and
  image.
* Broadcast webcams and the HUD are ignored too. Controlled-player flags are
  not trusted in this 2v2 co-op footage (4–5 markers per frame), so that head
  is trained on synthetic data only.

Result: 869 labelled training frames (matches 1–2, about 15 players per
frame) and 170 test frames (match 3, held out).

**Held-out match 3** (`tools/eval_detectors.py`, players matched within 2 %
of the image width, ignore regions excluded):

| detector | player precision / recall / F1 | ball recall / precision | foot error | ms (CPU, 1 thread) |
|---|---|---|---|---|
| colour detector (baseline) | 0.83 / 0.73 / 0.774 | 0.18 / 0.15 (full-frame search) | 1.3 px* | 14–33 |
| learned v003 (synthetic only) | 0.15 / 0.25 / 0.19 | 0 / 0 | 13 px | 17 |
| **learned v004 (real frames + synthetic)** | 0.76 / **0.80** / **0.778** | **0.22 / 0.32** | 7.7 px | 18 |

\* The labels partly come from the colour detector, so its precision and
foot error are flattered on this test.

The decisive test is the full pipeline. With v004, the screen↔pitch
calibration is available in **97 %** of live play on match 3, against 57 %
with the colour detector, which misses the dark kits, and it is faster. v004
is registered as the deployed detector, and `configs/fc27_1440p.json` uses
it (`runtime.detector = "onnx"`).

## End to end on real footage (4 minutes per match, every 2nd frame)

| | match 3 (held out) | match 1 |
|---|---|---|
| share of time with live play (radar panel visible) | 49 % | 32 % |
| live play: calibration available | **85 %** | **94 %** |
| live play: controlled player known | 98 % | 94 % |
| live play: advice shown | 52 % | 61 % |
| radar ↔ main-view registration error (median) | 0.50 m | 0.44 m |

When no advice is shown it is mostly on purpose: we are attacking but the
controlled player does not have the ball, or there is a loose ball. Replays,
close-ups and set-piece cameras show ANALYSING.

Fixes found only by running on real footage:

* **Kit colours guessed during replays could invert the teams.** Every radar
  registration then failed for the rest of the play, calibrated in 2 % of
  frames. Registration now retries without kit teams and resets the kit
  model on conflict: 98 %.
* **Line tracking could overwrite fresh radar fits.** A fresh radar fit now
  always wins.
* **Confidence over-penalised zoomed cameras** with only 7–10 visible
  players.
* **ONNX Runtime thread spinning** made the two models fight for the CPU.
  Spinning is now off: the pipeline went from 40 to **28.6 ms per frame
  (p95 52 ms)** on a 4-vCPU VM with no GPU, at 1080p. On a GPU both models
  take about 1–3 ms.

Watch it: `python tools/render_demo.py --video data/real/m1.mp4 --start 18 --seconds 24 --debug --out demo.mp4`.

## Use your own games (best data)

The livestream is 1080p, 2v2 co-op, with broadcast webcams. Your own 1440p
games are better training data:

1. Record one or more full matches with the 2D radar on (Kick-Off or Squad
   Battles vs the CPU is ideal: only your team shows a controlled player)
   (`python tools\record.py --out data\recordings\game1 --minutes 10`, or OBS
   or ShadowPlay at 1440p60).
2. Drag the recordings onto `scripts\train_my_games.bat` (or run
   `python tools\train_on_my_games.py game1.mp4 game2.mp4`).
3. It auto-labels your radar and players, fine-tunes both models, and
   compares new vs current on held-out footage (the last recording, or the
   last quarter of a single one). A new model is only switched
   on if it measured better. Results go to `data\my_games\summary.json`, and a
   radar QA picture to `data\my_games\radar_qa.jpg`.
