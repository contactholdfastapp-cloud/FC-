# Training on real FC 27 footage

## Where the footage came from

* YouTube refuses video downloads from this development server ("confirm
  you're not a bot"). That block was **not** worked around.
* EA's own Twitch channel (`twitch.tv/easportsfc`) hosts the official
  **"EA SPORTS FC 27 | First Gameplay Livestream"** VOD. It was downloaded at
  source quality (1080p60) through Twitch's public HLS playlist. It contains
  three 2v2 co-op matches, about 36 minutes of play in total, plus replays,
  close-ups and studio cuts. It is not committed (`data/real/` is gitignored).
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

RESULTS_RADAR

## Players in the main view

RESULTS_DETECTOR

## Use your own games (best data)

The livestream is 1080p, 2v2 co-op, with broadcast webcams. Your own 1440p
games are better training data:

1. Record 2 or more games (10+ minutes each) with the 2D radar on
   (`python tools\record.py --out data\recordings\game1 --minutes 10`, or OBS
   or ShadowPlay at 1440p60).
2. Drag the recordings onto `scripts\train_my_games.bat` (or run
   `python tools\train_on_my_games.py game1.mp4 game2.mp4`).
3. It auto-labels your radar and players, fine-tunes both models, and
   compares new vs current on the last recording. A new model is only switched
   on if it measured better. Results go to `data\my_games\summary.json`, and a
   radar QA picture to `data\my_games\radar_qa.jpg`.
