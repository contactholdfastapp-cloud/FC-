# How the assistant decides (and what changed)

Every frame the assistant lists the realistic options for the controlled
player: pass to each teammate, through ball (ground or lofted) into each
runner's path, lob or cross, shot, dribble and hold. It scores each one as

    expected value = P(success) × value if it works − P(lose it) × cost of losing it there

and shows the best one. Values are in "goal units" (probability that the
possession ends in a goal).

## Sources

* **Value of a position:** Karun Singh's open *Expected Threat (xT)* grid,
  12×8 zones, fitted on real match event data
  (`fctac/tactics/data/open_xt_12x8_v1.json`), interpolated smoothly. Inside
  the box it never drops below a fraction of a direct shot's xG.
* **Will the pass arrive?** A pitch-control race (Spearman-style): the ball
  travels with realistic speed and deceleration. Every player reacts after
  0.3 s and *accelerates* (5.5 m/s² to 7.8 m/s, keeping the momentum they
  already have). Whoever reaches the ball first along its path wins it. A
  receiver who is already there is not "intercepted" by a defender who could
  merely also get there; that becomes a duel won by position.
* **FC pro guidance:**
  * Through balls go into space behind the line or between centre-back and
    full-back, aimed along the runner's path, never into a crowded box.
  * Lobbed through balls beat a high line or a defender in the ground lane.
  * Long diagonal switches (up to about 60 m) are allowed.
  * You control the ball carrier when your team has the ball (auto-switch).
* **FC 27 changes:**
  * Through and lobbed through balls are much less assisted, so passes
    carry an execution error that grows with distance.
  * AI teammates win the ball less on their own, so the assistant tells you
    to switch to the right defender earlier.

## Football exam

`python tools/football_exam.py` runs 17 classic situations:
* the counter-attack through ball;
* the lofted through ball over a high line;
* never passing to an offside player;
* shooting close and central, and not shooting from distance into a crowd;
* releasing the ball under pressure, and avoiding a blocked lane;
* switching play away from an overload;
* driving on when through on goal;
* the cross or cut-back from the byline;
* no square ball across your own box;
* the runner over a marked striker;
* advising the ball carrier;
* defending: switch, jockey, press with support, and cover the runner.

| | before | now |
|---|---|---|
| exam | 11 / 16 | **17 / 17** |
| same situations mirrored + positions nudged ±1 m | – | 94 % |
| simulator counterfactual (76 decisions) | 0.0184 ± 0.005 | 0.0195 ± 0.005 (level) |

Bugs fixed on the way:
* Defenders could intercept a pass at a receiver who was already standing
  there (100 % "interception" on easy passes).
* Defenders went from standing to sprinting instantly.
* Switches of play over 50 m were never considered.
* Dribbling into a crowded corner looked safe.
* "Attacking off the ball" silenced advice while your teammate, whom FC
  gives you, had the ball.
