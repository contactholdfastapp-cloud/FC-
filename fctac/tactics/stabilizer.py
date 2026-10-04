"""Temporal stabilisation: decisive, non-flickering recommendations.

Rules
1. Context change (possession phase or controlled player) -> re-decide now.
2. The current action is refreshed every update (players move).  It is
   dropped immediately if it disappears or its success probability collapses
   (lane closed) -- stale advice is never kept.
3. A different action replaces the current one only if it beats it by a
   margin (hysteresis), has been best for ``confirm_n`` consecutive updates,
   and the current one has lived ``min_hold_s``.
4. Confidence hysteresis: appear above ``conf_on``, disappear below
   ``conf_off``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from fctac import types as T


@dataclass
class StabilizerConfig:
    min_hold_s: float = 0.40
    confirm_n: int = 2
    margin_abs_attack: float = 0.006      # goal units
    margin_rel_attack: float = 0.10
    margin_abs_defence: float = 0.10
    conf_on: float = 0.40
    conf_off: float = 0.28
    invalidate_p: float = 0.22            # pass success prob below which the lane is "closed"


class Stabilizer:
    def __init__(self, cfg: StabilizerConfig = StabilizerConfig()):
        self.cfg = cfg
        self.reset()

    def reset(self):
        self.current: Optional[T.Action] = None
        self.since_t = 0.0
        self.context = None
        self.challenger = None
        self.challenger_n = 0
        self.visible = False
        self.switches = 0

    def _margin(self, cur: T.Action) -> float:
        if cur.kind in T.DEFENCE_KINDS:
            return self.cfg.margin_abs_defence
        return max(self.cfg.margin_abs_attack, self.cfg.margin_rel_attack * abs(cur.score))

    def _accept(self, a: T.Action, t: float):
        if self.current is None or self.current.key() != a.key():
            self.switches += 1
            self.since_t = t
        self.current = a
        self.challenger, self.challenger_n = None, 0

    def update(self, t: float, ranked: list, confidence: float, context) -> T.Recommendation:
        cfg = self.cfg
        if not ranked:
            self.reset_soft(context)
            return T.Recommendation(status=T.STATUS_HIDDEN, reason="no candidates")
        best = ranked[0]
        if context != self.context:
            self.context = context
            self.current = None
            self.visible = False
        if self.current is None:
            self._accept(best, t)
        else:
            same = next((a for a in ranked if a.key() == self.current.key()), None)
            closed = same is None or (same.kind in T.PASS_KINDS and same.p_success < cfg.invalidate_p)
            if closed:
                self._accept(best, t)
            else:
                self.current = same           # fresh geometry/score
                if best.key() != same.key():
                    better = best.score > same.score + self._margin(same)
                    if better:
                        if self.challenger is not None and self.challenger == best.key():
                            self.challenger_n += 1
                        else:
                            self.challenger, self.challenger_n = best.key(), 1
                        if self.challenger_n >= cfg.confirm_n and t - self.since_t >= cfg.min_hold_s:
                            self._accept(best, t)
                    else:
                        self.challenger, self.challenger_n = None, 0
                else:
                    self.challenger, self.challenger_n = None, 0
        # confidence of the shown action
        shown = self.current
        if self.visible:
            self.visible = confidence >= cfg.conf_off
        else:
            self.visible = confidence >= cfg.conf_on
        alts = [a for a in ranked if a.key() != shown.key()][:3]
        if not self.visible:
            return T.Recommendation(status=T.STATUS_HIDDEN, action=shown, alternatives=alts,
                                    confidence=confidence, reason="low confidence", since_t=self.since_t)
        return T.Recommendation(status=T.STATUS_ACTIVE, action=shown, alternatives=alts,
                                confidence=confidence, since_t=self.since_t)

    def reset_soft(self, context):
        self.context = context
        self.current = None
        self.visible = False
        self.challenger, self.challenger_n = None, 0
