"""Football exam: classic situations with a clear right answer (or a clear mistake).

    python tools/football_exam.py            # score the current tactics engine

Each scenario is a hand-built game state (attack-aligned metres: we attack
towards x = 105, our goal is x = 0) plus a check on the engine's top choice.
The situations follow real coaching / FC pro principles: through balls into
runs behind a high line, lofted through balls when the ground lane is
blocked, never pass to an offside player, shoot close and central, don't
shoot from distance through a crowd, release the ball under pressure, switch
play away from an overload, never pass across your own box under pressure,
defend by switching early, jockeying near the box, pressing with support and
covering dangerous runners.  It is a sanity exam, not proof of perfect play.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Callable

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fctac import types as T  # noqa: E402

US, THEM = T.TEAM_US, T.TEAM_THEM


def P(i, team, x, y, vx=0.0, vy=0.0, role="", ctrl=False):
    return T.PlayerState(id=i, team=team, pos=np.array([x, y], float), vel=np.array([vx, vy], float),
                         acc=np.zeros(2), controlled=ctrl, role=role)


def state(players, ctrl_id, ball_owner, attack=True):
    owner = next(p for p in players if p.id == ball_owner)
    st = T.GameState(frame=0, t=0.0, players=players, ball=None, controlled_id=ctrl_id,
                     possession=US if attack else THEM, calib_conf=1.0)
    st.ball = T.BallState(pos=owner.pos + np.array([0.6 if attack else -0.6, 0.0]), vel=owner.vel.copy(),
                          confidence=1.0, owner_id=ball_owner, owner_team=owner.team)
    return st


def gk(i, team):
    return P(i, team, 103.0 if team == THEM else 2.0, 34.0, role="GK")


@dataclass
class Scenario:
    name: str
    why: str
    build: Callable
    check: Callable           # (top action, ranked list, state) -> bool


def target_is(a, pid):
    return a.target_id == pid


def scenarios() -> list:
    S = []

    def a1():
        ps = [P(1, US, 52, 34, role="CM", ctrl=True), P(2, US, 64, 14, 6.5, 0, role="RW"), P(3, US, 63, 34, role="ST"),
              P(4, US, 62, 58, role="LW"), P(5, US, 40, 30, role="CDM"), P(6, US, 30, 34, role="CB"), gk(11, US),
              P(20, THEM, 68, 28, role="CB"), P(21, THEM, 68, 40, role="CB"), P(22, THEM, 67, 18, role="FB"),
              P(23, THEM, 66, 56, role="FB"), P(24, THEM, 64.5, 33, role="CDM"), P(25, THEM, 63.5, 57, role="CM"),
              gk(29, THEM)]
        return state(ps, 1, 1)
    S.append(Scenario("counter_through_ball", "winger running in behind a high line on the counter -> through ball",
                      a1, lambda a, r, s: a.kind in (T.THROUGH, T.LOB) and target_is(a, 2)))

    def a2():
        ps = [P(1, US, 45, 34, role="CM", ctrl=True), P(2, US, 57, 34, 7.0, 0, role="ST"), P(3, US, 40, 12, role="RB"),
              P(4, US, 40, 56, role="LB"), P(5, US, 30, 30, role="CB"), P(6, US, 30, 40, role="CB"), gk(11, US),
              P(20, THEM, 60, 33, role="CB"), P(21, THEM, 60, 41, role="CB"), P(22, THEM, 59, 20, role="FB"),
              P(23, THEM, 59, 52, role="FB"), P(24, THEM, 51, 34.5, role="CDM"), P(25, THEM, 48, 26, role="CM"),
              P(26, THEM, 48, 43, role="CM"), gk(29, THEM)]
        return state(ps, 1, 1)
    S.append(Scenario("lofted_over_high_line", "striker running past a high line, ground lane blocked -> lofted through ball",
                      a2, lambda a, r, s: a.kind in (T.THROUGH, T.LOB) and target_is(a, 2)))

    def a3():
        ps = [P(1, US, 55, 34, role="CM", ctrl=True), P(2, US, 78, 34, role="ST"), P(3, US, 68, 9, role="RW"),
              P(4, US, 45, 50, role="LB"), P(5, US, 35, 30, role="CB"), P(6, US, 45, 20, role="CDM"), gk(11, US),
              P(20, THEM, 72, 30, role="CB"), P(21, THEM, 72, 40, role="CB"), P(22, THEM, 71, 50, role="FB"),
              P(23, THEM, 63, 30, role="CM"), P(24, THEM, 63, 42, role="CM"), gk(29, THEM)]
        return state(ps, 1, 1)
    S.append(Scenario("never_pass_offside", "striker stands beyond the last defender -> never pass to him",
                      a3, lambda a, r, s: not target_is(a, 2)))

    def a4():
        ps = [P(1, US, 93, 31, role="ST", ctrl=True), P(2, US, 80, 50, role="CM"), P(3, US, 75, 20, role="CM"),
              P(4, US, 60, 34, role="CDM"), P(5, US, 40, 30, role="CB"), P(6, US, 40, 40, role="CB"), gk(11, US),
              P(20, THEM, 89, 36, role="CB"), P(21, THEM, 88, 27, role="CB"), P(22, THEM, 85, 45, role="FB"),
              P(23, THEM, 80, 30, role="CM"), gk(29, THEM)]
        return state(ps, 1, 1)
    S.append(Scenario("shoot_when_clear", "12 m out, central, defenders behind -> shoot", a4,
                      lambda a, r, s: a.kind == T.SHOOT))

    def a5():
        ps = [P(1, US, 72, 34, role="CAM", ctrl=True), P(2, US, 80, 8, role="RW"), P(3, US, 70, 58, role="LW"),
              P(4, US, 60, 34, role="CDM"), P(5, US, 45, 30, role="CB"), P(6, US, 45, 40, role="CB"), gk(11, US),
              P(20, THEM, 82, 30, role="CB"), P(21, THEM, 82, 38, role="CB"), P(22, THEM, 85, 34, role="CDM"),
              P(23, THEM, 78, 44, role="CM"), P(24, THEM, 79, 24, role="CM"), gk(29, THEM)]
        return state(ps, 1, 1)
    S.append(Scenario("no_long_shot_into_crowd", "33 m out with a wall of defenders, winger free -> don't shoot", a5,
                      lambda a, r, s: a.kind != T.SHOOT))

    def a6():
        ps = [P(1, US, 30, 34, role="CM", ctrl=True), P(2, US, 22, 18, role="CB"), P(3, US, 40, 52, role="CM"),
              P(4, US, 45, 20, role="RM"), P(5, US, 55, 34, role="ST"), P(6, US, 20, 50, role="CB"), gk(11, US),
              P(20, THEM, 31.6, 34.6, -1, 0, role="ST"), P(21, THEM, 30.2, 32.2, 0, 1, role="CAM"),
              P(22, THEM, 50, 40, role="CM"), P(23, THEM, 60, 30, role="CB"), gk(29, THEM)]
        return state(ps, 1, 1)
    S.append(Scenario("release_under_pressure", "two opponents on top of me in my half -> pass, don't hold or dribble",
                      a6, lambda a, r, s: a.kind in (T.PASS, T.LOB, T.THROUGH)))

    def a7():
        ps = [P(1, US, 50, 34, role="CM", ctrl=True), P(2, US, 65, 34, role="ST"), P(3, US, 60, 12, role="RM"),
              P(4, US, 40, 50, role="LB"), P(5, US, 35, 28, role="CB"), P(6, US, 42, 34, role="CDM"), gk(11, US),
              P(20, THEM, 57, 34, role="CDM"), P(21, THEM, 67, 35, role="CB"), P(22, THEM, 70, 25, role="CB"),
              P(23, THEM, 66, 50, role="FB"), P(24, THEM, 55, 46, role="CM"), gk(29, THEM)]
        return state(ps, 1, 1)
    S.append(Scenario("avoid_blocked_lane", "defender stands in the lane to the striker, winger free -> not the striker",
                      a7, lambda a, r, s: not (target_is(a, 2) and a.kind == T.PASS)))

    def a8():
        ps = [P(1, US, 60, 60, role="RB", ctrl=True), P(2, US, 70, 8, role="LW"), P(3, US, 64, 52, role="RM"),
              P(4, US, 50, 40, role="CDM"), P(5, US, 40, 30, role="CB"), P(6, US, 40, 45, role="CB"), gk(11, US),
              P(20, THEM, 62, 57, role="FB"), P(21, THEM, 65, 52, role="CM"), P(22, THEM, 66, 61, role="WM"),
              P(23, THEM, 58, 54, role="ST"), P(24, THEM, 72, 45, role="CB"), P(25, THEM, 72, 32, role="CB"),
              gk(29, THEM)]
        return state(ps, 1, 1)
    S.append(Scenario("switch_play", "overload on my side, far winger unmarked -> switch it to him", a8,
                      lambda a, r, s: target_is(a, 2)))

    def a9():
        ps = [P(1, US, 70, 34, 6, 0, role="ST", ctrl=True), P(2, US, 60, 50, role="LW"), P(3, US, 55, 15, role="RW"),
              P(4, US, 50, 34, role="CM"), P(5, US, 35, 30, role="CB"), P(6, US, 35, 40, role="CB"), gk(11, US),
              P(20, THEM, 65, 30, 6, 0, role="CB"), P(21, THEM, 66, 38, 6, 0, role="CB"), P(22, THEM, 60, 20, role="FB"),
              P(23, THEM, 55, 45, role="CM"), gk(29, THEM)]
        return state(ps, 1, 1)
    S.append(Scenario("through_on_goal_keep_going", "clean through on goal 35 m out -> drive at goal, no back pass",
                      a9, lambda a, r, s: a.kind in (T.DRIBBLE, T.SHOOT) and (a.target_point is None
                                                                             or a.target_point[0] > 70)))

    def a10():
        ps = [P(1, US, 98, 6, role="RW", ctrl=True), P(2, US, 95, 33, role="ST"), P(3, US, 85, 40, role="CAM"),
              P(4, US, 70, 34, role="CM"), P(5, US, 50, 30, role="CB"), P(6, US, 50, 45, role="CB"), gk(11, US),
              P(20, THEM, 100, 38, role="CB"), P(21, THEM, 99, 12, role="FB"), P(22, THEM, 90, 45, role="CB"),
              P(23, THEM, 85, 25, role="CM"), gk(29, THEM)]
        return state(ps, 1, 1)
    S.append(Scenario("deliver_from_byline", "at the byline, striker free in the box -> cross / cut-back to him",
                      a10, lambda a, r, s: target_is(a, 2) and a.kind in (T.CROSS, T.PASS, T.LOB)))

    def a11():
        ps = [P(1, US, 25, 34, role="CM", ctrl=True), P(2, US, 8, 34, role="CB"), P(3, US, 28, 6, role="RB"),
              P(4, US, 40, 50, role="CM"), P(5, US, 50, 34, role="ST"), P(6, US, 30, 60, role="LB"), gk(11, US),
              P(20, THEM, 26.5, 34.2, -1, 0, role="CAM"), P(21, THEM, 11, 33, role="ST"), P(22, THEM, 40, 30, role="CM"),
              P(23, THEM, 60, 34, role="CB"), gk(29, THEM)]
        return state(ps, 1, 1)
    S.append(Scenario("no_square_ball_in_own_box", "pressed at the edge of my box, striker lurking -> never play it across goal",
                      a11, lambda a, r, s: not target_is(a, 2)))

    def a12():
        ps = [P(1, US, 50, 34, role="CM", ctrl=True), P(2, US, 66, 34, role="ST"), P(3, US, 64, 22, 6, 1.5, role="RW"),
              P(4, US, 45, 52, role="LM"), P(5, US, 35, 30, role="CB"), P(6, US, 40, 40, role="CDM"), gk(11, US),
              P(20, THEM, 67.5, 34, role="CB"), P(21, THEM, 68, 42, role="CB"), P(22, THEM, 66, 14, role="FB"),
              P(23, THEM, 66, 54, role="FB"), P(24, THEM, 56, 30, role="CM"), gk(29, THEM)]
        return state(ps, 1, 1)
    S.append(Scenario("runner_over_marked_striker", "winger's diagonal run between CB and FB vs a marked striker -> through to the runner",
                      a12, lambda a, r, s: target_is(a, 3) and a.kind in (T.THROUGH, T.LOB)))

    def a13():
        ps = [P(1, US, 30, 34, role="CB", ctrl=True), P(2, US, 70, 10, 5, 0, role="RW"), P(3, US, 75, 34, role="ST"),
              P(4, US, 55, 40, role="CM"), P(5, US, 30, 50, role="CB"), P(6, US, 45, 20, role="CDM"), gk(11, US),
              P(20, THEM, 80, 30, role="CB"), P(21, THEM, 80, 40, role="CB"), P(22, THEM, 74, 16, role="FB"),
              P(23, THEM, 65, 34, role="CM"), gk(29, THEM)]
        return state(ps, 1, 2)            # radar still highlights the CB, the winger has the ball
    S.append(Scenario("advise_the_ball_carrier", "FC auto-switches to the ball carrier -> advise the winger, not the CB",
                      a13, lambda a, r, s: s.controlled_id == 2 and a.kind in T.ATTACK_KINDS))

    # ---- defending (opponent attacks towards x = 0)
    def d1():
        ps = [P(1, US, 60, 50, role="CM", ctrl=True), P(2, US, 26, 34, role="CB"), P(3, US, 24, 45, role="CB"),
              P(4, US, 30, 15, role="RB"), P(5, US, 45, 30, role="CDM"), P(6, US, 70, 40, role="ST"), gk(11, US),
              P(20, THEM, 36, 34, -6, 0, role="ST"), P(21, THEM, 40, 50, -5, 0, role="LW"), P(22, THEM, 60, 30, role="CM"),
              P(23, THEM, 70, 34, role="CB"), gk(29, THEM)]
        return state(ps, 1, 20, attack=False)
    S.append(Scenario("switch_to_the_right_defender", "striker running at goal, I'm 30 m away, CB goal-side -> switch",
                      d1, lambda a, r, s: a.kind == T.SWITCH))

    def d2():
        ps = [P(1, US, 18, 34, role="CB", ctrl=True), P(2, US, 30, 10, role="RB"), P(3, US, 32, 58, role="LB"),
              P(4, US, 40, 34, role="CDM"), P(5, US, 55, 34, role="CM"), P(6, US, 70, 34, role="ST"), gk(11, US),
              P(20, THEM, 23, 34, -3, 0, role="ST"), P(21, THEM, 40, 50, role="LW"), P(22, THEM, 50, 30, role="CM"),
              gk(29, THEM)]
        return state(ps, 1, 20, attack=False)
    S.append(Scenario("jockey_at_the_box", "last defender facing the striker at my box, no help -> jockey, don't dive in",
                      d2, lambda a, r, s: a.kind == T.JOCKEY))

    def d3():
        ps = [P(1, US, 51, 30, role="CM", ctrl=True), P(2, US, 53, 36, role="CDM"), P(3, US, 35, 30, role="CB"),
              P(4, US, 35, 40, role="CB"), P(5, US, 70, 34, role="ST"), P(6, US, 45, 12, role="RM"), gk(11, US),
              P(20, THEM, 56, 30, role="CM"), P(21, THEM, 65, 50, role="CM"), P(22, THEM, 40, 20, role="ST"),
              gk(29, THEM)]
        return state(ps, 1, 20, attack=False)
    S.append(Scenario("press_with_support", "midfield, I'm close and goal-side with a teammate next to me -> press",
                      d3, lambda a, r, s: a.kind == T.PRESS))

    def d4():
        ps = [P(1, US, 25, 34, role="CB", ctrl=True), P(2, US, 42, 9, role="RB"), P(3, US, 22, 45, role="CB"),
              P(4, US, 40, 34, role="CDM"), P(5, US, 60, 34, role="CM"), P(6, US, 75, 34, role="ST"), gk(11, US),
              P(20, THEM, 45, 8, -3, 0, role="RW"), P(21, THEM, 31, 28, -6, 0.5, role="ST"),
              P(22, THEM, 55, 40, role="CM"), gk(29, THEM)]
        return state(ps, 1, 20, attack=False)
    S.append(Scenario("cover_the_runner", "ball on the wing with my full-back engaged, striker darting in -> cover the striker",
                      d4, lambda a, r, s: a.kind == T.COVER and target_is(a, 21)))
    return S


def run_exam(engine=None, verbose=True) -> dict:
    from fctac.tactics.engine import TacticsEngine
    res = {}
    for sc in scenarios():
        eng = engine or TacticsEngine()
        eng.reset()
        st = sc.build()
        for p in st.players:
            p.controlled = p.id == st.controlled_id
        ranked, _ = eng.decide(st)
        top = ranked[0] if ranked else None
        ok = bool(top is not None and sc.check(top, ranked, st))
        res[sc.name] = ok
        if verbose:
            txt = "-" if top is None else f"{top.text()} (#{top.target_id})"
            print(f"{'PASS' if ok else 'FAIL'}  {sc.name:30s} top: {txt:24s}  [{sc.why}]")
    n = sum(res.values())
    if verbose:
        print(f"\nscore {n}/{len(res)}")
    return res


if __name__ == "__main__":
    run_exam()


def robustness(n_variants: int = 10, jitter_m: float = 1.0, seed: int = 0) -> dict:
    """Same situations mirrored left/right and with every player nudged by up to ``jitter_m``:
    a sound decision must not hinge on exact positions."""
    from fctac.tactics.engine import TacticsEngine
    rng = np.random.default_rng(seed)
    out = {}
    for sc in scenarios():
        ok = 0
        for k in range(n_variants):
            st = sc.build()
            mirror = k % 2 == 1
            for p in st.players:
                if mirror:
                    p.pos[1] = T.PITCH_WIDTH - p.pos[1]
                    p.vel[1] = -p.vel[1]
                if p.id != st.controlled_id:
                    p.pos += rng.uniform(-jitter_m, jitter_m, 2)
                p.controlled = p.id == st.controlled_id
            owner = st.player(st.ball.owner_id)
            st.ball.pos = owner.pos + np.array([0.6 if st.ball.owner_team == US else -0.6, 0.0])
            ranked, _ = TacticsEngine().decide(st)
            ok += bool(ranked and sc.check(ranked[0], ranked, st))
        out[sc.name] = ok / n_variants
    return out
