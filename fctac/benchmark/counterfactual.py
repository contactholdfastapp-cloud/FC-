"""Counterfactual evaluation of rankers in the simulator.

At every decision of our ball carrier, each ranker's top recommendation is
*executed* in copies of the match, which are rolled forward; the value of the
resulting situation (xT-like, goal = 1, turnover = -cost) is averaged over
several random rollouts.  This measures "what happens if the player follows
the advice" -- impossible to observe in recorded FC 27 footage, but a clean
test of the ranking method itself.  Results only hold for the simulator's
football; real validation needs FC 27 recordings.

    python -m fctac.benchmark.counterfactual --minutes 4 --seed 7 --rollouts 3
"""
from __future__ import annotations

import argparse
import copy
import json

import numpy as np

from fctac import types as T
from fctac.sim.match import MatchSim, SimConfig
from fctac.state.gt import state_from_gt
from fctac.tactics.candidates import CandidateGenerator
from fctac.tactics.value import turnover_cost, zone_value

SIM_KIND = {T.PASS: "pass", T.THROUGH: "through", T.LOB: "lob", T.CROSS: "cross", T.SHOOT: "shot",
            T.DRIBBLE: "dribble", T.HOLD: "hold"}


def situation_value(sim: MatchSim, scored_before: int) -> float:
    if sim.score[0] > scored_before:
        return 1.0
    b = sim.to_team(sim.ball, 0)
    poss = sim.poss_team()
    if poss == 0:
        return float(zone_value(b)[0])
    if poss == 1:
        return -float(turnover_cost(b)[0])
    return 0.5 * float(zone_value(b)[0]) - 0.5 * float(turnover_cost(b)[0])


def rollout(sim: MatchSim, action: T.Action, sign: int, horizon_s: float, seed: int) -> float:
    s = copy.deepcopy(sim)
    s.rng = np.random.default_rng(seed)
    before = s.score[0]
    tgt = None
    if action.target_point is not None:
        tgt = action.target_point if sign > 0 else T.PITCH_SIZE - action.target_point
    s.force_action(SIM_KIND[action.kind], action.target_id, tgt)
    for _ in range(int(horizon_s * s.cfg.fps)):
        s.step()
        if s.score[0] > before:
            break
    return situation_value(s, before)


def run(rankers: dict, minutes: float, seed: int, rollouts: int = 3, horizon_s: float = 3.0, max_decisions: int = 0):
    sim = MatchSim(SimConfig(seed=seed))
    gen = CandidateGenerator()
    vals = {k: [] for k in rankers}
    vals["sim_policy"] = []
    picks = {k: {} for k in rankers}
    n = int(minutes * 60 * sim.cfg.fps)
    decisions = 0
    for _ in range(n):
        o = sim.owner
        if o is not None and sim.team[o] == 0 and sim.t >= sim.decision_at - 1e-9 and sim.restart_at < 0:
            snap = sim.snapshot()
            snap["video_frame"] = sim.frame
            st = state_from_gt(snap)
            st.controlled_id = o
            for p in st.players:
                p.controlled = p.id == o
            cands = gen.attacking(st)
            if cands:
                decisions += 1
                for name, rk in rankers.items():
                    sc = rk(cands, st)
                    a = cands[int(np.argmax(sc))]
                    picks[name][a.kind] = picks[name].get(a.kind, 0) + 1
                    vals[name].append(np.mean([rollout(sim, a, sim.sign[0], horizon_s, seed * 1000 + decisions * 10 + r)
                                               for r in range(rollouts)]))
                # the simulator's own (noisy human-like) policy from the same state
                res = []
                for r in range(rollouts):
                    s = copy.deepcopy(sim)
                    s.rng = np.random.default_rng(seed * 1000 + decisions * 10 + r)
                    before = s.score[0]
                    for _ in range(int(horizon_s * s.cfg.fps)):
                        s.step()
                        if s.score[0] > before:
                            break
                    res.append(situation_value(s, before))
                vals["sim_policy"].append(np.mean(res))
                if max_decisions and decisions >= max_decisions:
                    break
        sim.step()
    out = {}
    base = np.array(vals.get("heuristic", []))
    for k, v in vals.items():
        v = np.array(v)
        out[k] = {"mean_value": round(float(v.mean()), 5), "sem": round(float(v.std() / np.sqrt(max(len(v), 1))), 5),
                  "n": int(len(v))}
        if len(base) == len(v) and k != "heuristic" and len(v):
            d = v - base            # paired (common random numbers) -> much lower variance
            out[k]["vs_heuristic"] = round(float(d.mean()), 5)
            out[k]["vs_heuristic_sem"] = round(float(d.std() / np.sqrt(len(d))), 5)
        if k in picks:
            out[k]["action_mix"] = picks[k]
    return out


def default_rankers(ranker_json: str | None = None) -> dict:
    rk = {"heuristic": lambda c, st: np.array([a.score for a in c])}
    if ranker_json:
        from fctac.tactics.learned_ranker import LearnedRanker, MLP
        spec = json.load(open(ranker_json))
        lr = LearnedRanker(ranker_json, beta=0.0)
        pol = MLP(spec["policy"])
        platt = spec.get("platt", [1.0, 0.0])

        def p_platt(p):
            p = np.clip(p, 1e-4, 1 - 1e-4)
            return 1 / (1 + np.exp(-(platt[0] * np.log(p / (1 - p)) + platt[1])))

        def ev(c, p):
            v = np.array([a.value_success for a in c])
            cost = np.array([a.detail.get("cost", 0.0) for a in c])
            return p * v - (1 - p) * cost

        def F(c):
            return np.stack([a.features for a in c])
        rk["platt_ev"] = lambda c, st: ev(c, p_platt(np.array([a.p_success for a in c])))
        rk["learned_ev"] = lambda c, st: ev(c, lr.p_success(F(c)))
        rk["policy_only"] = lambda c, st: pol(F(c))
        rk["learned_ev+policy(0.01)"] = lambda c, st: ev(c, lr.p_success(F(c))) + 0.01 * pol(F(c))
        rk["platt_ev+policy(0.01)"] = lambda c, st: ev(c, p_platt(np.array([a.p_success for a in c]))) + 0.01 * pol(F(c))
    return rk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=4)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--rollouts", type=int, default=3)
    ap.add_argument("--ranker", default="runs/ranker/ranker.json")
    ap.add_argument("--max-decisions", type=int, default=0)
    a = ap.parse_args()
    import os
    res = run(default_rankers(a.ranker if os.path.exists(a.ranker) else None), a.minutes, a.seed, a.rollouts,
              max_decisions=a.max_decisions)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
