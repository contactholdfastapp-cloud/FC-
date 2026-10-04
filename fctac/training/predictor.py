"""Movement-prediction dataset, training and benchmark (learned vs kinematic).

Samples are built from any sequence of GameStates (oracle sim states or
tracked vision states): for each player at time t the input is the recent
trajectory (relative displacements over the last 0.5 s), velocity and
ball/team context; the target is the displacement at 0.25/0.5/0.75/1.0/1.5 s.
Evaluation: ADE/FDE per horizon on held-out clips.  The learned model is only
deployed if it beats the (tau-fitted) kinematic baseline.

    python -m fctac.training.predictor --sim-minutes 20 --seeds 1 2 3 4
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from fctac import types as T
from fctac.prediction.kinematic import HORIZONS, KinematicPredictor

HIST_STEPS = (0.1, 0.2, 0.3, 0.4, 0.5)       # seconds in the past
N_IN = 2 * len(HIST_STEPS) + 2 + 2 + 2 + 3   # history, vel, ball rel, ball vel, flags


def features(pos_hist: np.ndarray, vel: np.ndarray, ball: np.ndarray, ball_vel: np.ndarray,
             team: int, possession: int, x_att: float) -> np.ndarray:
    """pos_hist: (len(HIST_STEPS),2) past positions relative to the current one."""
    return np.concatenate([pos_hist.ravel() / 5.0, vel / 8.0, np.clip(ball, -40, 40) / 20.0, ball_vel / 20.0,
                           [float(team == possession), float(team == T.TEAM_US), x_att / T.PITCH_LENGTH]]).astype(np.float32)


def build_samples(states: list, stride: int = 6, fps: float = 30.0):
    """States must be consecutive frames at ``fps``.  Returns X (N,N_IN), Y (N,H,2), P0 (N,2), V0 (N,2)."""
    by_t = {}
    for k, st in enumerate(states):
        by_t[k] = {p.id: p for p in st.players}
    hist_k = [int(round(h * fps)) for h in HIST_STEPS]
    fut_k = [int(round(h * fps)) for h in HORIZONS]
    X, Y, P0, V0 = [], [], [], []
    for k in range(max(hist_k), len(states) - max(fut_k), stride):
        st = states[k]
        if st.ball is None:
            continue
        for p in st.players:
            hist = []
            ok = True
            for hk in hist_k:
                q = by_t[k - hk].get(p.id)
                if q is None:
                    ok = False
                    break
                hist.append(q.pos - p.pos)
            fut = []
            for fk in fut_k:
                q = by_t[k + fk].get(p.id)
                if q is None:
                    ok = False
                    break
                fut.append(q.pos - p.pos)
            if not ok:
                continue
            X.append(features(np.array(hist), p.vel, st.ball.pos - p.pos, st.ball.vel, p.team, st.possession, p.pos[0]))
            Y.append(np.array(fut))
            P0.append(p.pos)
            V0.append(p.vel)
    return np.array(X, np.float32), np.array(Y, np.float32), np.array(P0), np.array(V0)


def sim_states(minutes: float, seed: int) -> list:
    from fctac.sim.match import MatchSim, SimConfig
    from fctac.state.gt import state_from_gt
    sim = MatchSim(SimConfig(seed=seed))
    out = []
    for k in range(int(minutes * 60 * sim.cfg.fps)):
        sim.step()
        s = sim.snapshot()
        s["video_frame"] = k
        out.append(state_from_gt(s))
    return out


def ade_fde(pred_disp: np.ndarray, Y: np.ndarray) -> dict:
    e = np.linalg.norm(pred_disp - Y, axis=2)          # (N,H)
    return {f"{h}s": round(float(e[:, i].mean()), 3) for i, h in enumerate(HORIZONS)} | {"ADE": round(float(e.mean()), 3)}


def fit_tau(V0, Y) -> float:
    best = (np.inf, 1.0)
    for tau in (0.4, 0.6, 0.8, 1.0, 1.3, 1.6, 2.0, 3.0, 5.0, 1e6):
        kp = KinematicPredictor(tau=tau)
        f = kp.factor(np.array(HORIZONS))
        pred = V0[:, None, :] * f[None, :, None]
        err = np.linalg.norm(pred - Y, axis=2).mean()
        if err < best[0]:
            best = (err, tau)
    return best[1]


def train_mlp(X, Y, Xv, Yv, hidden=(64, 64), epochs=40, seed=0):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    mean, std = X.mean(0), X.std(0) + 1e-3
    sizes = [X.shape[1], *hidden, Y.shape[1] * 2]
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(nn.ReLU())
    model = nn.Sequential(*layers)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    xt, yt = torch.tensor((X - mean) / std), torch.tensor(Y.reshape(len(Y), -1))
    xv, yv = torch.tensor((Xv - mean) / std), torch.tensor(Yv.reshape(len(Yv), -1))
    best = (1e9, None)
    for ep in range(epochs):
        perm = torch.randperm(len(xt))
        for k in range(0, len(xt), 512):
            b = perm[k:k + 512]
            loss = torch.mean(torch.norm((model(xt[b]) - yt[b]).view(len(b), -1, 2), dim=2))
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            vl = float(torch.mean(torch.norm((model(xv) - yv).view(len(yv), -1, 2), dim=2)))
        if vl < best[0]:
            best = (vl, {k: v.clone() for k, v in model.state_dict().items()})
    model.load_state_dict(best[1])
    spec = {"layers": [[m.weight.detach().numpy().T.tolist(), m.bias.detach().numpy().tolist()]
                       for m in model if isinstance(m, nn.Linear)],
            "mean": mean.tolist(), "std": std.tolist()}
    return spec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim-minutes", type=float, default=15)
    ap.add_argument("--seeds", nargs="*", type=int, default=[1, 2, 3, 4])
    ap.add_argument("--out", default="runs/predictor")
    ap.add_argument("--register", action="store_true")
    ap.add_argument("--epochs", type=int, default=25)
    a = ap.parse_args()
    from fctac.prediction.learned import LearnedPredictor
    data = []
    for s in a.seeds:
        X, Y, P0, V0 = build_samples(sim_states(a.sim_minutes, s))
        data.append((X, Y, P0, V0))
        print(f"seed {s}: {len(X)} samples")
    test = data[-1]
    val = data[-2]
    Xtr = np.concatenate([d[0] for d in data[:-2]])
    Ytr = np.concatenate([d[1] for d in data[:-2]])
    Vtr = np.concatenate([d[3] for d in data[:-2]])
    tau = fit_tau(Vtr, Ytr)
    spec = train_mlp(Xtr, Ytr, val[0], val[1], epochs=a.epochs)
    spec["tau_fallback"] = tau
    spec["horizons"] = list(HORIZONS)
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "predictor.json")
    json.dump(spec, open(path, "w"))
    Xt, Yt, Pt, Vt = test
    rep = {}
    for name, tau_ in (("constant_velocity", 1e6), (f"kinematic_tau={tau}", tau)):
        kp = KinematicPredictor(tau=tau_)
        f = kp.factor(np.array(HORIZONS))
        rep[name] = ade_fde(Vt[:, None, :] * f[None, :, None], Yt)
    lp = LearnedPredictor(path)
    rep["learned_mlp"] = ade_fde(lp.predict_disp(Xt), Yt)
    rep["test_samples"] = int(len(Xt))
    json.dump(rep, open(os.path.join(a.out, "report.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1))
    if a.register:
        from fctac.training.registry import Registry
        base = rep[f"kinematic_tau={tau}"]["ADE"]
        entry = Registry().register("predictor", path, {**rep["learned_mlp"], "baseline_ADE": base},
                                    primary="ADE", higher_is_better=False, data=f"sim seeds {a.seeds}")
        # never deploy a learned predictor that does not beat the baseline
        if rep["learned_mlp"]["ADE"] >= base and entry["deployed"]:
            reg = Registry()
            for e in reg.entries("predictor"):
                if e["version"] == entry["version"]:
                    e["deployed"] = False
            reg._save()
            entry["deployed"] = False
        print("registered", entry["version"], "deployed" if entry["deployed"] else "(not deployed)")


if __name__ == "__main__":
    main()
