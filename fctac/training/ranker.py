"""Train and evaluate the learned tactical ranker against the physics baseline.

    python -m fctac.training.ranker --data data/decisions --out runs/ranker

Splits are by clip/recording (no leakage between train and test).
Reported metrics (held-out):
  success model   log-loss, Brier, AUC, ECE   (learned vs physics p)
  ranking         top-1 / top-3 agreement with the human choice
                  success rate & progression of human actions that the
                  ranker would also have recommended vs. not (off-policy proxy)
                  false-recommendation rate (matched recommendations that failed)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from dataclasses import dataclass

import numpy as np

from fctac import types as T
from fctac.tactics.candidates import CandidateGenerator
from fctac.training.decisions import read_examples, state_from_compact


@dataclass
class Example:
    F: np.ndarray            # (n_cand, n_feat)
    v: np.ndarray            # value if success
    c: np.ndarray            # turnover cost
    p_phys: np.ndarray
    ev_phys: np.ndarray
    kinds: list
    human: int               # index of the human's choice (-1 unknown)
    success: float           # outcome of the human's action (nan if unresolved)
    progress: float
    clip: str
    realized: float = 0.0        # goal units: xT gain if it worked, minus turnover cost if not


def rebuild(ex: dict, gen: CandidateGenerator) -> Example | None:
    st = state_from_compact(ex["state"])
    actor = ex["state"].get("controlled")
    if st.controlled is None:
        return None
    cands = gen.attacking(st)
    if not cands:
        return None
    h = ex["human"]
    idx = -1
    if h["kind"] in T.PASS_KINDS and h["target_id"] is not None:
        same = [i for i, a in enumerate(cands) if a.target_id == h["target_id"]]
        exact = [i for i in same if cands[i].kind == h["kind"]]
        idx = exact[0] if exact else (same[0] if same else -1)
    elif h["kind"] in (T.SHOOT, T.DRIBBLE, T.HOLD):
        idx = next((i for i, a in enumerate(cands) if a.kind == h["kind"]), -1)
    res = ex["outcome"]["result"]
    succ = np.nan if res == "unresolved" else float(ex["outcome"]["success"])
    from fctac.tactics.value import turnover_cost
    o = ex["outcome"]
    ep = h.get("end_point") or [st.ball.pos[0], st.ball.pos[1]]
    realized = float(o.get("xt_gain", 0.0)) if o["success"] else -float(turnover_cost(np.array(ep))[0])
    if h["kind"] == T.SHOOT:
        realized = 1.0 if res == "on_target" else -0.02
    return Example(realized=realized, F=np.stack([a.features for a in cands]).astype(np.float32),
                   v=np.array([a.value_success for a in cands]), c=np.array([a.detail.get("cost", 0.0) for a in cands]),
                   p_phys=np.array([a.p_success for a in cands]), ev_phys=np.array([a.score for a in cands]),
                   kinds=[a.kind for a in cands], human=idx, success=succ,
                   progress=float(ex["outcome"].get("progress_m", 0.0)), clip=ex["clip"])


def load(paths: list[str]) -> list[Example]:
    gen = CandidateGenerator()
    out = []
    for p in paths:
        for ex in read_examples(p):
            e = rebuild(ex, gen)
            if e is not None:
                out.append(e)
    return out


def split_by_clip(exs: list[Example], test_clips: set, val_clips: set):
    tr = [e for e in exs if e.clip not in test_clips | val_clips]
    va = [e for e in exs if e.clip in val_clips]
    te = [e for e in exs if e.clip in test_clips]
    return tr, va, te


# --- metrics -------------------------------------------------------------------
def logloss(p, y):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def auc(p, y):
    pos, neg = p[y == 1], p[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    r = np.argsort(np.argsort(np.concatenate([pos, neg])))
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) - 1) / 2) / (len(pos) * len(neg)))


def ece(p, y, bins=10):
    e = 0.0
    for k in range(bins):
        m = (p >= k / bins) & (p < (k + 1) / bins + (k == bins - 1))
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)


def success_metrics(p, y) -> dict:
    return {"logloss": round(logloss(p, y), 4), "brier": round(float(np.mean((p - y) ** 2)), 4),
            "auc": round(auc(p, y), 4), "ece": round(ece(p, y), 4), "n": int(len(y))}


def ranking_metrics(exs: list[Example], score_fn) -> dict:
    top1 = top3 = n = 0
    m_succ, u_succ, m_prog, u_prog, m_val, u_val = [], [], [], [], [], []
    for e in exs:
        if e.human < 0:
            continue
        s = score_fn(e)
        order = np.argsort(-s)
        n += 1
        top1 += int(order[0] == e.human)
        top3 += int(e.human in order[:3])
        if np.isnan(e.success):
            continue
        if order[0] == e.human:
            m_succ.append(e.success)
            m_prog.append(e.progress)
            m_val.append(e.realized)
        else:
            u_succ.append(e.success)
            u_prog.append(e.progress)
            u_val.append(e.realized)
    mean = lambda a: round(float(np.mean(a)), 4) if a else None   # noqa: E731
    return {"top1": round(top1 / max(n, 1), 4), "top3": round(top3 / max(n, 1), 4), "n": n,
            "matched_success": mean(m_succ), "unmatched_success": mean(u_succ),
            "matched_progress_m": mean(m_prog), "unmatched_progress_m": mean(u_prog),
            "false_rec_rate": round(1 - float(np.mean(m_succ)), 4) if m_succ else None,
            "matched_value": round(float(np.mean(m_val)), 5) if m_val else None,
            "unmatched_value": round(float(np.mean(u_val)), 5) if u_val else None,
            # value advantage: how much better the human did when following the ranker
            "value_advantage": round(float(np.mean(m_val) - np.mean(u_val)), 5) if m_val and u_val else None,
            "matched_n": len(m_succ)}


# --- training (torch) ------------------------------------------------------------
def _mlp(sizes):
    import torch.nn as nn
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def _export(model, mean, std) -> dict:
    import torch.nn as nn
    layers = [[m.weight.detach().numpy().T.tolist(), m.bias.detach().numpy().tolist()]
              for m in model if isinstance(m, nn.Linear)]
    return {"layers": layers, "mean": mean.tolist(), "std": std.tolist()}


def train_success(tr: list[Example], va: list[Example], hidden=(32, 16), epochs=300, seed=0):
    import torch
    torch.manual_seed(seed)
    X = np.stack([e.F[e.human] for e in tr if e.human >= 0 and not np.isnan(e.success)])
    y = np.array([e.success for e in tr if e.human >= 0 and not np.isnan(e.success)], np.float32)
    Xv = np.stack([e.F[e.human] for e in va if e.human >= 0 and not np.isnan(e.success)])
    yv = np.array([e.success for e in va if e.human >= 0 and not np.isnan(e.success)], np.float32)
    mean, std = X.mean(0), X.std(0) + 1e-3
    model = _mlp([X.shape[1], *hidden, 1])
    opt = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-3)
    xt, yt = torch.tensor((X - mean) / std), torch.tensor(y)
    xv, yvt = torch.tensor((Xv - mean) / std), torch.tensor(yv)
    lossf = torch.nn.BCEWithLogitsLoss()
    best = (1e9, None)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(xt))
        for k in range(0, len(xt), 128):
            b = perm[k:k + 128]
            loss = lossf(model(xt[b])[:, 0], yt[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = float(lossf(model(xv)[:, 0], yvt))
        if vl < best[0]:
            best = (vl, {k: v.clone() for k, v in model.state_dict().items()})
    model.load_state_dict(best[1])
    return _export(model, mean, std)


def train_policy(tr: list[Example], va: list[Example], hidden=(32,), epochs=60, fail_weight=0.3, seed=0):
    """Outcome-weighted behaviour cloning: failed human choices count less."""
    import torch
    torch.manual_seed(seed)
    data = [e for e in tr if e.human >= 0]
    allF = np.concatenate([e.F for e in data])
    mean, std = allF.mean(0), allF.std(0) + 1e-3
    model = _mlp([allF.shape[1], *hidden, 1])
    opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-3)

    def nll(exs, train):
        tot = 0.0
        for e in exs:
            s = model(torch.tensor((e.F - mean) / std))[:, 0]
            w = 1.0 if (np.isnan(e.success) or e.success > 0) else fail_weight
            l = -w * torch.log_softmax(s, 0)[e.human]
            if train:
                l.backward()
            tot += float(l)
        return tot / max(len(exs), 1)

    vdata = [e for e in va if e.human >= 0]
    best = (1e9, None)
    rng = np.random.default_rng(seed)
    for ep in range(epochs):
        model.train()
        order = rng.permutation(len(data))
        for k in range(0, len(order), 32):
            opt.zero_grad()
            nll([data[i] for i in order[k:k + 32]], True)
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = nll(vdata, False)
        if vl < best[0]:
            best = (vl, {k: v.clone() for k, v in model.state_dict().items()})
    model.load_state_dict(best[1])
    return _export(model, mean, std)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/decisions")
    ap.add_argument("--glob", default="sim_seed*.jsonl")
    ap.add_argument("--test-clips", nargs="*", default=["sim_seed6"])
    ap.add_argument("--val-clips", nargs="*", default=["sim_seed5"])
    ap.add_argument("--out", default="runs/ranker")
    ap.add_argument("--beta", type=float, default=0.0)
    ap.add_argument("--register", action="store_true")
    a = ap.parse_args()
    from fctac.tactics.learned_ranker import LearnedRanker, MLP
    paths = sorted(glob.glob(os.path.join(a.data, a.glob)))
    exs = load(paths)
    tr, va, te = split_by_clip(exs, set(a.test_clips), set(a.val_clips))
    print(f"examples train {len(tr)} val {len(va)} test {len(te)}  (files: {[os.path.basename(p) for p in paths]})")
    succ = train_success(tr, va)
    pol = train_policy(tr, va)
    os.makedirs(a.out, exist_ok=True)
    spec = {"success": succ, "policy": pol, "beta": a.beta,
            "meta": {"train_clips": sorted({e.clip for e in tr}), "features": int(tr[0].F.shape[1])}}
    path = os.path.join(a.out, "ranker.json")
    json.dump(spec, open(path, "w"))
    rk = LearnedRanker(path)
    pm = MLP(pol)
    # success-model comparison on chosen actions
    rows = [(e.F[e.human], e.p_phys[e.human], e.success) for e in te if e.human >= 0 and not np.isnan(e.success)]
    X = np.stack([r[0] for r in rows])
    y = np.array([r[2] for r in rows])
    rep = {"success_physics": success_metrics(np.array([r[1] for r in rows]), y),
           "success_learned": success_metrics(rk.p_success(X), y)}

    # Platt-calibrated physics: fixes calibration but keeps the physics ranking
    # structure, which generalises to actions humans rarely choose
    tr_rows = [(e.p_phys[e.human], e.success) for e in tr if e.human >= 0 and not np.isnan(e.success)]
    lp = np.log(np.clip([r[0] for r in tr_rows], 1e-4, 1 - 1e-4) / (1 - np.clip([r[0] for r in tr_rows], 1e-4, 1 - 1e-4)))
    ya = np.array([r[1] for r in tr_rows])
    A = np.c_[lp, np.ones_like(lp)]
    w = np.zeros(2)
    for _ in range(200):                                 # Newton steps for 1-D logistic regression
        pr = 1 / (1 + np.exp(-A @ w))
        g = A.T @ (pr - ya)
        Hm = A.T @ (A * (pr * (1 - pr))[:, None]) + 1e-6 * np.eye(2)
        w -= np.linalg.solve(Hm, g)
    platt = [float(w[0]), float(w[1])]

    def p_platt(p):
        p = np.clip(p, 1e-4, 1 - 1e-4)
        return 1 / (1 + np.exp(-(platt[0] * np.log(p / (1 - p)) + platt[1])))
    rep["success_platt_physics"] = success_metrics(p_platt(np.array([r[1] for r in rows])), y)

    def ev_learned(e):
        p = rk.p_success(e.F)
        return p * e.v - (1 - p) * e.c

    def ev_platt(e):
        p = p_platt(e.p_phys)
        return p * e.v - (1 - p) * e.c
    variants = {"heuristic": lambda e: e.ev_phys, "platt_ev": ev_platt, "learned_ev": ev_learned,
                "policy_only": lambda e: pm(e.F)}
    for beta in (0.002, 0.005, 0.01, 0.02):
        variants[f"platt_ev+policy(b={beta})"] = lambda e, b=beta: ev_platt(e) + b * pm(e.F)
        variants[f"learned_ev+policy(b={beta})"] = lambda e, b=beta: ev_learned(e) + b * pm(e.F)
    val_rep = {k: ranking_metrics(va, f) for k, f in variants.items()}
    best = max(val_rep, key=lambda k: (val_rep[k]["matched_value"] or -1e9))
    rep["validation_value_by_variant"] = {k: v["matched_value"] for k, v in val_rep.items()}
    rep["selected_on_validation"] = best
    for k, f in variants.items():
        rep[f"rank_{k}"] = ranking_metrics(te, f)
    spec["platt"] = platt
    spec["mode"] = best
    json.dump(spec, open(path, "w"))
    json.dump(rep, open(os.path.join(a.out, "report.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1))
    if a.register:
        from fctac.training.registry import Registry
        m = rep["success_learned"]
        entry = Registry().register("ranker", path, {**m, "baseline_logloss": rep["success_physics"]["logloss"]},
                                    primary="logloss", higher_is_better=False, data=a.data)
        print("registered", entry["version"], "deployed" if entry["deployed"] else "(not deployed)")


if __name__ == "__main__":
    main()
