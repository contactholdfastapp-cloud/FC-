"""Train the FC 27 radar reader (TinyRadarNet) and register it.

    python tools/train_radar.py --synth data/radar/synth --epochs 10
    python tools/train_radar.py --synth data/radar/synth --real data/radar/real_pseudo --init runs/radar/last.pt
    python tools/train_radar.py --export-only runs/radar/best.pt --register

Validation: the last ``--val`` synthetic samples (exact labels) and, when
given, labelled real crops (``--real-val``).  The best epoch (synthetic F1 +
ball recall) is exported to ONNX and, with ``--register``, added to
models/registry.json under "radar" (deployed only if better).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fctac.training.radar_model import (RadarDataset, TinyRadarNet, evaluate, export_onnx, focal_loss,  # noqa: E402
                                        load_labels, onnx_predictor, torch_predictor)


def score(m: dict) -> float:
    return 0.6 * m["f1"] + 0.2 * m["shape_acc"] + 0.2 * m["ball_recall"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synth", nargs="*", default=["data/radar/synth"])
    ap.add_argument("--real", nargs="*", default=[], help="real crops with (pseudo-)labels, same format")
    ap.add_argument("--real-weight", type=float, default=1.0, help="sampling weight of real vs synthetic rows")
    ap.add_argument("--real-val", default="", help="labelled real crops for validation")
    ap.add_argument("--val", type=int, default=1500)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--samples-per-epoch", type=int, default=0, help="0 = all training rows")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--patch", type=int, nargs=2, default=[128, 192])
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--init", default="")
    ap.add_argument("--out", default="runs/radar")
    ap.add_argument("--export-only", default="")
    ap.add_argument("--register", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)

    synth = []
    for d in a.synth:
        synth += load_labels(d)
    val = synth[-a.val:] if a.val else []
    train = synth[:-a.val] if a.val else synth
    real = []
    for d in a.real:
        real += load_labels(d)
    real_val = load_labels(a.real_val) if a.real_val else []

    net = TinyRadarNet()
    if a.init or a.export_only:
        net.load_state_dict(torch.load(a.export_only or a.init, map_location="cpu"))
    best_path = os.path.join(a.out, "best.pt")
    hist = []
    if not a.export_only:
        rows = train + real
        weights = np.array([1.0] * len(train) + [a.real_weight * len(train) / max(len(real), 1)] * len(real))
        weights /= weights.sum()
        n_ep = a.samples_per_epoch or len(rows)
        ds = RadarDataset(rows, tuple(a.patch), train=True)
        opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
        steps = a.epochs * (n_ep // a.batch)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.1)
        best = -1.0
        for ep in range(a.epochs):
            sampler = torch.utils.data.WeightedRandomSampler(weights, n_ep, replacement=True)
            dl = torch.utils.data.DataLoader(ds, batch_size=a.batch, sampler=sampler, num_workers=a.workers,
                                             drop_last=True, persistent_workers=False)
            net.train()
            t0 = time.time()
            tot = 0.0
            for k, (x, t, m) in enumerate(dl):
                loss = focal_loss(net(x), t, m)
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()
                tot += float(loss)
                if k % 200 == 0:
                    print(f"ep {ep} it {k}/{n_ep // a.batch} loss {tot / (k + 1):.4f} {time.time() - t0:.0f}s", flush=True)
            mv = evaluate(torch_predictor(net), val[:600]) if val else {}
            rec = {"epoch": ep, "loss": round(tot / max(k + 1, 1), 4), "val": mv, "sec": round(time.time() - t0)}
            if real_val:
                rec["real_val"] = evaluate(torch_predictor(net), real_val)
            hist.append(rec)
            print(json.dumps(rec), flush=True)
            torch.save(net.state_dict(), os.path.join(a.out, "last.pt"))
            s = score(mv) if mv else -ep
            if s > best:
                best = s
                torch.save(net.state_dict(), best_path)
        net.load_state_dict(torch.load(best_path, map_location="cpu"))
    onnx_path = os.path.join(a.out, "radar.onnx")
    torch.save(net.state_dict(), onnx_path + ".pt")        # weights travel with the model (fine-tuning)
    meta = {"input": [192, 320], "stride": 2, "channels": ["triangle", "circle", "ball", "highlight"],
            "thr": 0.35, "train_synth": a.synth, "train_real": a.real}
    export_onnx(net, onnx_path, meta)
    m = evaluate(onnx_predictor(onnx_path, ["CPUExecutionProvider"]), val) if val else {}
    rep = {"synthetic_val": m, "history": hist}
    if real_val:
        rep["real_val"] = evaluate(onnx_predictor(onnx_path, ["CPUExecutionProvider"]), real_val)
    print("ONNX:", json.dumps({k: v for k, v in rep.items() if k != "history"}))
    with open(os.path.join(a.out, "report.json"), "w") as f:
        json.dump(rep, f, indent=1)
    if a.register:
        from fctac.training.registry import Registry
        metrics = {f"synth_{k}": v for k, v in m.items()}
        metrics.update({f"real_{k}": v for k, v in rep.get("real_val", {}).items()})
        metrics["radar_score"] = round(score(rep["real_val"]) if real_val else score(m), 4)
        e = Registry().register("radar", onnx_path, metrics, primary="radar_score",
                                data=",".join(a.synth + a.real), extra_files=(".json", ".pt"))
        print("registered", e["version"], "deployed" if e["deployed"] else "(not better: kept current)")


if __name__ == "__main__":
    main()
