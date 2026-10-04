"""Phase 12: benchmark detector variants (precision x provider x input size).

    python tools/bench_models.py --model models/detector_v001.onnx --data data/datasets/fc27

For each variant it reports latency (mean/p95 over N runs, after warm-up)
and accuracy on the val split, then recommends the fastest variant whose F1
is within ``--max-f1-drop`` of FP32.  Variants:
  fp32        the exported model
  fp16        onnxruntime.transformers.float16 conversion (GPU providers)
  int8        static QDQ quantisation calibrated on dataset frames
Providers: whatever this machine's ONNX Runtime offers (TensorRT, CUDA,
DirectML, CPU).  Nothing here is assumed -- it is all measured.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import onnxruntime as ort  # noqa: E402


def to_fp16(src: str, dst: str):
    import onnx
    from onnxruntime.transformers.float16 import convert_float_to_float16
    m = onnx.load(src)
    m16 = convert_float_to_float16(m, keep_io_types=False)
    onnx.save(m16, dst)


def to_int8(src: str, dst: str, images: list[str], in_w: int, in_h: int):
    from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.it = iter(images)

        def get_next(self):
            p = next(self.it, None)
            if p is None:
                return None
            img = cv2.imread(p)
            s = in_w / img.shape[1]
            rh = min(int(round(img.shape[0] * s)), in_h)
            x = np.zeros((in_h, in_w, 3), np.uint8)
            x[:rh] = cv2.resize(img, (in_w, rh), interpolation=cv2.INTER_AREA)[:rh]
            return {"image": x.transpose(2, 0, 1)[None].astype(np.float32) / 255.0}
    pre = dst + ".pre.onnx"
    quant_pre_process(src, pre)
    quantize_static(pre, dst, Reader(), quant_format=QuantFormat.QDQ, activation_type=QuantType.QUInt8,
                    weight_type=QuantType.QInt8, per_channel=True)
    os.remove(pre)


def time_session(path: str, provider: str, in_h: int, in_w: int, n: int = 100) -> dict:
    from fctac.vision.learned import make_session
    sess = make_session(path, [provider])
    if provider not in sess.get_providers():
        return {"skipped": f"{provider} unavailable"}
    inp = sess.get_inputs()[0]
    dt = np.float16 if "float16" in inp.type else np.float32
    x = np.random.default_rng(0).random((1, 3, in_h, in_w)).astype(dt)
    for _ in range(10):
        sess.run(None, {inp.name: x})
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        sess.run(None, {inp.name: x})
        ts.append((time.perf_counter() - t0) * 1000)
    return {"ms_mean": round(float(np.mean(ts)), 3), "ms_p95": round(float(np.percentile(ts, 95)), 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="", help="dataset root (val split used for accuracy, train for INT8 calibration)")
    ap.add_argument("--providers", nargs="*", default=None)
    ap.add_argument("--max-f1-drop", type=float, default=0.01)
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    meta = json.load(open(a.model + ".json"))
    in_h, in_w = meta["input"]
    os.makedirs(os.path.join("runs", "bench"), exist_ok=True)
    base = os.path.join("runs", "bench", os.path.splitext(os.path.basename(a.model))[0])
    variants = {"fp32": a.model}
    try:
        to_fp16(a.model, base + ".fp16.onnx")
        variants["fp16"] = base + ".fp16.onnx"
    except Exception as e:
        print("fp16 conversion failed:", e)
    val, train = [], []
    if a.data:
        val = [p for p in open(os.path.join(a.data, "val.txt")).read().split() if p]
        train = [p for p in open(os.path.join(a.data, "train.txt")).read().split() if p]
    if train:
        try:
            to_int8(a.model, base + ".int8.onnx", train[:64], in_w, in_h)
            variants["int8"] = base + ".int8.onnx"
        except Exception as e:
            print("int8 quantisation failed:", e)
    for v, p in variants.items():
        if v != "fp32" and not os.path.exists(p + ".json"):
            json.dump(meta, open(p + ".json", "w"))
    providers = a.providers or ort.get_available_providers()
    providers = [p for p in providers if p != "AzureExecutionProvider"]
    from fctac.training.det_eval import evaluate_onnx_on_labels
    rows = []
    for v, p in variants.items():
        for prov in providers:
            if v == "fp16" and prov == "CPUExecutionProvider":
                continue              # CPU has no fast fp16 kernels; not a deployment option
            r = {"variant": v, "provider": prov, **time_session(p, prov, in_h, in_w, a.runs)}
            if "skipped" not in r and val:
                m = evaluate_onnx_on_labels(p, val[:60], providers=[prov])
                r.update({"f1": m["f1"], "ball_recall": m["ball_recall"], "controlled_acc": m["controlled_acc"]})
            rows.append(r)
            print(json.dumps(r))
    ok = [r for r in rows if "ms_mean" in r]
    ref = {r["provider"]: r.get("f1") for r in ok if r["variant"] == "fp32"}
    good = [r for r in ok if r.get("f1") is None or ref.get(r["provider"]) is None
            or r["f1"] >= ref[r["provider"]] - a.max_f1_drop]
    best = min(good, key=lambda r: r["ms_mean"]) if good else None
    rep = {"input": [in_h, in_w], "results": rows, "recommended": best}
    print("RECOMMENDED:", json.dumps(best))
    if a.out:
        json.dump(rep, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
