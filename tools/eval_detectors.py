"""Compare player detectors on the same labelled frames.

    python tools/eval_detectors.py --labels data/datasets/real/m3 --models color models/detector_v003.onnx runs/detector/real/detector.onnx

"color" is the classical colour detector (with the given config's settings;
"color_global" also searches the whole frame for the ball);
anything else is an ONNX model.  Same metrics for all (fctac.training.det_eval):
players within 2 % of the image width, ignore regions excluded.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fctac.config import load_config  # noqa: E402
from fctac.training.det_dataset import list_images  # noqa: E402
from fctac.training.det_eval import evaluate_on_labels, evaluate_onnx_on_labels  # noqa: E402
from fctac.vision.detector import ColorDetector  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--models", nargs="+", default=["color"])
    ap.add_argument("--config", default="configs/fc27_1440p.json")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    paths = []
    for d in a.labels:
        paths += list_images(d)
    cfg = load_config(a.config)
    rows = {}
    for m in a.models:
        if m in ("color", "color_global"):
            import copy
            dc = copy.deepcopy(cfg.detector)
            if m == "color_global":
                dc.ball_global = True              # full-frame ball search (no radar prior in this test)
            r = evaluate_on_labels(ColorDetector(dc), paths)
        else:
            r = evaluate_onnx_on_labels(m, paths, providers=["CPUExecutionProvider"])
        rows[m] = r
        print(f"{m:45s} det_score {r['det_score']:.4f}  P {r['player_precision']:.3f} R {r['player_recall']:.3f} "
              f"F1 {r['f1']:.3f}  ball R {r['ball_recall']:.3f} P {r['ball_precision']:.3f}  "
              f"foot err {r['foot_err_px']} px  {r['ms_mean']} ms", flush=True)
    if a.out:
        with open(a.out, "w") as f:
            json.dump({"images": len(paths), "labels": a.labels, "results": rows}, f, indent=1)


if __name__ == "__main__":
    main()
