import numpy as np

from fctac import types as T
from fctac.benchmark.pipeline_eval import evaluate
from fctac.benchmark.vision_eval import evaluate_detector
from fctac.state.gt import load_gt
from fctac.tracking.kalman import CVKalman
from fctac.vision.detector import ColorDetector
from fctac.vision.radar import RadarReader


def test_radar_reader_accuracy(synth_clip):
    import cv2
    gt = load_gt(synth_clip + ".gt.jsonl")
    cap = cv2.VideoCapture(synth_clip + ".mp4")
    rr = RadarReader()
    errs, found = [], []
    for i in range(60):
        ok, f = cap.read()
        r = rr.read(f)
        if not r.ok:
            continue
        g = np.array([[p["x"], p["y"]] for p in gt[i]["players"]])
        d = np.linalg.norm(g[:, None] - r.points[None], axis=2).min(1)
        errs += list(d[d < 3])
        found.append((d < 3).mean())
    assert len(found) > 50
    assert np.mean(found) > 0.9 and np.mean(errs) < 0.6


def test_detector_baseline(synth_clip):
    gt = load_gt(synth_clip + ".gt.jsonl")
    s = evaluate_detector(ColorDetector(), synth_clip + ".mp4", gt, 90)
    assert s["player_precision"] > 0.75 and s["player_recall"] > 0.75


def test_kalman_tracks_constant_velocity():
    kf = CVKalman([0, 0], q_acc=1.0)
    rng = np.random.default_rng(0)
    for k in range(60):
        kf.predict(1 / 30)
        kf.update(np.array([5.0, -2.0]) * (k + 1) / 30 + rng.normal(0, 0.1, 2), 0.01)
    assert np.allclose(kf.vel, [5.0, -2.0], atol=0.6)


def test_full_pipeline_against_ground_truth(synth_clip):
    r = evaluate(synth_clip, frames=150)
    assert r["state_valid_rate"] > 0.9
    assert r["player_pos_err_m"] < 0.6
    assert r["team_acc"] > 0.95
    assert r["attack_dir_acc"] > 0.9
    assert r["calib_err_m_median"] is not None and r["calib_err_m_median"] < 1.5
    assert r["latency_ms"]["total"]["mean"] < 100
