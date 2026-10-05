import os
import json
import threading
import time

import numpy as np

from fctac import types as T
from fctac.capture.base import LatestFrameBuffer


def test_latest_frame_buffer_drops_stale_frames():
    buf = LatestFrameBuffer()
    for k in range(5):
        buf.put(np.full((2, 2, 3), k, np.uint8))
    f = buf.get(-1, timeout=0.1)
    assert f.image[0, 0, 0] == 4            # newest frame, not the oldest queued one
    assert buf.dropped == 4
    assert buf.get(f.frame_id, timeout=0.05) is None   # nothing newer yet

    def producer():
        time.sleep(0.05)
        buf.put(np.full((2, 2, 3), 9, np.uint8))
    threading.Thread(target=producer).start()
    g = buf.get(f.frame_id, timeout=1.0)
    assert g is not None and g.image[0, 0, 0] == 9


def test_live_runtime_on_video(synth_clip):
    from fctac.capture.base import VideoFileSource
    from fctac.config import AppConfig
    from fctac.live import LiveRuntime, NullPresenter
    cfg = AppConfig()
    rt = LiveRuntime(cfg, VideoFileSource(synth_clip + ".mp4", speed=1.0), NullPresenter())
    rep = rt.run(seconds=4.0)
    assert rep["frames_processed"] > 30
    assert rep["frames_processed"] + rep["frames_dropped"] <= rep["frames_captured"] + 1
    assert rep["latency_ms"]["e2e"]["mean"] < 500


def test_decision_labeller_matches_sim_events():
    from fctac.sim.match import MatchSim, SimConfig
    from fctac.state.gt import state_from_gt
    from fctac.training.decisions import DecisionLabeler
    sim = MatchSim(SimConfig(seed=4))
    lab = DecisionLabeler("t")
    ex = []
    true_passes = 0
    for k in range(30 * 60):
        sim.step()
        s = sim.snapshot()
        s["video_frame"] = k
        true_passes += sum(1 for e in s["events"] if e["team"] == 0 and e["kind"] in ("pass", "through", "lob", "cross"))
        ex += lab.push(state_from_gt(s))
    ex += lab.flush()
    passes = [e for e in ex if e["human"]["kind"] in T.PASS_KINDS]
    assert true_passes > 3
    assert abs(len(passes) - true_passes) <= max(2, 0.2 * true_passes)
    for e in ex:
        assert e["features"] and len(e["features"]) == len(e["candidates"])
        assert "success" in e["outcome"]


def test_registry_never_deploys_worse(tmp_path):
    from fctac.training.registry import Registry
    art = tmp_path / "m.onnx"
    art.write_bytes(b"x")
    reg = Registry(str(tmp_path / "models"))
    a = reg.register("detector", str(art), {"f1": 0.9}, primary="f1")["deployed"]
    b = reg.register("detector", str(art), {"f1": 0.8}, primary="f1")["deployed"]
    c = reg.register("detector", str(art), {"f1": 0.95}, primary="f1")
    assert a and not b and c["deployed"]
    assert reg.deployed("detector")["version"] == c["version"]
    assert sum(e["deployed"] for e in reg.entries("detector")) == 1
    l1 = reg.register("predictor", str(art), {"ADE": 1.0}, primary="ADE", higher_is_better=False)["deployed"]
    l2 = reg.register("predictor", str(art), {"ADE": 1.2}, primary="ADE", higher_is_better=False)["deployed"]
    assert l1 and not l2
    # same-data comparison done by the caller overrides stored metrics from other data
    (tmp_path / "m.onnx.pt").write_bytes(b"w")
    r1 = reg.register("radar", str(art), {"radar_score": 0.9}, primary="radar_score", extra_files=(".json", ".pt"))
    d1 = r1["deployed"]
    r2 = reg.register("radar", str(art), {"radar_score": 0.5}, primary="radar_score", better=True)
    d2 = r2["deployed"]
    r3 = reg.register("radar", str(art), {"radar_score": 0.99}, primary="radar_score", better=False)
    assert d1 and d2 and not r3["deployed"] and not r1["deployed"]
    assert reg.deployed("radar")["version"] == r2["version"]
    assert os.path.exists(r1["path"] + ".pt")


def test_learned_ranker_scores_candidates(tmp_path, synth_clip):
    from fctac.pipeline import OracleAnalyzer
    from fctac.state.gt import load_gt
    from fctac.tactics.candidates import N_FEATURES
    from fctac.tactics.learned_ranker import LearnedRanker
    rng = np.random.default_rng(0)
    mlp = {"layers": [[rng.normal(0, 0.1, (N_FEATURES, 4)).tolist(), [0.0] * 4], [rng.normal(0, 0.1, (4, 1)).tolist(), [0.0]]],
           "mean": [0.0] * N_FEATURES, "std": [1.0] * N_FEATURES}
    p = tmp_path / "r.json"
    p.write_text(json.dumps({"success": mlp, "policy": mlp, "platt": [1.0, 0.0], "mode": "learned_ev"}))
    rk = LearnedRanker(str(p))
    gt = load_gt(synth_clip + ".gt.jsonl")
    an = OracleAnalyzer(gt)
    for i in range(len(gt)):
        fa = an.process(None, i, i / 30)
        atk = [a for a in fa.candidates if a.kind in T.ATTACK_KINDS]
        if atk:
            s = rk.score(atk)
            assert s.shape == (len(atk),) and np.all(np.isfinite(s))
            assert all(0.0 <= a.p_success <= 1.0 for a in atk)
            return
    raise AssertionError("no attacking frame found")
