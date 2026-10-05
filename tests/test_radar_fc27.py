"""FC 27 radar: geometry, renderer labels, decoding, panel alignment, reader plumbing."""
import os

import cv2
import numpy as np
import pytest

from fctac.vision.radar import RadarConfig
from fctac.vision.radar_fc27 import (CH, CW, STRIDE, canon_to_pitch, canonical_crop, decode, estimate_panel,
                                     pitch_to_canon)


def test_canon_roundtrip_and_panel_corners():
    xy = np.array([[0.0, 0.0], [105.0, 68.0], [52.5, 34.0], [10.0, 60.0]])
    assert np.allclose(canon_to_pitch(pitch_to_canon(xy)), xy)
    uv = pitch_to_canon(np.array([[0.0, 68.0], [105.0, 0.0]]))
    assert np.allclose(uv, [[14.5, 16.0], [305.5, 176.0]])


def test_canonical_crop_maps_panel_to_fixed_box():
    panel = RadarConfig().panel
    for w, h in ((1920, 1080), (2560, 1440)):
        f = np.zeros((h, w, 3), np.uint8)
        x0, y0, x1, y1 = panel[0] * w, panel[1] * h, panel[2] * w, panel[3] * h
        cv2.rectangle(f, (int(round(x0)), int(round(y0))), (int(round(x1)) - 1, int(round(y1)) - 1), (255, 255, 255), -1)
        crop, M = canonical_crop(f, panel)
        assert crop.shape == (CH, CW, 3)
        m = crop[..., 0] > 128
        ys, xs = np.nonzero(m)
        assert abs(xs.min() - 14.5) <= 1.5 and abs(xs.max() - 305.5) <= 1.5
        assert abs(ys.min() - 16) <= 1.5 and abs(ys.max() - 176) <= 1.5


def test_decode_peaks_shapes_ball_highlight():
    heat = np.zeros((4, CH // STRIDE, CW // STRIDE), np.float32)
    yy, xx = np.mgrid[0:CH // STRIDE, 0:CW // STRIDE]

    def put(ch, u, v, amp=0.9):
        cx, cy = (u + 0.5) / STRIDE - 0.5, (v + 0.5) / STRIDE - 0.5
        heat[ch] = np.maximum(heat[ch], amp * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / 2.0))

    put(0, 100.3, 60.7)
    put(1, 200.6, 120.2)
    put(1, 100.6, 61.0, 0.5)          # same symbol, weaker other shape: suppressed
    put(2, 150.0, 90.0)
    put(3, 200.6, 120.2)
    d = decode(heat)
    assert len(d["uv"]) == 2
    i_t = int(np.nonzero(d["shape"] == 0)[0][0])
    i_c = int(np.nonzero(d["shape"] == 1)[0][0])
    assert np.hypot(*(d["uv"][i_t] - [100.3, 60.7])) < 0.6
    assert np.hypot(*(d["uv"][i_c] - [200.6, 120.2])) < 0.6
    assert d["highlight"][i_c] > d["hl_thr"] > d["highlight"][i_t]
    assert np.hypot(*(d["ball_uv"] - [150.0, 90.0])) < 0.6


@pytest.fixture(scope="module")
def pool():
    from fctac.sim.radar_fc27 import PositionPool
    return PositionPool(n_runs=1, steps=240, every=60, seed=3)


def test_renderer_labels(pool):
    from fctac.sim.radar_fc27 import BackgroundPool, render
    rng = np.random.default_rng(0)
    bg = BackgroundPool("")
    img, lab = render(rng, pool, bg, mode="panel")
    assert img.shape == (CH, CW, 3) and img.dtype == np.uint8
    assert len(lab["uv"]) >= 20
    assert set(np.unique(lab["shape"])) <= {0, 1}
    assert (lab["shape"] == 0).sum() <= 11 and (lab["shape"] == 1).sum() <= 11
    img2, lab2 = render(rng, pool, bg, mode="none")
    assert len(lab2["uv"]) == 0 and lab2["ball"] is None


def test_estimate_panel_recovers_shift(pool):
    from fctac.sim.radar_fc27 import BackgroundPool, render
    rng = np.random.default_rng(1)
    bg = BackgroundPool("")
    true = RadarConfig().panel
    w, h = 1920, 1080
    frames = []
    for _ in range(4):
        img, _ = render(rng, pool, bg, mode="panel")
        f = np.full((h, w, 3), (40, 110, 80), np.uint8)
        # paste the canonical crop at the true panel (1080p canonical scale is ~1:1)
        x0, y0 = int(round(true[0] * w - 14.5)), int(round(true[1] * h - 16))
        f[y0:y0 + CH, x0:x0 + CW] = img
        frames.append(f)
    off = (5 / w, -4 / h)
    guess = (true[0] + off[0], true[1] + off[1], true[2] + off[0], true[3] + off[1])
    est, score = estimate_panel(frames, guess)
    assert score > 0.3
    err = max(abs(a - b) * (w if i % 2 == 0 else h) for i, (a, b) in enumerate(zip(est, true)))
    assert err <= 2.0


def test_reader_plumbing(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("onnxruntime")
    from fctac.training.radar_model import TinyRadarNet, export_onnx
    from fctac.vision.radar_fc27 import FC27RadarReader
    torch.manual_seed(0)
    path = str(tmp_path / "radar.onnx")
    export_onnx(TinyRadarNet(), path, {"thr": 0.35})
    cfg = RadarConfig(mode="fc27", fc27_model=path)
    rd = FC27RadarReader(cfg)
    res = rd.read(np.zeros((1080, 1920, 3), np.uint8))
    assert res.ok is False                 # untrained net, empty frame: nothing to read
    rd.swap()
    assert rd.us_shape in (0, 1) and rd.fixed


@pytest.mark.skipif(not os.path.exists("models/registry.json"), reason="no registry")
def test_deployed_radar_model_reads_synthetic(pool):
    from fctac.training.registry import Registry
    e = Registry().deployed("radar")
    if not e:
        pytest.skip("no deployed radar model")
    from fctac.sim.radar_fc27 import BackgroundPool, render
    from fctac.training.radar_model import evaluate, onnx_predictor
    rng = np.random.default_rng(5)
    bg = BackgroundPool("")
    rows = []
    import tempfile
    d = tempfile.mkdtemp()
    for k in range(6):
        img, lab = render(rng, pool, bg, mode="panel")
        p = os.path.join(d, f"{k}.jpg")
        cv2.imwrite(p, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        rows.append({"path": p, "players": [[u, v, int(s), int(hl)] for (u, v), s, hl in
                                            zip(lab["uv"], lab["shape"], lab["highlight"])],
                     "ball": None if lab["ball"] is None else list(lab["ball"])})
    m = evaluate(onnx_predictor(e["path"], ["CPUExecutionProvider"]), rows)
    assert m["f1"] > 0.85 and m["shape_acc"] > 0.9


def test_team_decision_needs_clear_evidence_and_locks():
    from fctac.vision.radar_fc27 import FC27RadarReader
    rd = FC27RadarReader.__new__(FC27RadarReader)
    rd.us_shape, rd.fixed, rd.votes, rd.generation = None, False, np.zeros(2), 0

    def frame(hl_tri, hl_cir):
        return {"shape": np.array([0, 0, 1, 1]), "highlight": np.array([hl_tri, 0.0, hl_cir, 0.0]), "hl_thr": 0.35}

    for _ in range(100):                    # online 1v1 / co-op: both teams highlighted -> never guess
        rd._update_team(frame(0.8, 0.75))
    assert not rd.decided
    for _ in range(40):                     # vs CPU: only your team (triangles) highlighted
        rd._update_team(frame(0.8, 0.0))
    assert rd.decided and rd.us_shape == 0 and rd.generation == 1
    for _ in range(200):                    # later noise must not flip it
        rd._update_team(frame(0.0, 0.9))
    assert rd.us_shape == 0 and rd.generation == 1
    rd.swap()                               # F7 still overrides
    assert rd.us_shape == 1 and rd.fixed
