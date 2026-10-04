import numpy as np

from fctac import types as T
from fctac.pitch.camera import CameraParams, apply_h
from fctac.sim.match import MatchSim, SimConfig


def test_camera_homography_roundtrip():
    cam = CameraParams(width=1280, height=720, focal=1800).look_at(np.array([60.0, 30.0, 0.0]))
    pts = np.array([[60.0, 30.0], [40.0, 10.0], [80.0, 50.0]])
    uv, z = cam.project(np.c_[pts, np.zeros(3)])
    assert np.all(z > 0)
    assert np.allclose(apply_h(cam.H_pitch2img(), pts), uv, atol=1e-6)
    assert np.allclose(apply_h(cam.H_img2pitch(), uv), pts, atol=1e-6)
    # look-at point projects to the image centre; +x appears to the right, far side up
    c, _ = cam.project(np.array([[60.0, 30.0, 0.0]]))
    assert np.allclose(c[0], [640, 360], atol=1e-6)
    a, _ = cam.project(np.array([[50.0, 30.0, 0.0], [70.0, 30.0, 0.0], [60.0, 50.0, 0.0]]))
    assert a[0, 0] < a[1, 0] and a[2, 1] < 360


def test_sim_deterministic_and_plausible():
    def run(seed):
        s = MatchSim(SimConfig(seed=seed))
        out = []
        for _ in range(300):
            s.step()
            out.append(s.snapshot())
        return out
    a, b = run(5), run(5)
    assert a[-1]["ball"] == b[-1]["ball"]
    snap = a[-1]
    assert len(snap["players"]) == 22
    assert sum(p["controlled"] for p in snap["players"]) == 1
    for p in snap["players"]:
        assert -3 <= p["x"] <= 108 and -3 <= p["y"] <= 71
        assert np.hypot(p["vx"], p["vy"]) < 9.5


def test_attack_alignment_involution():
    gs = T.GameState(frame=0, t=0, players=[], ball=None, attack_sign=-1)
    p = np.array([10.0, 20.0])
    assert np.allclose(gs.to_attack(p), [95.0, 48.0])
    assert np.allclose(gs.to_raw(gs.to_attack(p)), p)
