"""End-to-end evaluation of the vision pipeline against ground truth.

Reports state-reconstruction quality (positions, teams, IDs, controlled
player, ball, calibration, attack direction), decision agreement with the
oracle (same tactics on perfect state) and per-stage latency.

    python -m fctac.benchmark.pipeline_eval data/synthetic/clip01 --frames 900
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
from scipy.optimize import linear_sum_assignment

from fctac import types as T
from fctac.capture.video import VideoSource
from fctac.pipeline import OracleAnalyzer
from fctac.pitch.camera import CameraParams, apply_h
from fctac.state.gt import load_gt
from fctac.timing import LatencyStats
from fctac.vision.analyzer import VisionAnalyzer


def evaluate(clip: str, frames: int = 0, analyzer=None, config: str = "") -> dict:
    gt = load_gt(clip + ".gt.jsonl")
    src = VideoSource(clip + ".mp4")
    an = analyzer or VisionAnalyzer.from_files(config)
    oracle = OracleAnalyzer(gt)
    n = len(gt) if frames <= 0 else min(frames, len(gt))
    lat = LatencyStats(window=100000)
    pos_err, ball_err, calib_err, sign_ok, valid = [], [], [], [], 0
    team_ok, ctrl_ok, n_tracks = [], [], []
    id_map: dict[int, int] = {}          # gt id -> track id
    id_switches = 0
    rec_same, rec_both = 0, 0
    reasons: dict[str, int] = {}
    poss_ok, phase_ok = [], []
    phase_conf: dict[str, int] = {}
    active_vis, active_or = 0, 0
    wall = []
    for i in range(n):
        frame = src.get(i)
        if frame is None:
            break
        h, w = frame.shape[:2]
        t0 = time.perf_counter()
        fa = an.process(frame, i, i / src.fps)
        wall.append((time.perf_counter() - t0) * 1000)
        lat.add(fa.timings_ms)
        fo = oracle.process(None, i, i / src.fps)
        snap = gt[i]
        st = fa.state
        sign_ok.append(int(st.attack_sign == snap["us_attack_sign"]))
        g = np.array([[p["x"], p["y"]] for p in snap["players"]])
        gteam = np.array([p["team"] for p in snap["players"]])
        if st.players:
            raw = np.array([st.to_raw(p.pos) for p in st.players])
            D = np.linalg.norm(g[:, None] - raw[None], axis=2)
            pteam = np.array([0 if p.team == T.TEAM_US else 1 for p in st.players])
            Dm = D + np.where(gteam[:, None] != pteam[None, :], 5.0, 0.0)    # team-consistent matching
            a, b = linear_sum_assignment(Dm)
            ok = D[a, b] < 3.0
            pos_err += list(D[a, b][ok])
            n_tracks.append(len(st.players))
            for gi, pi in zip(a[ok], b[ok]):
                team_ok.append(int((st.players[pi].team == T.TEAM_US) == (gteam[gi] == 0)))
                tid = st.players[pi].id
                if gi in id_map and id_map[gi] != tid:
                    id_switches += 1
                id_map[gi] = tid
            gc = next(k for k, p in enumerate(snap["players"]) if p["controlled"])
            me = st.controlled
            ctrl_ok.append(int(me is not None and np.linalg.norm(st.to_raw(me.pos) - g[gc]) < 2.0))
        if st.ball is not None:
            ball_err.append(float(np.linalg.norm(st.to_raw(st.ball.pos) - [snap["ball"]["x"], snap["ball"]["y"]])))
        if st.H_img2pitch is not None:
            cam = CameraParams(**snap["camera"]).scaled(w, h)
            gx, gy = np.meshgrid(np.linspace(0.1 * w, 0.9 * w, 7), np.linspace(0.45 * h, 0.95 * h, 5))
            pts = np.c_[gx.ravel(), gy.ravel()]
            calib_err.append(float(np.median(np.linalg.norm(apply_h(st.H_img2pitch, pts) - apply_h(cam.H_img2pitch(), pts), axis=1))))
        valid += int(st.valid)
        gp = snap.get("possession", -1)
        if gp in (0, 1):
            poss_ok.append(int(st.possession == (T.TEAM_US if gp == 0 else T.TEAM_THEM)))
        pv, po = an.engine.phase(st), oracle.engine.phase(fo.state)
        pv = "attack" if pv == "receiving" else pv
        po = "attack" if po == "receiving" else po
        phase_ok.append(int(pv == po))
        if pv != po:
            k = f"{po}->{pv}"
            phase_conf[k] = phase_conf.get(k, 0) + 1
        rv, ro = fa.recommendation, fo.recommendation
        va, oa = rv.status == T.STATUS_ACTIVE, ro.status == T.STATUS_ACTIVE
        if not va:
            reasons[rv.reason or rv.status] = reasons.get(rv.reason or rv.status, 0) + 1
        active_vis += int(va)
        active_or += int(oa)
        if va and oa:
            rec_both += 1
            # compare by kind + target role/position (track ids differ from gt ids)
            same = rv.action.kind == ro.action.kind
            if same and rv.action.target_id is not None and ro.action.target_id is not None:
                tv = st.player(rv.action.target_id)
                to = fo.state.player(ro.action.target_id)
                same = tv is not None and to is not None and np.linalg.norm(tv.pos - to.pos) < 2.5
            rec_same += int(same)
    s = lat.summary()
    return {
        "clip": clip, "frames": n,
        "state_valid_rate": round(valid / max(n, 1), 3),
        "attack_dir_acc": round(float(np.mean(sign_ok)), 3),
        "player_pos_err_m": round(float(np.mean(pos_err)), 3) if pos_err else None,
        "player_pos_err_p95_m": round(float(np.percentile(pos_err, 95)), 3) if pos_err else None,
        "tracks_per_frame": round(float(np.mean(n_tracks)), 2) if n_tracks else 0,
        "team_acc": round(float(np.mean(team_ok)), 4) if team_ok else None,
        "id_switches_per_min": round(id_switches / max(n / 30 / 60, 1e-6), 2),
        "controlled_acc": round(float(np.mean(ctrl_ok)), 4) if ctrl_ok else None,
        "ball_err_m_median": round(float(np.median(ball_err)), 3) if ball_err else None,
        "ball_found_rate": round(len(ball_err) / max(n, 1), 3),
        "calib_err_m_median": round(float(np.median(calib_err)), 3) if calib_err else None,
        "calib_rate": round(len(calib_err) / max(n, 1), 3),
        "rec_active_rate_vision": round(active_vis / max(n, 1), 3),
        "rec_active_rate_oracle": round(active_or / max(n, 1), 3),
        "rec_agreement_with_oracle": round(rec_same / max(rec_both, 1), 3),
        "inactive_reasons": reasons,
        "possession_acc": round(float(np.mean(poss_ok)), 3) if poss_ok else None,
        "phase_acc": round(float(np.mean(phase_ok)), 3) if phase_ok else None,
        "phase_errors": dict(sorted(phase_conf.items(), key=lambda kv: -kv[1])[:6]),
        "latency_ms": {k: {"mean": round(v["mean"], 2), "p95": round(v["p95"], 2)} for k, v in s.items()},
        "wall_ms_mean": round(float(np.mean(wall)), 2), "wall_ms_p95": round(float(np.percentile(wall, 95)), 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    ap.add_argument("--frames", type=int, default=0)
    ap.add_argument("--config", default="")
    a = ap.parse_args()
    print(json.dumps(evaluate(a.clip, a.frames, config=a.config), indent=1))


if __name__ == "__main__":
    main()
