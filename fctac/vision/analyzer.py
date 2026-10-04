"""The pixels-only analysis pipeline.

frame -> radar read + main-view detection -> team colours -> calibration
      -> tracking (fusion) -> GameState -> movement prediction -> tactics

Used identically by replay mode and live mode.  Every stage is timed.
"""
from __future__ import annotations

import json
from typing import Optional

import numpy as np

from fctac import types as T
from fctac.config import AppConfig, load_config
from fctac.pitch.calibration import RadarViewCalibrator
from fctac.pitch.camera import apply_h
from fctac.prediction.kinematic import KinematicPredictor
from fctac.state.builder import StateBuilder
from fctac.tactics.engine import TacticsEngine
from fctac.timing import StageTimer
from fctac.tracking.tracker import Tracker
from fctac.vision.colors import TeamPrototypes
from fctac.vision.detector import ColorDetector
from fctac.vision.radar import RadarReader


def _from_registry(kind: str):
    from fctac.training.registry import Registry
    e = Registry().deployed(kind)
    return e["path"] if e else None


def make_ranker(cfg: AppConfig):
    r = cfg.runtime.ranker
    if r in ("", "heuristic"):
        return None
    path = _from_registry("ranker") if r == "registry" else r
    if not path:
        return None
    from fctac.tactics.learned_ranker import LearnedRanker
    return LearnedRanker(path, mode=cfg.runtime.ranker_mode or None)


def make_predictor(cfg: AppConfig):
    from fctac.prediction.kinematic import KinematicPredictor
    p = cfg.runtime.predictor
    if p in ("", "kinematic"):
        return KinematicPredictor()
    path = _from_registry("predictor") if p == "registry" else p
    if not path:
        return KinematicPredictor()
    from fctac.prediction.learned import LearnedPredictor
    return LearnedPredictor(path)


def make_detector(cfg: AppConfig):
    if cfg.runtime.detector == "onnx":
        from fctac.vision.learned import OnnxDetector
        path = cfg.runtime.onnx_model or _from_registry("detector")
        if not path:
            raise RuntimeError("runtime.detector=onnx but no onnx_model given and no deployed detector in models/registry.json")
        return OnnxDetector(path, cfg.detector)
    return ColorDetector(cfg.detector)


class VisionAnalyzer:
    name = "vision"

    def __init__(self, cfg: Optional[AppConfig] = None, manual_H: Optional[np.ndarray] = None,
                 ranker=None, predictor=None):
        self.cfg = cfg or AppConfig()
        self.manual_H = manual_H
        self.ranker = ranker if ranker is not None else make_ranker(self.cfg)
        self.predictor = predictor or make_predictor(self.cfg)
        self._build()

    def _build(self):
        c = self.cfg
        self.radar = RadarReader(c.radar) if c.radar.enabled else None
        self.detector = make_detector(c)
        self.calib = RadarViewCalibrator(c.calib)
        if self.manual_H is not None:
            self.calib.set_manual(self.manual_H)
        self.tracker = Tracker(c.tracker)
        self.state = StateBuilder(c.state)
        self.engine = TacticsEngine(c.engine, ranker=self.ranker, predictor=self.predictor)
        self.kits = TeamPrototypes(rate=0.05)
        self.n = 0
        self.last_dets: list = []

    def reset(self):
        self._build()

    @classmethod
    def from_files(cls, config_path: str = "", calib_path: Optional[str] = None, width: int = 0, height: int = 0,
                   overrides: Optional[dict] = None, **kw) -> "VisionAnalyzer":
        cfg = load_config(config_path or None, overrides)
        H = None
        if calib_path:
            with open(calib_path) as f:
                cal = json.load(f)
            if "H_img2pitch" in cal:
                H = np.array(cal["H_img2pitch"], float)
                if width and cal.get("width") and cal["width"] != width:
                    s = cal["width"] / width             # calibration made at another resolution
                    H = H @ np.diag([s, s, 1.0])
            if "radar_rect" in cal:
                cfg.radar.rect = tuple(cal["radar_rect"])
            if "attack_sign" in cal:
                cfg.state.attack_sign = int(cal["attack_sign"])
        return cls(cfg, manual_H=H, **kw)

    # ------------------------------------------------------------------------
    def _ball_prior(self, t) -> Optional[np.ndarray]:
        b = self.tracker.ball
        if b.kf is None or self.calib.H is None:
            return None
        p = b.kf.pos + b.kf.vel * max(0.0, t - (self.tracker.t or t))
        try:
            return apply_h(np.linalg.inv(self.calib.H), p)
        except np.linalg.LinAlgError:
            return None

    def process(self, frame: np.ndarray, idx: int, t: float) -> T.FrameAnalysis:
        tm = StageTimer()
        h, w = frame.shape[:2]
        radar = None
        if self.radar is not None:
            with tm.stage("radar"):
                b = self.tracker.ball
                radar = self.radar.read(frame, b.kf.pos if b.kf is not None else None)
        # main-view detection may run at a lower rate; skipped frames use the radar
        # only (never re-feed stale detections -- the camera has moved since)
        dets: list = []
        detected = self.n % max(1, self.cfg.runtime.detect_every) == 0
        if detected:
            with tm.stage("detect"):
                dets = self.detector.detect(frame, self._ball_prior(t))
                self.last_dets = dets
        self.n += 1
        players = [d for d in dets if d.cls == T.CLS_PLAYER]
        with tm.stage("teams"):
            if self.kits.ready() and players:
                feats = np.array([d.feature for d in players])
                team, conf = self.kits.classify(feats)
                for d, tt, cc in zip(players, team, conf):
                    d.team, d.team_conf = int(tt), float(cc)
        with tm.stage("calib"):
            if not detected:
                H, cconf = self.calib.H, self.calib.conf
            elif radar is not None and radar.ok:
                H, cconf = self.calib.update(players, radar.points, radar.teams, w, h)
                if cconf >= 0.5:
                    # detections registered to radar dots inherit the radar's team label
                    for di, ri in self.calib.last_matches:
                        if di < len(players):
                            players[di].team, players[di].team_conf = int(radar.teams[ri]), 1.0
                self._learn_kits(players, radar)
            else:
                H, cconf = self.calib._coast() if self.calib.H is not None else (None, 0.0)
        with tm.stage("track"):
            tracks = self.tracker.step(t, radar, dets, H, cconf)
        with tm.stage("state"):
            st = self.state.build(idx, t, tracks, self.tracker.ball, self.tracker.controlled_id, H, cconf)
            if radar is not None and radar.ok:
                st.state_conf = 0.95          # radar sees every player in pitch coordinates
            else:
                # main view only: positions depend on calibration and coverage is partial
                st.state_conf = cconf
                if cconf < 0.5:
                    st.valid, st.invalid_reason = False, "radar + calibration unavailable"
        with tm.stage("prediction"):
            self.predictor.annotate(st)
        with tm.stage("decision"):
            cands, rec = self.engine.decide(st)
        tm.ms["total"] = sum(tm.ms.values())
        return T.FrameAnalysis(frame=idx, t=t, detections=dets, tracks=list(tracks), state=st,
                               candidates=cands, recommendation=rec, timings_ms=tm.ms)

    def _learn_kits(self, players, radar):
        """Kit colours are learned from detections the calibration matched to radar dots
        (the radar provides the team label) -> no per-match colour setup needed."""
        if not self.calib.last_matches or self.calib.conf < 0.5:
            return
        feats, teams = [], []
        for di, ri in self.calib.last_matches:
            if di < len(players) and players[di].feature is not None:
                feats.append(players[di].feature)
                teams.append(int(radar.teams[ri]))
        if len(feats) < 4:
            return
        feats = np.array(feats)
        teams = np.array(teams)
        for t_, attr in ((T.TEAM_US, "us"), (T.TEAM_THEM, "them")):
            m = teams == t_
            if m.sum() >= 2:
                med = np.median(feats[m], 0)
                cur = getattr(self.kits, attr)
                setattr(self.kits, attr, med if cur is None else (1 - self.kits.rate) * cur + self.kits.rate * med)
