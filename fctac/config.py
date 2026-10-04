"""Application configuration: nested dataclasses <-> JSON.

Every tunable lives in a dataclass next to the code that uses it; this module
just aggregates them so one JSON file (configs/*.json) can override any field.
"""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field

from fctac.overlay.renderer import OverlayConfig
from fctac.pitch.calibration import CalibConfig
from fctac.state.builder import StateConfig
from fctac.tactics.engine import EngineConfig
from fctac.tracking.tracker import TrackerConfig
from fctac.vision.detector import DetectorConfig
from fctac.vision.radar import RadarConfig


@dataclass
class CaptureConfig:
    backend: str = "auto"              # auto | wgc | dxgi | mss | video
    window_title: str = "FC 27"        # Windows Graphics Capture target window (substring)
    monitor: int = 0
    region: tuple = ()                 # (x, y, w, h) screen pixels; () = whole window/monitor
    width: int = 2560                  # expected gameplay resolution (user plays at 1440p)
    height: int = 1440
    target_fps: int = 60


@dataclass
class RuntimeConfig:
    detector: str = "color"            # color | onnx
    onnx_model: str = ""               # path to a registered detector model
    detect_every: int = 1              # run the main-view detector every N frames (tracker bridges)
    decision_hz: float = 30.0


@dataclass
class AppConfig:
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    radar: RadarConfig = field(default_factory=RadarConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    calib: CalibConfig = field(default_factory=CalibConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    state: StateConfig = field(default_factory=StateConfig)
    engine: EngineConfig = field(default_factory=EngineConfig)
    overlay: OverlayConfig = field(default_factory=OverlayConfig)


def _update(obj, data: dict):
    for k, v in data.items():
        if not hasattr(obj, k):
            raise KeyError(f"unknown config key {type(obj).__name__}.{k}")
        cur = getattr(obj, k)
        if dataclasses.is_dataclass(cur) and isinstance(v, dict):
            _update(cur, v)
        elif isinstance(cur, tuple) and isinstance(v, list):
            setattr(obj, k, tuple(v))
        else:
            setattr(obj, k, v)
    return obj


def load_config(path: str | None = None, overrides: dict | None = None) -> AppConfig:
    cfg = AppConfig()
    if path:
        with open(path) as f:
            _update(cfg, json.load(f))
    if overrides:
        _update(cfg, overrides)
    return cfg


def save_config(cfg: AppConfig, path: str):
    with open(path, "w") as f:
        json.dump(dataclasses.asdict(cfg), f, indent=1)
