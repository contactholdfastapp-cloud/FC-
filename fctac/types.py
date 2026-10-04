"""Core data types shared by every pipeline stage.

Coordinate conventions
----------------------
* Screen: pixels of the captured gameplay region, origin top-left.
* Raw pitch: metres. x in [0, 105] runs left->right as seen by the broadcast
  camera, y in [0, 68] runs from the near touchline (bottom of screen) to the
  far touchline (top of screen), z is up.
* Attack-aligned pitch ("pitch" in the tactical layer): metres, rotated so
  that x = 0 is our own goal line and x = 105 the opponent goal line.
  ``GameState.to_attack`` / ``to_raw`` convert between the two.
  Normalised coordinates (0..1) are ``pos / PITCH_SIZE``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0
PITCH_SIZE = np.array([PITCH_LENGTH, PITCH_WIDTH], dtype=np.float64)

TEAM_UNKNOWN = -1
TEAM_US = 0      # team of the human player (the controlled player's team)
TEAM_THEM = 1
TEAM_REF = 2

CLS_PLAYER = 0
CLS_BALL = 1
CLS_REF = 2
CLS_NAMES = {CLS_PLAYER: "player", CLS_BALL: "ball", CLS_REF: "referee"}


@dataclass(slots=True)
class Detection:
    """One object found in a frame (screen space)."""
    cls: int
    x: float            # anchor x: foot point for people, centre for the ball
    y: float            # anchor y
    w: float
    h: float
    conf: float
    team: int = TEAM_UNKNOWN
    team_conf: float = 0.0
    controlled: bool = False
    controlled_conf: float = 0.0
    feature: Optional[np.ndarray] = None   # e.g. kit colour descriptor
    pitch: Optional[np.ndarray] = None     # raw pitch position (filled by calibration)


@dataclass(slots=True)
class PlayerState:
    id: int
    team: int
    pos: np.ndarray                 # attack-aligned metres
    vel: np.ndarray                 # m/s
    acc: np.ndarray                 # m/s^2
    controlled: bool = False
    confidence: float = 1.0
    role: str = ""
    screen: Optional[np.ndarray] = None
    visible: bool = True
    # derived features (filled by state.features)
    dist_to_ball: float = 0.0
    dist_to_goal: float = 0.0
    nearby_opponents: int = 0
    pressure: float = 0.0
    open_space: float = 0.0
    predicted: dict = field(default_factory=dict)   # horizon_s -> np.ndarray(2)

    @property
    def speed(self) -> float:
        return float(np.hypot(self.vel[0], self.vel[1]))

    @property
    def direction(self) -> float:
        return float(np.arctan2(self.vel[1], self.vel[0]))

    @property
    def label(self) -> str:
        return self.role or f"P{self.id:02d}"


@dataclass(slots=True)
class BallState:
    pos: np.ndarray                 # attack-aligned metres
    vel: np.ndarray
    confidence: float
    owner_id: Optional[int] = None
    owner_team: int = TEAM_UNKNOWN
    screen: Optional[np.ndarray] = None
    height: float = 0.0


@dataclass(slots=True)
class GameState:
    frame: int
    t: float
    players: list
    ball: Optional[BallState]
    attack_sign: int = 1            # +1: we attack towards raw x=105
    H_img2pitch: Optional[np.ndarray] = None   # 3x3, screen -> raw pitch
    calib_conf: float = 0.0         # reliability of the screen <-> pitch mapping (drawing)
    state_conf: float = 1.0         # reliability of pitch positions (decisions)
    possession: int = TEAM_UNKNOWN  # team in possession
    controlled_id: Optional[int] = None
    valid: bool = True
    invalid_reason: str = ""
    formation_us: str = ""
    formation_them: str = ""

    # --- coordinate helpers -------------------------------------------------
    def to_attack(self, p_raw: np.ndarray) -> np.ndarray:
        p = np.asarray(p_raw, dtype=np.float64)
        if self.attack_sign > 0:
            return p.copy()
        return PITCH_SIZE - p

    def to_raw(self, p_att: np.ndarray) -> np.ndarray:
        # the transform is an involution
        return self.to_attack(p_att)

    def vel_to_raw(self, v_att: np.ndarray) -> np.ndarray:
        return np.asarray(v_att, dtype=np.float64) * self.attack_sign

    # --- lookups -------------------------------------------------------------
    def player(self, pid: Optional[int]) -> Optional[PlayerState]:
        if pid is None:
            return None
        for p in self.players:
            if p.id == pid:
                return p
        return None

    @property
    def controlled(self) -> Optional[PlayerState]:
        return self.player(self.controlled_id)

    def team(self, team: int) -> list:
        return [p for p in self.players if p.team == team]


# Action kinds ---------------------------------------------------------------
PASS = "PASS"
THROUGH = "THROUGH"
LOB = "LOB"
CROSS = "CROSS"
SHOOT = "SHOOT"
DRIBBLE = "DRIBBLE"
HOLD = "HOLD"
SWITCH = "SWITCH"
PRESS = "PRESS"
COVER = "COVER"
JOCKEY = "JOCKEY"

ATTACK_KINDS = (PASS, THROUGH, LOB, CROSS, SHOOT, DRIBBLE, HOLD)
DEFENCE_KINDS = (SWITCH, PRESS, COVER, JOCKEY)
ALL_KINDS = ATTACK_KINDS + DEFENCE_KINDS
PASS_KINDS = (PASS, THROUGH, LOB, CROSS)


@dataclass(slots=True)
class Action:
    kind: str
    target_id: Optional[int] = None
    target_label: str = ""
    target_point: Optional[np.ndarray] = None   # attack-aligned metres
    score: float = 0.0          # expected value used for ranking
    p_success: float = 0.0
    value_success: float = 0.0
    detail: dict = field(default_factory=dict)
    features: Optional[np.ndarray] = None

    def key(self) -> tuple:
        """Identity of the action for temporal stabilisation."""
        return (self.kind, self.target_id)

    def text(self) -> str:
        if self.kind in (HOLD, DRIBBLE, PRESS, JOCKEY):
            return self.kind
        if self.kind == SHOOT:
            place = self.detail.get("placement", "")
            return f"SHOOT {place}".strip()
        if self.kind == COVER:
            return f"COVER {self.target_label}".strip()
        return f"{self.kind} -> {self.target_label}" if self.target_label else self.kind


STATUS_ACTIVE = "ACTIVE"
STATUS_ANALYSING = "ANALYSING"
STATUS_HIDDEN = "HIDDEN"


@dataclass(slots=True)
class Recommendation:
    status: str = STATUS_ANALYSING
    action: Optional[Action] = None
    alternatives: list = field(default_factory=list)
    confidence: float = 0.0
    reason: str = ""
    since_t: float = 0.0

    @property
    def confidence_level(self) -> str:
        if self.confidence >= 0.7:
            return "HIGH"
        if self.confidence >= 0.45:
            return "MEDIUM"
        return "LOW"


@dataclass(slots=True)
class FrameAnalysis:
    """Everything the pipeline knows about one frame; consumed by the UI."""
    frame: int
    t: float
    detections: list = field(default_factory=list)
    tracks: list = field(default_factory=list)
    state: Optional[GameState] = None
    candidates: list = field(default_factory=list)
    recommendation: Recommendation = field(default_factory=Recommendation)
    timings_ms: dict = field(default_factory=dict)
