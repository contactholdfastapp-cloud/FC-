import numpy as np

from fctac import types as T
from fctac.tactics.engine import TacticsEngine
from fctac.tactics.value import xg, zone_value


def _state(opp_near_path: bool, ball_owner=None):
    ps = [T.PlayerState(id=1, team=T.TEAM_US, pos=np.array([50.0, 30.0]), vel=np.zeros(2), acc=np.zeros(2), controlled=True)]
    ps += [T.PlayerState(id=10 + k, team=T.TEAM_US, pos=np.array([30.0 + 5 * k, 10.0]), vel=np.zeros(2), acc=np.zeros(2))
           for k in range(6)]
    opp_pos = [np.array([95.0 - 3 * k, 60.0 - 2 * k]) for k in range(7)]
    if opp_near_path:
        opp_pos[0] = np.array([42.0, 31.5])      # sits right in the path of the incoming ball
    ps += [T.PlayerState(id=30 + k, team=T.TEAM_THEM, pos=p, vel=np.zeros(2), acc=np.zeros(2)) for k, p in enumerate(opp_pos)]
    st = T.GameState(frame=0, t=0.0, players=ps, ball=None, controlled_id=1, possession=T.TEAM_US, calib_conf=1.0)
    st.ball = T.BallState(pos=np.array([30.0, 30.0]), vel=np.array([14.0, 0.0]), confidence=1.0,
                          owner_id=ball_owner, owner_team=T.TEAM_UNKNOWN)
    return st


def test_reception_point_and_contested_ball():
    eng = TacticsEngine()
    free = _state(False)
    r = eng.reception(free)
    assert r is not None and abs(r[0][1] - 30.0) < 0.5 and 30 < r[0][0] <= 52
    assert eng.phase(free) == "receiving"
    cands, rec = eng.decide(free)
    assert cands and all("origin" in a.detail for a in cands)   # drawn from the collection point
    assert eng.reception(_state(True)) is None                  # opponent wins the race -> no plan


def test_value_function_lets_shots_win_near_goal():
    spot = np.array([[94.0, 34.0]])          # penalty spot: shooting should beat merely holding
    assert xg(spot)[0] > zone_value(spot)[0]
    far = np.array([[70.0, 34.0]])           # 35 m out: keeping the ball is worth more than a shot
    assert xg(far)[0] < zone_value(far)[0]
