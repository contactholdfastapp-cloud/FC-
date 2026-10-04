from fctac import types as T
from fctac.tactics.stabilizer import Stabilizer, StabilizerConfig


def A(kind, tid, score, p=0.8):
    return T.Action(kind=kind, target_id=tid, target_label=str(tid), score=score, p_success=p)


def test_hysteresis_keep_then_switch():
    s = Stabilizer(StabilizerConfig(min_hold_s=0.3, confirm_n=2))
    ctx = ("attack", 1)
    r = s.update(0.0, [A(T.PASS, 9, 0.082)], 0.9, ctx)
    assert r.action.key() == (T.PASS, 9)
    # slightly better alternative (82 -> 84): keep current
    for k in range(5):
        r = s.update(0.5 + k * 0.05, [A(T.PASS, 7, 0.084), A(T.PASS, 9, 0.082)], 0.9, ctx)
        assert r.action.key() == (T.PASS, 9)
    # clearly better (94): switch, but only once confirmed over 2 updates
    r = s.update(1.0, [A(T.PASS, 7, 0.094), A(T.PASS, 9, 0.082)], 0.9, ctx)
    assert r.action.key() == (T.PASS, 9)
    r = s.update(1.05, [A(T.PASS, 7, 0.094), A(T.PASS, 9, 0.082)], 0.9, ctx)
    assert r.action.key() == (T.PASS, 7)


def test_lane_closed_invalidates_immediately():
    s = Stabilizer()
    ctx = ("attack", 1)
    s.update(0.0, [A(T.PASS, 9, 0.08)], 0.9, ctx)
    r = s.update(0.05, [A(T.PASS, 7, 0.05), A(T.PASS, 9, 0.081, p=0.1)], 0.9, ctx)
    assert r.action.key() == (T.PASS, 7)


def test_context_change_and_confidence_hysteresis():
    s = Stabilizer(StabilizerConfig(conf_on=0.4, conf_off=0.28))
    r = s.update(0.0, [A(T.PASS, 9, 0.08)], 0.35, ("attack", 1))
    assert r.status == T.STATUS_HIDDEN            # below conf_on
    r = s.update(0.1, [A(T.PASS, 9, 0.08)], 0.45, ("attack", 1))
    assert r.status == T.STATUS_ACTIVE
    r = s.update(0.2, [A(T.PASS, 9, 0.08)], 0.30, ("attack", 1))
    assert r.status == T.STATUS_ACTIVE            # still above conf_off
    r = s.update(0.3, [A(T.PRESS, 4, 0.7)], 0.9, ("defence", 2))
    assert r.action.kind == T.PRESS               # new context decides immediately
