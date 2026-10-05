"""Tactics must keep passing the football exam (tools/football_exam.py) and stay stable
when the situations are mirrored or nudged."""
from tools.football_exam import robustness, run_exam


def test_exam_all_situations():
    res = run_exam(verbose=False)
    failed = [k for k, v in res.items() if not v]
    assert not failed, f"failed situations: {failed}"


def test_exam_robust_to_mirroring_and_jitter():
    r = robustness(n_variants=6)
    assert sum(r.values()) / len(r) >= 0.85, r
