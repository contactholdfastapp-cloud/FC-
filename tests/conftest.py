import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


@pytest.fixture(scope="session")
def synth_clip(tmp_path_factory):
    """A short synthetic clip with ground truth (generated once per test session)."""
    from tools.make_synthetic import generate
    d = tmp_path_factory.mktemp("synth")
    out = str(d / "clip")
    generate(out, seconds=6, seed=3, width=640, height=360, fps=30, kit_us="red", kit_them="blue", attack_sign=1)
    return out
