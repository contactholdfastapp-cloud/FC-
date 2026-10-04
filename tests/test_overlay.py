import numpy as np

from fctac import types as T
from fctac.overlay.renderer import Canvas, OverlayConfig, OverlayRenderer
from fctac.pipeline import OracleAnalyzer
from fctac.state.gt import load_gt


def test_canvas_premultiplied_and_composite():
    c = Canvas(200, 100)
    c.rect(10, 10, 50, 50, (0, 0, 255), 128)
    px = c.img[20, 20]
    assert px[3] == 128 and abs(int(px[2]) - 128) <= 1 and px[0] == 0   # premultiplied red
    frame = np.full((100, 200, 3), 100, np.uint8)
    out = c.composite_onto(frame.copy())
    assert abs(int(out[20, 20, 2]) - (100 * 127 // 255 + 128)) <= 2
    assert np.all(out[80, 150] == 100)          # untouched outside dirty rect
    c.clear()
    assert c.img.sum() == 0 and c.dirty is None


def test_overlay_draws_active_recommendation(synth_clip):
    gt = load_gt(synth_clip + ".gt.jsonl")
    an = OracleAnalyzer(gt)
    ov = OverlayRenderer(960, 540, OverlayConfig(show_debug=True))
    drawn = 0
    for i in range(len(gt)):
        fa = an.process(None, i, i / 30)
        c = ov.render(fa, ["debug"])
        if fa.recommendation.status == T.STATUS_ACTIVE:
            assert c.dirty is not None and c.img[..., 3].max() > 0
            drawn += 1
    assert drawn > len(gt) * 0.3


def test_viewer_export_and_keys(synth_clip, tmp_path):
    from fctac.capture.video import VideoSource
    from fctac.replay.viewer import ReplayViewer, PANEL_H
    from fctac.training.events import events_from_gt
    gt = load_gt(synth_clip + ".gt.jsonl")
    src = VideoSource(synth_clip + ".mp4")
    v = ReplayViewer(src, OracleAnalyzer(gt), events_from_gt(gt))
    img = v.compose(10)
    assert img.shape == (540 + PANEL_H, 960, 3)
    # backward step uses the cache, forward continues sequentially
    v.idx = 10
    assert v.handle_key(ord("a")) and v.idx == 9
    assert v.handle_key(ord("d")) and v.idx == 10
    assert v.handle_key(ord(" ")) and v.playing
    assert not v.handle_key(27)
    assert 9 in v.cache.results
    info = v.export(str(tmp_path / "o.mp4"), 0, 30)
    assert info["frames"] == 30
