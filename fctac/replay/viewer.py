"""Replay viewer: recorded gameplay -> analysis -> overlay, frame-accurate.

    python -m fctac.replay.viewer --video data/synthetic/clip01.mp4 --oracle
    python -m fctac.replay.viewer --video my_fc27.mp4 --calib my_fc27.calib.json
    python -m fctac.replay.viewer --video clip.mp4 --oracle --export out.mp4   (headless)

Keys:  SPACE play/pause   D / RIGHT next   A / LEFT prev   W / S speed +/-
       E / Q jump +-5 s   O overlay   G debug panel   C confidence
       2 secondary action   R reset analysis   H help   ESC quit
"""
from __future__ import annotations

import argparse
import os
import time
from typing import Optional

import cv2
import numpy as np

from fctac import types as T
from fctac.capture.video import VideoSource
from fctac.overlay.renderer import ACCENT, DANGER, NEUTRAL, SECOND, OverlayConfig, OverlayRenderer
from fctac.timing import LatencyStats
from fctac.training.events import describe, events_from_gt, load_events

PANEL_H = 132
SPEEDS = (0.1, 0.25, 0.5, 1.0, 2.0)
KEY_LEFT = {2424832, 65361, 81}
KEY_RIGHT = {2555904, 65363, 83}


def kind_color(kind: str):
    if kind in T.DEFENCE_KINDS:
        return DANGER
    if kind == T.SHOOT:
        return SECOND
    if kind in (T.HOLD, T.DRIBBLE):
        return NEUTRAL
    return ACCENT


class AnalysisCache:
    """Sequential analysis with per-frame caching so seeking backwards is free.

    Trackers need frames in order, so jumping forward analyses the skipped
    frames; a jump far ahead resets the analyzer instead (as live mode would).
    """

    def __init__(self, analyzer, source: VideoSource, max_gap: int = 150):
        self.an = analyzer
        self.src = source
        self.results: dict[int, T.FrameAnalysis] = {}
        self.last = -1
        self.max_gap = max_gap
        self.stats = LatencyStats()
        self.changes: dict[int, str] = {}     # frame -> kind where the shown action changed
        self._prev_key = None

    def reset(self):
        self.results.clear()
        self.changes.clear()
        self._prev_key = None
        self.last = -1
        self.an.reset()

    def get(self, idx: int) -> Optional[T.FrameAnalysis]:
        if idx in self.results:
            return self.results[idx]
        if idx > self.last + self.max_gap or idx < self.last:
            self.an.reset()
            self.last = idx - 1
        for k in range(self.last + 1, idx + 1):
            frame = self.src.get(k)
            if frame is None:
                return None
            fa = self.an.process(frame, k, self.src.time_of(k))
            self.results[k] = fa
            r = fa.recommendation
            a = r.action if r.status == T.STATUS_ACTIVE else None
            key = a.key() if a is not None else None
            if key != self._prev_key and a is not None:
                self.changes[k] = a.kind
            self._prev_key = key
            self.stats.add(fa.timings_ms)
            self.last = k
        return self.results.get(idx)


class ReplayViewer:
    def __init__(self, source: VideoSource, analyzer, events: Optional[list] = None,
                 overlay_cfg: OverlayConfig = OverlayConfig()):
        self.src = source
        self.cache = AnalysisCache(analyzer, source)
        self.events = events or []
        self.ocfg = overlay_cfg
        self.ov = OverlayRenderer(source.width, source.height, overlay_cfg)
        self.idx = 0
        self.playing = False
        self.speed_i = SPEEDS.index(1.0)
        self.show_overlay = True
        self.show_help = False
        self.win = "FC27 Tactical Replay"

    # --- composition ---------------------------------------------------------
    def compose(self, idx: int) -> Optional[np.ndarray]:
        frame = self.src.get(idx)
        if frame is None:
            return None
        fa = self.cache.get(idx)
        img = frame.copy()
        if fa is not None and self.show_overlay:
            dbg = self._debug_lines(fa) if self.ocfg.show_debug else None
            self.ov.render(fa, dbg).composite_onto(img)
        panel = self._panel(idx, fa)
        out = np.vstack([img, panel])
        if self.show_help:
            self._help(out)
        return out

    def _debug_lines(self, fa: T.FrameAnalysis) -> list:
        s = self.cache.stats.summary()
        lines = []
        tot = s.get("total", {})
        if tot:
            lines.append(f"analysis {tot['mean']:.1f} ms (p95 {tot['p95']:.1f})  ~{1000 / max(tot['mean'], 1e-3):.0f} fps")
        for k in ("capture", "detect", "radar", "calib", "track", "state", "prediction", "decision"):
            if k in fa.timings_ms:
                lines.append(f"  {k:<10s} {fa.timings_ms[k]:6.2f} ms")
        st = fa.state
        if st is not None:
            b = st.ball
            lines.append(f"ball {'-' if b is None else f'({b.pos[0]:.0f},{b.pos[1]:.0f}) c={b.confidence:.2f}'}")
            me = st.controlled
            lines.append(f"ctrl {'-' if me is None else f'{me.label} ({me.pos[0]:.0f},{me.pos[1]:.0f})'}"
                         f"  calib {st.calib_conf:.2f}  players {len(st.players)}")
        rec = fa.recommendation
        lines.append(f"rec {rec.status} {rec.action.text() if rec.action else ''} {rec.confidence * 100:.0f}% {rec.reason}")
        for a in fa.candidates[:4]:
            lines.append(f"  {a.text():<20s} ev={a.score:+.3f} p={a.p_success:.2f}")
        return lines

    def _panel(self, idx: int, fa: Optional[T.FrameAnalysis]) -> np.ndarray:
        w = self.src.width
        p = np.full((PANEL_H, w, 3), 24, np.uint8)
        n = max(len(self.src), 1)
        x0, x1, y = 12, w - 12, 14
        cv2.line(p, (x0, y), (x1, y), (80, 80, 80), 4)
        done = self.cache.last
        if done >= 0:
            cv2.line(p, (x0, y), (x0 + int((x1 - x0) * done / n), y), (130, 130, 130), 4)
        # recommendation history ticks
        for k, kind in self.cache.changes.items():
            x = x0 + int((x1 - x0) * k / n)
            cv2.line(p, (x, y - 7), (x, y - 2), kind_color(kind), 2)
        # human events
        for e in self.events:
            if e.get("team", 0) != 0:
                continue
            x = x0 + int((x1 - x0) * e["frame"] / n)
            ok = e.get("outcome") in ("complete", "goal", "won")
            c = (200, 200, 200) if ok else (60, 60, 230)
            cv2.fillConvexPoly(p, np.array([[x, y + 4], [x - 4, y + 10], [x + 4, y + 10]], np.int32), c)
        cx = x0 + int((x1 - x0) * idx / n)
        cv2.line(p, (cx, y - 9), (cx, y + 11), (255, 255, 255), 2)

        def put(s, row, col=(225, 225, 225), x=12, sc=0.5):
            cv2.putText(p, s, (x, 44 + row * 22), cv2.FONT_HERSHEY_SIMPLEX, sc, col, 1, cv2.LINE_AA)

        state = "PLAY" if self.playing else "PAUSED"
        put(f"{state}  {SPEEDS[self.speed_i]:g}x   frame {idx}/{n - 1}   t={self.src.time_of(idx):6.2f}s   "
            f"[H] help", 0)
        if fa is not None:
            rec = fa.recommendation
            if rec.status == T.STATUS_ACTIVE and rec.action is not None:
                put(f"REC  {rec.action.text()}   {rec.confidence_level} ({rec.confidence * 100:.0f}%)", 1,
                    kind_color(rec.action.kind))
            else:
                put(f"REC  {rec.status}  {rec.reason}", 1, (160, 160, 160))
            alts = "   ".join(f"{a.text()} {a.score:+.3f}" for a in rec.alternatives[:3])
            put(f"ALT  {alts}", 2, (170, 170, 170))
        nxt = next((e for e in self.events if e.get("team", 0) == 0 and e["frame"] >= idx - 2), None)
        if nxt is not None:
            dt = (nxt["frame"] - idx) / self.src.fps
            col = (200, 200, 200) if nxt.get("outcome") in ("complete", "goal", "won") else (90, 90, 240)
            put(f"HUMAN next: {describe(nxt)}  in {dt:+.2f}s", 3, col)
        return p

    def _help(self, img):
        lines = __doc__.strip().splitlines()[-4:]
        y = 30
        for ln in lines:
            cv2.putText(img, ln.strip(), (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(img, ln.strip(), (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            y += 22

    def handle_key(self, key: int) -> bool:
        """Apply one key press. Returns False when the viewer should quit."""
        k = key & 0xFF
        last = max(len(self.src) - 1, 0)
        if key == 27 or k == 27:
            return False
        if k == ord(" "):
            self.playing = not self.playing
        elif key in KEY_RIGHT or k == ord("d"):
            self.playing, self.idx = False, min(self.idx + 1, last)
        elif key in KEY_LEFT or k == ord("a"):
            self.playing, self.idx = False, max(self.idx - 1, 0)
        elif k == ord("w"):
            self.speed_i = min(self.speed_i + 1, len(SPEEDS) - 1)
        elif k == ord("s"):
            self.speed_i = max(self.speed_i - 1, 0)
        elif k == ord("e"):
            self.idx = min(self.idx + int(5 * self.src.fps), last)
        elif k == ord("q"):
            self.idx = max(self.idx - int(5 * self.src.fps), 0)
        elif k == ord("o"):
            self.show_overlay = not self.show_overlay
        elif k == ord("g"):
            self.ocfg.show_debug = not self.ocfg.show_debug
        elif k == ord("c"):
            self.ocfg.show_confidence = not self.ocfg.show_confidence
        elif k == ord("2"):
            self.ocfg.show_secondary = not self.ocfg.show_secondary
        elif k == ord("r"):
            self.cache.reset()
        elif k == ord("h"):
            self.show_help = not self.show_help
        return True

    # --- interactive loop ------------------------------------------------------
    def run(self):
        cv2.namedWindow(self.win, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(self.win, self.src.width, self.src.height + PANEL_H)
        n = len(self.src)
        cv2.createTrackbar("frame", self.win, 0, max(n - 1, 1), lambda v: None)
        last_tb = 0
        t_next = time.perf_counter()
        while True:
            img = self.compose(self.idx)
            if img is None:
                self.playing = False
                self.idx = max(0, min(self.idx, len(self.src) - 1))
                img = self.compose(self.idx)
                if img is None:
                    break
            cv2.imshow(self.win, img)
            if cv2.getTrackbarPos("frame", self.win) != self.idx:
                cv2.setTrackbarPos("frame", self.win, self.idx)
            last_tb = self.idx
            delay = 1
            if self.playing:
                t_next += 1.0 / (self.src.fps * SPEEDS[self.speed_i])
                delay = max(1, int((t_next - time.perf_counter()) * 1000))
            key = cv2.waitKeyEx(delay if self.playing else 30)
            tb = cv2.getTrackbarPos("frame", self.win)
            if tb != last_tb:
                self.idx = tb
                continue
            if key == -1:
                if self.playing:
                    self.idx += 1
                continue
            if not self.handle_key(key):
                break
            t_next = time.perf_counter() if self.playing and key & 0xFF == ord(" ") else t_next
            if cv2.getWindowProperty(self.win, cv2.WND_PROP_VISIBLE) < 1:
                break
        cv2.destroyAllWindows()

    def export(self, out_path: str, start: int = 0, end: Optional[int] = None) -> dict:
        end = len(self.src) if end is None else min(end, len(self.src))
        vw = None
        t0 = time.perf_counter()
        frames = 0
        for i in range(start, end):
            img = self.compose(i)
            if img is None:
                break
            if vw is None:
                vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), self.src.fps,
                                     (img.shape[1], img.shape[0]))
            vw.write(img)
            frames += 1
        if vw is not None:
            vw.release()
        return {"frames": frames, "seconds": round(time.perf_counter() - t0, 2),
                "latency": self.cache.stats.summary()}


def build_analyzer(args, source: VideoSource):
    base = os.path.splitext(args.video)[0]
    if args.oracle:
        from fctac.pipeline import OracleAnalyzer
        from fctac.state.gt import load_gt
        gt = load_gt(args.gt or base + ".gt.jsonl")
        return OracleAnalyzer(gt), events_from_gt(gt)
    from fctac.vision.analyzer import VisionAnalyzer
    an = VisionAnalyzer.from_files(args.config, args.calib or None, source.width, source.height)
    events = load_events(args.events) if args.events else []
    if not events and os.path.exists(base + ".gt.jsonl"):
        from fctac.state.gt import load_gt
        events = events_from_gt(load_gt(base + ".gt.jsonl"))
    return an, events


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--oracle", action="store_true", help="use synthetic ground truth instead of vision")
    ap.add_argument("--gt", default="")
    ap.add_argument("--events", default="", help="human action labels (jsonl)")
    ap.add_argument("--config", default="", help="pipeline config json")
    ap.add_argument("--calib", default="", help="calibration json (manual landmarks/radar rect)")
    ap.add_argument("--export", default="", help="render to this video instead of opening a window")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=-1)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--anchor", default="player", choices=("player", "top"))
    a = ap.parse_args(argv)
    src = VideoSource(a.video)
    an, events = build_analyzer(a, src)
    v = ReplayViewer(src, an, events, OverlayConfig(show_debug=a.debug, label_anchor=a.anchor))
    if a.export:
        info = v.export(a.export, a.start, None if a.end < 0 else a.end)
        import json
        print(json.dumps(info, indent=1))
    else:
        v.idx = a.start
        v.run()


if __name__ == "__main__":
    main()
