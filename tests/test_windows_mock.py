"""Executes the Windows-only code paths against fake Win32 / capture libraries.

This cannot prove the real APIs behave as expected (that needs a Windows run),
but it catches Python-level mistakes (wrong names, struct/arg handling,
buffer wiring) in code the Linux CI could otherwise never execute.
"""
import ctypes
import importlib
import sys
import types

import numpy as np
import pytest


class FakeFunc:
    def __init__(self, name, log, impl=None):
        self.name, self.log, self.impl = name, log, impl
        self.argtypes = None
        self.restype = None

    def __call__(self, *args):
        self.log.append((self.name, args))
        return self.impl(*args) if self.impl else 1


class FakeDLL:
    def __init__(self, log, impls):
        self._log, self._impls = log, impls

    def __getattr__(self, name):
        f = FakeFunc(name, self._log, self._impls.get(name))
        setattr(self, name, f)
        return f


@pytest.fixture
def fake_windows(monkeypatch):
    log = []
    buffers = []

    def create_dib(hdc, bmi, usage, ppbits, hsec, off):
        h = -bmi._obj.bmiHeader.biHeight
        w = bmi._obj.bmiHeader.biWidth
        buf = (ctypes.c_uint8 * (w * h * 4))()
        buffers.append(buf)
        ppbits._obj.value = ctypes.addressof(buf)
        return 99

    def enum_windows(cb, lparam):
        cb(77, 0)
        return 1

    def get_text(hwnd, buf, n):
        buf.value = "EA SPORTS FC 27"
        return len(buf.value)

    def client_to_screen(hwnd, p):
        p._obj.x, p._obj.y = 100, 50
        return 1

    def get_client_rect(hwnd, r):
        r._obj.right, r._obj.bottom = 2560, 1440
        return 1

    def frame_attr(hwnd, attr, r, size):
        r._obj.left, r._obj.top, r._obj.right, r._obj.bottom = 92, 19, 2668, 1498   # frame incl. title bar
        return 0

    impls = {"CreateDIBSection": create_dib, "DwmGetWindowAttribute": frame_attr, "EnumWindows": enum_windows, "GetWindowTextW": get_text,
             "GetWindowTextLengthW": lambda h: 15, "IsWindowVisible": lambda h: 1,
             "PeekMessageW": lambda *a: 0, "ClientToScreen": client_to_screen, "GetClientRect": get_client_rect,
             "GetAsyncKeyState": lambda vk: 0}
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(ctypes, "WinDLL", lambda name, use_last_error=False: FakeDLL(log, impls), raising=False)
    monkeypatch.setattr(ctypes, "WINFUNCTYPE", ctypes.CFUNCTYPE, raising=False)
    monkeypatch.setattr(ctypes, "windll", types.SimpleNamespace(user32=FakeDLL(log, impls), shcore=FakeDLL(log, impls),
                                                                kernel32=FakeDLL(log, impls)), raising=False)
    monkeypatch.setattr(ctypes, "get_last_error", lambda: 0, raising=False)
    for m in ("fctac.overlay.win32", "fctac.capture.windows"):
        sys.modules.pop(m, None)
    yield log
    for m in ("fctac.overlay.win32", "fctac.capture.windows"):
        sys.modules.pop(m, None)


def test_layered_overlay_present(fake_windows):
    win32 = importlib.import_module("fctac.overlay.win32")
    from fctac.overlay.renderer import Canvas
    win32.set_dpi_aware()
    ov = win32.LayeredOverlay(10, 20, 64, 32)
    c = Canvas(64, 32)
    c.rect(4, 4, 20, 12, (0, 255, 0), 200)
    ov.present(c.img, c.dirty)
    assert ov.buffer[6, 6, 3] == 200 and ov.buffer[30, 60, 3] == 0
    names = [n for n, _ in fake_windows]
    assert "CreateWindowExW" in names and names.count("UpdateLayeredWindow") >= 2
    ex_style = next(a for n, a in fake_windows if n == "CreateWindowExW")[0]
    assert ex_style & win32.WS_EX_TRANSPARENT and ex_style & win32.WS_EX_LAYERED   # click-through
    ov.close()
    assert win32.find_window(("FC 27",)) == 77
    assert win32.client_rect_on_screen(77) == (100, 50, 2560, 1440)


def test_wgc_source_delivers_bgr_frames(fake_windows, monkeypatch):
    class Frame:
        def __init__(self):
            self.frame_buffer = np.full((1479, 2576, 4), 7, np.uint8)   # whole window incl. title bar

    class WindowsCapture:
        def __init__(self, **kw):
            self.kw = kw
            self.handlers = {}

        def event(self, f):
            self.handlers[f.__name__] = f
            return f

        def start_free_threaded(self):
            ctl = types.SimpleNamespace(stop=lambda: None)
            for _ in range(3):
                self.handlers["on_frame_arrived"](Frame(), ctl)
            return ctl
    monkeypatch.setitem(sys.modules, "windows_capture", types.SimpleNamespace(WindowsCapture=WindowsCapture))
    cw = importlib.import_module("fctac.capture.windows")
    src = cw.WGCSource("FC 27").start()
    f = src.buffer.get(-1, timeout=1.0)
    assert f is not None and f.image.shape == (1440, 2560, 3) and src.buffer.dropped == 2
    assert src._cap.kw["window_name"] == "EA SPORTS FC 27"
    assert src.region == (8, 31, 2560, 1440)          # client area inside the captured window
    src.stop()


def test_dxgi_source_and_auto_selection(fake_windows, monkeypatch):
    class Cam:
        width, height = 64, 36

        def grab(self, region=None):
            return np.zeros((36, 64, 3), np.uint8)

        def release(self):
            pass
    monkeypatch.setitem(sys.modules, "dxcam", types.SimpleNamespace(create=lambda **kw: Cam()))
    monkeypatch.setitem(sys.modules, "windows_capture", None)        # WGC unavailable -> fall back
    cw = importlib.import_module("fctac.capture.windows")
    from fctac.config import CaptureConfig
    src = cw.open_source(CaptureConfig())
    assert src.name == "dxgi"
    src.start()
    assert src.buffer.get(-1, timeout=1.0) is not None
    src.stop()


def test_hotkeys_poll(fake_windows):
    import fctac.live as live
    hk = live.Hotkeys()
    assert hk.ok and not hk.pressed("F8")


def test_game_window_title_ranking(fake_windows):
    win32 = importlib.import_module("fctac.overlay.win32")
    subs = ("FC 27", "FC27")
    assert win32._title_rank("EA SPORTS FC 27", subs) == 0
    assert win32._title_rank("EA SPORTS FC 27 Ultimate Team", subs) == 1
    for t in ("FC 27 tips - YouTube - Google Chrome", "FC27 tactical preview", "Discord | #fc27", "Notepad"):
        assert win32._title_rank(t, subs) == -1
