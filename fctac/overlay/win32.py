"""Transparent, click-through, always-on-top overlay window (Windows, ctypes only).

Uses a standard layered window (WS_EX_LAYERED | WS_EX_TRANSPARENT) updated
with UpdateLayeredWindow from a premultiplied BGRA buffer.  It never touches
the game process: it is an ordinary top-level window drawn by DWM above the
game.  FC 27 must run in *borderless/windowed* mode for any overlay to be
visible (exclusive fullscreen bypasses desktop composition).

The overlay is deliberately NOT excluded from screen capture: capture the
game window with Windows Graphics Capture (window capture never contains
other windows), or mask the overlay's own pixels when using desktop
duplication (see fctac.capture.windows).
"""
from __future__ import annotations

import ctypes
import sys
from ctypes import wintypes

import numpy as np

if sys.platform != "win32":     # pragma: no cover - import guard for other OSes
    raise ImportError("fctac.overlay.win32 requires Windows")

user32 = ctypes.WinDLL("user32", use_last_error=True)
gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

WS_POPUP = 0x80000000
WS_EX_LAYERED = 0x00080000
WS_EX_TRANSPARENT = 0x00000020
WS_EX_TOPMOST = 0x00000008
WS_EX_TOOLWINDOW = 0x00000080
WS_EX_NOACTIVATE = 0x08000000
SW_SHOWNOACTIVATE = 4
ULW_ALPHA = 0x02
AC_SRC_OVER = 0x00
AC_SRC_ALPHA = 0x01
BI_RGB = 0
DIB_RGB_COLORS = 0
PM_REMOVE = 0x0001
HWND_TOPMOST = wintypes.HWND(-1)
SWP_NOMOVE, SWP_NOSIZE, SWP_NOACTIVATE = 0x0002, 0x0001, 0x0010

LRESULT = ctypes.c_ssize_t
WNDPROC = ctypes.WINFUNCTYPE(LRESULT, wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM)


class WNDCLASSEXW(ctypes.Structure):
    _fields_ = [("cbSize", wintypes.UINT), ("style", wintypes.UINT), ("lpfnWndProc", WNDPROC),
                ("cbClsExtra", ctypes.c_int), ("cbWndExtra", ctypes.c_int), ("hInstance", wintypes.HINSTANCE),
                ("hIcon", wintypes.HICON), ("hCursor", wintypes.HANDLE), ("hbrBackground", wintypes.HBRUSH),
                ("lpszMenuName", wintypes.LPCWSTR), ("lpszClassName", wintypes.LPCWSTR), ("hIconSm", wintypes.HICON)]


class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [("biSize", wintypes.DWORD), ("biWidth", wintypes.LONG), ("biHeight", wintypes.LONG),
                ("biPlanes", wintypes.WORD), ("biBitCount", wintypes.WORD), ("biCompression", wintypes.DWORD),
                ("biSizeImage", wintypes.DWORD), ("biXPelsPerMeter", wintypes.LONG),
                ("biYPelsPerMeter", wintypes.LONG), ("biClrUsed", wintypes.DWORD), ("biClrImportant", wintypes.DWORD)]


class BITMAPINFO(ctypes.Structure):
    _fields_ = [("bmiHeader", BITMAPINFOHEADER), ("bmiColors", wintypes.DWORD * 3)]


class BLENDFUNCTION(ctypes.Structure):
    _fields_ = [("BlendOp", ctypes.c_byte), ("BlendFlags", ctypes.c_byte),
                ("SourceConstantAlpha", ctypes.c_ubyte), ("AlphaFormat", ctypes.c_byte)]


class MSG(ctypes.Structure):
    _fields_ = [("hwnd", wintypes.HWND), ("message", wintypes.UINT), ("wParam", wintypes.WPARAM),
                ("lParam", wintypes.LPARAM), ("time", wintypes.DWORD), ("pt", wintypes.POINT)]


user32.DefWindowProcW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
user32.DefWindowProcW.restype = LRESULT
user32.CreateWindowExW.restype = wintypes.HWND
user32.CreateWindowExW.argtypes = [wintypes.DWORD, wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD,
                                   ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                   wintypes.HWND, wintypes.HMENU, wintypes.HINSTANCE, wintypes.LPVOID]
user32.GetDC.restype = wintypes.HDC
user32.GetDC.argtypes = [wintypes.HWND]
user32.ReleaseDC.argtypes = [wintypes.HWND, wintypes.HDC]
user32.UpdateLayeredWindow.argtypes = [wintypes.HWND, wintypes.HDC, ctypes.POINTER(wintypes.POINT),
                                       ctypes.POINTER(wintypes.SIZE), wintypes.HDC, ctypes.POINTER(wintypes.POINT),
                                       wintypes.COLORREF, ctypes.POINTER(BLENDFUNCTION), wintypes.DWORD]
user32.UpdateLayeredWindow.restype = wintypes.BOOL
user32.PeekMessageW.argtypes = [ctypes.POINTER(MSG), wintypes.HWND, wintypes.UINT, wintypes.UINT, wintypes.UINT]
user32.SetWindowPos.argtypes = [wintypes.HWND, wintypes.HWND, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                ctypes.c_int, wintypes.UINT]
gdi32.CreateCompatibleDC.restype = wintypes.HDC
gdi32.CreateCompatibleDC.argtypes = [wintypes.HDC]
gdi32.CreateDIBSection.restype = wintypes.HBITMAP
gdi32.CreateDIBSection.argtypes = [wintypes.HDC, ctypes.POINTER(BITMAPINFO), wintypes.UINT,
                                   ctypes.POINTER(ctypes.c_void_p), wintypes.HANDLE, wintypes.DWORD]
gdi32.SelectObject.restype = wintypes.HGDIOBJ
gdi32.SelectObject.argtypes = [wintypes.HDC, wintypes.HGDIOBJ]
gdi32.DeleteObject.argtypes = [wintypes.HGDIOBJ]
gdi32.DeleteDC.argtypes = [wintypes.HDC]
kernel32.GetModuleHandleW.restype = wintypes.HMODULE


def set_dpi_aware():
    """Make window/capture coordinates physical pixels (call once at start-up)."""
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)   # per-monitor aware
    except Exception:
        try:
            user32.SetProcessDPIAware()
        except Exception:
            pass


@WNDPROC
def _wndproc(hwnd, msg, wparam, lparam):
    return user32.DefWindowProcW(hwnd, msg, wparam, lparam)


class LayeredOverlay:
    """Full-region transparent overlay. ``present(bgra_premultiplied)`` each update."""

    _class_registered = False
    CLASS = "FCTacOverlay"

    def __init__(self, x: int, y: int, width: int, height: int, title: str = "FC Tactical Overlay"):
        self.x, self.y, self.w, self.h = x, y, width, height
        hinst = kernel32.GetModuleHandleW(None)
        if not LayeredOverlay._class_registered:
            wc = WNDCLASSEXW()
            wc.cbSize = ctypes.sizeof(WNDCLASSEXW)
            wc.lpfnWndProc = _wndproc
            wc.hInstance = hinst
            wc.lpszClassName = self.CLASS
            if not user32.RegisterClassExW(ctypes.byref(wc)):
                raise ctypes.WinError(ctypes.get_last_error())
            LayeredOverlay._class_registered = True
        ex = WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOPMOST | WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE
        self.hwnd = user32.CreateWindowExW(ex, self.CLASS, title, WS_POPUP, x, y, width, height,
                                           None, None, hinst, None)
        if not self.hwnd:
            raise ctypes.WinError(ctypes.get_last_error())
        self.screen_dc = user32.GetDC(None)
        self.mem_dc = gdi32.CreateCompatibleDC(self.screen_dc)
        bmi = BITMAPINFO()
        bmi.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
        bmi.bmiHeader.biWidth = width
        bmi.bmiHeader.biHeight = -height            # top-down rows, like numpy
        bmi.bmiHeader.biPlanes = 1
        bmi.bmiHeader.biBitCount = 32
        bmi.bmiHeader.biCompression = BI_RGB
        bits = ctypes.c_void_p()
        self.dib = gdi32.CreateDIBSection(self.mem_dc, ctypes.byref(bmi), DIB_RGB_COLORS, ctypes.byref(bits), None, 0)
        if not self.dib:
            raise ctypes.WinError(ctypes.get_last_error())
        self.old = gdi32.SelectObject(self.mem_dc, self.dib)
        buf = (ctypes.c_uint8 * (width * height * 4)).from_address(bits.value)
        self.buffer = np.frombuffer(buf, dtype=np.uint8).reshape(height, width, 4)
        self.buffer[:] = 0
        self._blend = BLENDFUNCTION(AC_SRC_OVER, 0, 255, AC_SRC_ALPHA)
        user32.ShowWindow(self.hwnd, SW_SHOWNOACTIVATE)
        self.present(None)

    def present(self, bgra: np.ndarray | None, dirty=None):
        """Copy a premultiplied BGRA image (same size as the overlay) and show it."""
        if bgra is not None:
            if dirty is None:
                np.copyto(self.buffer, bgra)
            else:
                x0, y0, x1, y1 = dirty
                np.copyto(self.buffer[y0:y1, x0:x1], bgra[y0:y1, x0:x1])
        pt_dst = wintypes.POINT(self.x, self.y)
        size = wintypes.SIZE(self.w, self.h)
        pt_src = wintypes.POINT(0, 0)
        ok = user32.UpdateLayeredWindow(self.hwnd, self.screen_dc, ctypes.byref(pt_dst), ctypes.byref(size),
                                        self.mem_dc, ctypes.byref(pt_src), 0, ctypes.byref(self._blend), ULW_ALPHA)
        if not ok:
            raise ctypes.WinError(ctypes.get_last_error())
        self.pump()

    def clear_region(self, dirty):
        if dirty is not None:
            x0, y0, x1, y1 = dirty
            self.buffer[y0:y1, x0:x1] = 0

    def pump(self):
        msg = MSG()
        while user32.PeekMessageW(ctypes.byref(msg), None, 0, 0, PM_REMOVE):
            user32.TranslateMessage(ctypes.byref(msg))
            user32.DispatchMessageW(ctypes.byref(msg))

    def keep_on_top(self):
        user32.SetWindowPos(self.hwnd, HWND_TOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE)

    def close(self):
        if getattr(self, "hwnd", None):
            gdi32.SelectObject(self.mem_dc, self.old)
            gdi32.DeleteObject(self.dib)
            gdi32.DeleteDC(self.mem_dc)
            user32.ReleaseDC(None, self.screen_dc)
            user32.DestroyWindow(self.hwnd)
            self.hwnd = None


# --- window lookup (standard user32 queries; no access to the game process) ---
EnumWindowsProc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)


def find_window(title_substrings=("FC 27", "FC27", "FC 26", "FC 25")) -> int | None:
    found = []

    def cb(hwnd, _):
        if not user32.IsWindowVisible(hwnd):
            return True
        n = user32.GetWindowTextLengthW(hwnd)
        if n == 0:
            return True
        buf = ctypes.create_unicode_buffer(n + 1)
        user32.GetWindowTextW(hwnd, buf, n + 1)
        if any(s.lower() in buf.value.lower() for s in title_substrings):
            found.append(hwnd)
            return False
        return True

    user32.EnumWindows(EnumWindowsProc(cb), 0)
    return found[0] if found else None


def frame_bounds(hwnd) -> tuple[int, int, int, int]:
    """Visible window bounds (x, y, w, h) as composed by DWM -- what Windows
    Graphics Capture returns for a window (includes title bar in windowed mode)."""
    r = wintypes.RECT()
    try:
        dwm = ctypes.WinDLL("dwmapi")
        DWMWA_EXTENDED_FRAME_BOUNDS = 9
        if dwm.DwmGetWindowAttribute(wintypes.HWND(hwnd), DWMWA_EXTENDED_FRAME_BOUNDS,
                                     ctypes.byref(r), ctypes.sizeof(r)) == 0:
            return r.left, r.top, r.right - r.left, r.bottom - r.top
    except Exception:
        pass
    user32.GetWindowRect(hwnd, ctypes.byref(r))
    return r.left, r.top, r.right - r.left, r.bottom - r.top


def client_rect_on_screen(hwnd) -> tuple[int, int, int, int]:
    """(x, y, w, h) of a window's client area in screen pixels."""
    r = wintypes.RECT()
    user32.GetClientRect(hwnd, ctypes.byref(r))
    p = wintypes.POINT(0, 0)
    user32.ClientToScreen(hwnd, ctypes.byref(p))
    return p.x, p.y, r.right - r.left, r.bottom - r.top
