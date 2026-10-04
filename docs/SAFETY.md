# Game-integrity constraints (what this software does and does not do)

FCTAC is a **read-only visual assistant / training tool**.

It only:

* reads pixels that are already on screen, using standard Windows capture APIs
  (Windows Graphics Capture of the game window, DXGI Desktop Duplication, or
  GDI) — the same mechanisms OBS, Discord or ShadowPlay use;
* reads the state of three keyboard keys (F8/F9/F10) to toggle its own overlay
  (`GetAsyncKeyState`), and finds the game window's position by title
  (`EnumWindows`/`GetClientRect`) so the overlay can sit on top of it;
* draws an ordinary transparent, click-through, top-most window
  (`WS_EX_LAYERED | WS_EX_TRANSPARENT`).

It never:

* reads or writes FC 27 process memory, injects DLLs/code, hooks the game or
  its graphics API, or modifies game files;
* sends keyboard/mouse/controller input or plays for you — every action is the
  human's;
* hides itself from anti-cheat, disguises its process, uses drivers, or tries
  to bypass EA security. Its overlay window is deliberately *not* excluded from
  screen capture.

Development and validation happen on recordings first (replay mode).

**This does not guarantee that using it is permitted.** EA's terms and its
anti-cheat decide what is allowed, especially in online modes. Check the
current rules before using the live overlay in any online match. Replay mode
on your own recordings is the low-risk way to use it.
