"""A small terminal-screen emulator for reading what lobster-tui draws on a pty.

``pyte`` is not a dependency of this repository (and ``pyproject.toml`` is not
ours to change), so this is the minimum VT/xterm subset BubbleTea's renderer
uses: printable text with wide characters, CR/LF/BS/TAB, cursor movement
(CUU/CUD/CUF/CUB/CNL/CPL/CHA/CUP/VPA), erase (ED/EL/ECH), insert/delete
(ICH/DCH/IL/DL), scroll (SU/SD, scroll region, reverse index), save/restore
cursor and the cursor-visibility mode. Colours, OSC/DCS strings and mode
queries are parsed and ignored. Unknown CSI finals are counted in
``unknown`` so a test can tell when the emulator falls short.
"""

from __future__ import annotations

import re
import threading
import unicodedata
from typing import Dict, List

_ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)|\x1b.")


def strip_ansi(text: str) -> str:
    """Text without escape sequences (for messages, not for the screen)."""
    return _ANSI.sub("", text)


def _width(ch: str) -> int:
    if unicodedata.combining(ch):
        return 0
    if unicodedata.east_asian_width(ch) in ("W", "F"):
        return 2
    return 1


class Screen:
    """A rows x cols character grid fed with the pty's output bytes."""

    def __init__(self, rows: int, cols: int) -> None:
        self.rows, self.cols = rows, cols
        self.grid: List[List[str]] = [[" "] * cols for _ in range(rows)]
        self.y = self.x = 0
        self.top, self.bottom = 0, rows - 1
        self.saved = (0, 0)
        self.wrap_pending = False
        self.cursor_visible = True
        self.unknown: Dict[str, int] = {}
        self.scrolled_off: List[str] = []  # lines pushed above the screen
        self._state = "text"
        self._buf = ""
        self._decoder_tail = b""
        self.lock = threading.Lock()

    # -- public ------------------------------------------------------------

    def feed(self, data: bytes) -> None:
        data = self._decoder_tail + data
        # Keep an incomplete UTF-8 sequence for the next chunk.
        cut = len(data)
        for back in range(1, min(4, len(data)) + 1):
            byte = data[-back]
            if byte & 0xC0 == 0x80:
                continue
            need = 1
            if byte & 0xE0 == 0xC0:
                need = 2
            elif byte & 0xF0 == 0xE0:
                need = 3
            elif byte & 0xF8 == 0xF0:
                need = 4
            if need > back:
                cut = len(data) - back
            break
        self._decoder_tail = data[cut:]
        text = data[:cut].decode("utf-8", errors="replace")
        with self.lock:
            for ch in text:
                self._char(ch)

    def resize(self, rows: int, cols: int) -> None:
        with self.lock:
            grid = [[" "] * cols for _ in range(rows)]
            # Keep the bottom rows (the inline renderer draws at the bottom).
            offset = max(self.rows - rows, 0)
            for r in range(min(rows, self.rows)):
                src = self.grid[r + offset]
                for c in range(min(cols, self.cols)):
                    grid[r][c] = src[c]
            self.grid, self.rows, self.cols = grid, rows, cols
            self.top, self.bottom = 0, rows - 1
            self.y = min(max(self.y - offset, 0), rows - 1)
            self.x = min(self.x, cols - 1)
            self.wrap_pending = False

    def lines(self) -> List[str]:
        with self.lock:
            return [
                "".join(c for c in row if c is not None).rstrip() for row in self.grid
            ]

    def text(self) -> str:
        return "\n".join(self.lines())

    # -- parser ------------------------------------------------------------

    def _char(self, ch: str) -> None:
        state = self._state
        if state == "text":
            if ch == "\x1b":
                self._state, self._buf = "esc", ""
            elif ch == "\r":
                self.x, self.wrap_pending = 0, False
            elif ch == "\n" or ch == "\x0b" or ch == "\x0c":
                self._linefeed()
            elif ch == "\b":
                self.x, self.wrap_pending = max(self.x - 1, 0), False
            elif ch == "\t":
                self.x = min((self.x // 8 + 1) * 8, self.cols - 1)
            elif ch == "\x07" or ord(ch) < 0x20 or ch == "\x7f":
                pass
            else:
                self._put(ch)
        elif state == "esc":
            if ch == "[":
                self._state, self._buf = "csi", ""
            elif ch == "]":
                self._state, self._buf = "osc", ""
            elif ch in "PX^_":
                self._state, self._buf = "str", ""
            elif ch in "()*+#%":
                self._state = "esc_arg"
            else:
                self._state = "text"
                if ch == "7":
                    self.saved = (self.y, self.x)
                elif ch == "8":
                    self.y, self.x = self.saved
                elif ch == "M":
                    if self.y == self.top:
                        self._scroll_down(1)
                    else:
                        self.y = max(self.y - 1, 0)
                elif ch == "D":
                    self._linefeed()
                elif ch == "E":
                    self.x = 0
                    self._linefeed()
                elif ch == "c":
                    self.__init__(self.rows, self.cols)
        elif state == "esc_arg":
            self._state = "text"
        elif state == "csi":
            if "@" <= ch <= "~":
                self._state = "text"
                self._csi(self._buf, ch)
            else:
                self._buf += ch
        elif state in ("osc", "str"):
            if ch == "\x07":
                self._state = "text"
            elif ch == "\x1b":
                self._state = state + "_esc"
            else:
                self._buf += ch
        elif state in ("osc_esc", "str_esc"):
            # ESC \ ends the string; anything else is part of it.
            self._state = "text" if ch == "\\" else state[:-4]

    def _put(self, ch: str) -> None:
        w = _width(ch)
        if w == 0:
            return
        if self.wrap_pending:
            self.x, self.wrap_pending = 0, False
            self._linefeed()
        if self.x + w > self.cols:
            self.x = 0
            self._linefeed()
        row = self.grid[self.y]
        row[self.x] = ch
        if w == 2 and self.x + 1 < self.cols:
            row[self.x + 1] = None  # type: ignore[call-overload]
        self.x += w
        if self.x >= self.cols:
            self.x, self.wrap_pending = self.cols - 1, True

    def _linefeed(self) -> None:
        self.wrap_pending = False
        if self.y == self.bottom:
            self._scroll_up(1)
        elif self.y < self.rows - 1:
            self.y += 1

    def _scroll_up(self, n: int) -> None:
        for _ in range(n):
            gone = self.grid.pop(self.top)
            if self.top == 0:
                self.scrolled_off.append("".join(c for c in gone if c).rstrip())
            self.grid.insert(self.bottom, [" "] * self.cols)

    def _scroll_down(self, n: int) -> None:
        for _ in range(n):
            self.grid.pop(self.bottom)
            self.grid.insert(self.top, [" "] * self.cols)

    def _erase(self, row: int, start: int, end: int) -> None:
        line = self.grid[row]
        for c in range(max(start, 0), min(end, self.cols)):
            line[c] = " "

    def _csi(self, raw: str, final: str) -> None:
        private = raw[:1] in ("?", ">", "=", "<")
        body = raw[1:] if private else raw
        if any(c in body for c in "$ \"'"):
            return  # DECRQM and friends: queries, nothing to draw
        params = [int(p) if p.isdigit() else 0 for p in body.split(";")] if body else []

        def p(i: int, default: int = 1) -> int:
            return params[i] if len(params) > i and params[i] else default

        self.wrap_pending = False if final not in "m" else self.wrap_pending
        if private:
            if final in "hl" and raw.startswith("?"):
                if 25 in params:
                    self.cursor_visible = final == "h"
            return  # other private modes and kitty keyboard queries
        if final == "A":
            self.y = max(self.y - p(0), 0)
        elif final == "B":
            self.y = min(self.y + p(0), self.rows - 1)
        elif final == "C":
            self.x = min(self.x + p(0), self.cols - 1)
        elif final == "D":
            self.x = max(self.x - p(0), 0)
        elif final == "E":
            self.y, self.x = min(self.y + p(0), self.rows - 1), 0
        elif final == "F":
            self.y, self.x = max(self.y - p(0), 0), 0
        elif final in "G`":
            self.x = min(p(0) - 1, self.cols - 1)
        elif final in "Hf":
            self.y = min(p(0) - 1, self.rows - 1)
            self.x = min(p(1) - 1, self.cols - 1)
        elif final == "d":
            self.y = min(p(0) - 1, self.rows - 1)
        elif final == "J":
            mode = params[0] if params else 0
            if mode == 0:
                self._erase(self.y, self.x, self.cols)
                for r in range(self.y + 1, self.rows):
                    self._erase(r, 0, self.cols)
            elif mode == 1:
                for r in range(0, self.y):
                    self._erase(r, 0, self.cols)
                self._erase(self.y, 0, self.x + 1)
            else:
                for r in range(self.rows):
                    self._erase(r, 0, self.cols)
        elif final == "K":
            mode = params[0] if params else 0
            if mode == 0:
                self._erase(self.y, self.x, self.cols)
            elif mode == 1:
                self._erase(self.y, 0, self.x + 1)
            else:
                self._erase(self.y, 0, self.cols)
        elif final == "X":
            self._erase(self.y, self.x, self.x + p(0))
        elif final == "P":
            line = self.grid[self.y]
            n = min(p(0), self.cols - self.x)
            del line[self.x : self.x + n]
            line.extend([" "] * n)
        elif final == "@":
            line = self.grid[self.y]
            n = min(p(0), self.cols - self.x)
            for _ in range(n):
                line.insert(self.x, " ")
            del line[self.cols :]
        elif final == "L":
            for _ in range(p(0)):
                self.grid.pop(self.bottom)
                self.grid.insert(self.y, [" "] * self.cols)
        elif final == "M":
            for _ in range(p(0)):
                self.grid.pop(self.y)
                self.grid.insert(self.bottom, [" "] * self.cols)
        elif final == "S":
            self._scroll_up(p(0))
        elif final == "T":
            self._scroll_down(p(0))
        elif final == "r":
            self.top = p(0) - 1
            self.bottom = (
                params[1] if len(params) > 1 and params[1] else self.rows
            ) - 1
            self.y, self.x = 0, 0
        elif final == "s":
            self.saved = (self.y, self.x)
        elif final == "u":
            self.y, self.x = self.saved
        elif final == "b":  # REP: repeat the previous character
            prev = self.grid[self.y][self.x - 1] if self.x > 0 else " "
            for _ in range(p(0)):
                self._put(prev or " ")
        elif final in "mntqc":
            pass  # colours, reports, cursor style, device attributes
        else:
            self.unknown[final] = self.unknown.get(final, 0) + 1
