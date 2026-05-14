"""Subprocess wrapper around GNU Backgammon (gnubg -t -q).

Install (Ubuntu): sudo apt-get install gnubg

Tested against GNU Backgammon 1.07.001.

Protocol (per-query):
  new game / set automatic ... off            (once at startup)
  set board simple <26 ints>                  (24 board points + bar_us + bar_them)
  set turn X                                  (make us the player on roll)
  eval                                        (prints static / 1-ply / 2-ply rows)

Output rows have the form:
        Win     W(g)    W(bg)   L(g)    L(bg)   Equity    Cubeful
 2 ply: 0.525   0.149   0.007   0.125   0.005   +0.076    +0.099

We parse the row matching the requested ply (configurable).

Throughput is ~5-20 evaluations/sec; gnubg is for offline label generation
and the gold-standard bench, not the self-play inner loop.
"""
from __future__ import annotations
import os
import re
import subprocess
import threading
import time
from dataclasses import dataclass
from queue import Empty, Queue
from typing import List, Optional

from engine import Position, POINTS, starting_position


GNUBG_BINARY = os.environ.get("GNUBG_BINARY", "gnubg")


@dataclass
class GnubgEval:
    p_win: float
    p_gammon_win: float       # gnubg's W(g) — nested in P(win), e.g. 0.149 with win 0.525
    p_backgammon_win: float
    p_loss: float
    p_gammon_loss: float
    p_backgammon_loss: float
    equity: float             # cubeless equity in roughly [-3, +3]


def gnubg_installed() -> bool:
    return subprocess.run(["which", GNUBG_BINARY],
                          stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL).returncode == 0


def _position_id(p: Position) -> str:
    """Encode our Position as a 14-char gnubg Position ID.

    GNU Backgammon's Position ID is 80 bits (10 bytes, base64-encoded to 14
    chars). For each player it walks 25 positions in this order: 1-pt count,
    2-pt count, …, 24-pt count, bar count (each in *that player's* home
    direction). For each count we emit that many 1-bits followed by a single
    0-bit separator. Player on roll's 25 positions come first, opponent's
    25 positions second. Bits are packed LSB-first within bytes.

    Our convention: positive p.points[i] = us (on roll), negative = opp.
    Index 0 in our points array = our 1-pt, index 23 = our 24-pt. Opp's 1-pt
    is our 24-pt and vice versa.
    """
    bits: List[int] = []
    # Player on roll = us. gnubg's Position-ID puts the opponent's 25
    # positions FIRST and the player-on-roll's 25 positions SECOND. (The
    # starting position's symmetry hid this until we tested an asymmetric
    # bear-off.)
    us_counts = [max(int(p.points[i]), 0) for i in range(POINTS)] + [p.bar_us]
    them_counts = [max(-int(p.points[i]), 0) for i in range(POINTS - 1, -1, -1)] + [p.bar_them]
    for counts in (them_counts, us_counts):
        for c in counts:
            bits.extend([1] * c)
            bits.append(0)
    # Pad / truncate to 80 bits.
    if len(bits) < 80:
        bits.extend([0] * (80 - len(bits)))
    bits = bits[:80]
    byts = bytearray(10)
    for i, b in enumerate(bits):
        if b:
            byts[i // 8] |= 1 << (i % 8)
    import base64
    return base64.b64encode(bytes(byts)).decode("ascii").rstrip("=")




class GnubgClient:
    """Long-lived gnubg subprocess. Per-query: set_board → set turn X → eval
    → read until the eval row arrives or timeout (then kill+respawn)."""

    def __init__(self, timeout: float = 5.0, ply: int = 2):
        if ply not in (0, 1, 2):
            raise ValueError(f"ply must be 0, 1, or 2; got {ply}")
        self.timeout = timeout
        self.ply = ply
        # Label gnubg uses in the eval row for the requested ply:
        self._label = "static" if ply == 0 else f"{ply} ply"
        label_norm = self._label.replace(" ", r"\s*")
        self._row_re = re.compile(
            rf"^\s*{label_norm}\s*:\s+"
            r"([+\-]?\d+\.\d+)\s+([+\-]?\d+\.\d+)\s+([+\-]?\d+\.\d+)\s+"
            r"([+\-]?\d+\.\d+)\s+([+\-]?\d+\.\d+)\s+"
            r"([+\-]?\d+\.\d+)\s+([+\-]?\d+\.\d+)",
            re.MULTILINE,
        )
        self._proc: Optional[subprocess.Popen] = None
        self._reader: Optional[threading.Thread] = None
        self._out_q: Queue = Queue()
        self._start()

    def _start(self):
        if not gnubg_installed():
            raise RuntimeError(
                f"{GNUBG_BINARY} not in PATH. Install: sudo apt-get install gnubg")
        self._proc = subprocess.Popen(
            [GNUBG_BINARY, "-t", "-q"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=0,
        )
        self._out_q = Queue()
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()
        # Configure once. Disable auto game/roll so `eval` returns the
        # static position eval rather than triggering a move list.
        self._write("set automatic game off")
        self._write("set automatic roll off")
        self._write("set automatic move off")
        self._write("set output cubeful off")  # suppress redundant cube line
        self._write("new game")
        # Drain anything emitted by startup.
        time.sleep(0.2)
        while True:
            try:
                self._out_q.get_nowait()
            except Empty:
                break

    def _read_loop(self):
        assert self._proc and self._proc.stdout
        try:
            for line in iter(self._proc.stdout.readline, b""):
                self._out_q.put(line)
        except Exception:
            pass

    def _write(self, cmd: str):
        if not self._proc or self._proc.stdin is None:
            raise RuntimeError("gnubg subprocess not running")
        self._proc.stdin.write((cmd + "\n").encode())
        try:
            self._proc.stdin.flush()
        except Exception:
            pass

    def _read_eval_row(self, row_re, timeout: float) -> Optional[re.Match]:
        """Accumulate gnubg output lines until one matches `row_re` (the
        ply-specific eval row), and return the match. Returns None on
        timeout — the caller should then kill+respawn."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                line = self._out_q.get(timeout=0.1)
            except Empty:
                continue
            text = line.decode("utf8", errors="replace")
            m = row_re.search(text)
            if m:
                return m
        return None

    def _kill_and_respawn(self):
        try:
            if self._proc:
                self._proc.kill()
        except Exception:
            pass
        self._proc = None
        self._start()

    def evaluate_position(self, p: Position) -> Optional[GnubgEval]:
        """Send `p` to gnubg and return its cubeless evaluation at self.ply.
        Returns None on parse/timeout failure."""
        if self._proc is None:
            self._start()
        # Drain any output left over from the previous query so we don't pick
        # up its eval row instead of the current one's.
        while True:
            try:
                self._out_q.get_nowait()
            except Empty:
                break
        try:
            self._write(f"set board {_position_id(p)}")
            self._write("eval")
        except (BrokenPipeError, OSError):
            self._kill_and_respawn()
            return None
        m = self._read_eval_row(self._row_re, timeout=self.timeout)
        if m is None:
            return None
        win, wg, wbg, lg, lbg, eq, _cubeful = (float(g) for g in m.groups())
        return GnubgEval(
            p_win=win, p_gammon_win=wg, p_backgammon_win=wbg,
            p_loss=1.0 - win, p_gammon_loss=lg, p_backgammon_loss=lbg,
            equity=eq,
        )

    def evaluate_starting(self) -> Optional[GnubgEval]:
        return self.evaluate_position(starting_position())

    def close(self):
        try:
            if self._proc:
                self._write("quit")
                self._proc.wait(timeout=2.0)
        except Exception:
            pass
        self._proc = None


def smoke_test() -> int:
    """Convention check on starting position: equity should be very close to
    +0.05..+0.10 (slight player-on-roll edge)."""
    if not gnubg_installed():
        print(f"gnubg not in PATH. Install: sudo apt-get install gnubg")
        return 1
    print("Starting gnubg, requesting 2-ply eval of starting position...")
    c = GnubgClient(ply=2)
    try:
        r = c.evaluate_starting()
    finally:
        c.close()
    if r is None:
        print("FAIL: gnubg returned no evaluation (parse or timeout).")
        return 1
    print(f"gnubg 2-ply eval of starting position:")
    print(f"  P(win)        = {r.p_win:.4f}")
    print(f"  W(g)          = {r.p_gammon_win:.4f}")
    print(f"  W(bg)         = {r.p_backgammon_win:.4f}")
    print(f"  L(g)          = {r.p_gammon_loss:.4f}")
    print(f"  L(bg)         = {r.p_backgammon_loss:.4f}")
    print(f"  equity        = {r.equity:+.4f}")
    if not (0.50 <= r.p_win <= 0.55):
        print(f"WARN: P(win) {r.p_win:.4f} outside expected 0.50–0.55 range.")
        return 2
    if not (0.03 <= r.equity <= 0.12):
        print(f"WARN: equity {r.equity:.4f} outside expected +0.03..+0.12.")
        return 2
    print("OK: starting position convention check passes.")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(smoke_test())
