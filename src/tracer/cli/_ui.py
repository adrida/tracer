"""Terminal UI utilities -- colors, spinner, formatted output.

Zero external dependencies. Falls back to plain text when not a TTY.
"""

from __future__ import annotations

import itertools
import sys
import threading
import time
from contextlib import contextmanager
from typing import Optional


# ── ANSI ──────────────────────────────────────────────────────────────────────

def _is_tty() -> bool:
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


class _Ansi:
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    DIM    = "\033[2m"

    BLACK   = "\033[30m"
    RED     = "\033[31m"
    GREEN   = "\033[32m"
    YELLOW  = "\033[33m"
    BLUE    = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN    = "\033[36m"
    WHITE   = "\033[37m"

    BG_BLACK = "\033[40m"

    def __getattr__(self, name):
        return ""  # fallback: return empty string when color disabled


_NO_COLOR = _Ansi()
_NO_COLOR.RESET = _NO_COLOR.BOLD = _NO_COLOR.DIM = ""
for _k in ("BLACK","RED","GREEN","YELLOW","BLUE","MAGENTA","CYAN","WHITE","BG_BLACK"):
    setattr(_NO_COLOR, _k, "")

def _C() -> _Ansi:
    """Return color codes if TTY, empty strings otherwise."""
    return _Ansi() if _is_tty() else _NO_COLOR


# ── Helpers ───────────────────────────────────────────────────────────────────

def hr(char="─", width=56) -> str:
    c = _C()
    return f"{c.DIM}{char * width}{c.RESET}"


def header(title: str, subtitle: str = "") -> None:
    c = _C()
    print()
    print(f"  {c.BOLD}{c.CYAN}▸ TRACER{c.RESET}  {c.WHITE}{title}{c.RESET}")
    if subtitle:
        print(f"  {c.DIM}{subtitle}{c.RESET}")
    print(f"  {hr()}")


def section(title: str) -> None:
    c = _C()
    print()
    print(f"  {c.BOLD}{c.YELLOW}{title}{c.RESET}")
    print(f"  {hr('·')}")


def stat(label: str, value: str, note: str = "", color: str = "") -> None:
    c = _C()
    val_color = getattr(c, color.upper(), "") if color else c.WHITE
    pad = 18
    print(f"  {c.DIM}{label:<{pad}}{c.RESET}{c.BOLD}{val_color}{value}{c.RESET}"
          + (f"  {c.DIM}{note}{c.RESET}" if note else ""))


def success(msg: str) -> None:
    c = _C()
    print(f"  {c.GREEN}✔{c.RESET}  {msg}")


def warn(msg: str) -> None:
    c = _C()
    print(f"  {c.YELLOW}⚠{c.RESET}  {msg}")


def info(msg: str) -> None:
    c = _C()
    print(f"  {c.DIM}→{c.RESET}  {msg}")


def bar_line(label: str, rate: float, count: int, width: int = 22) -> None:
    c = _C()
    filled = int(rate * width)
    empty  = width - filled
    if rate >= 0.85:
        fill_c = c.GREEN
    elif rate >= 0.60:
        fill_c = c.YELLOW
    else:
        fill_c = c.RED
    bar = f"{fill_c}{'█' * filled}{c.DIM}{'░' * empty}{c.RESET}"
    pct = f"{rate:>5.1%}"
    print(f"  {c.DIM}{label:<28}{c.RESET}{bar}  {c.BOLD}{pct}{c.RESET}  {c.DIM}n={count}{c.RESET}")


def pair_block(teacher_label: str, handled: str, deferred: str,
               h_score: Optional[float] = None, d_score: Optional[float] = None) -> None:
    c = _C()
    hs = f" {c.DIM}score={h_score:.2f}{c.RESET}" if h_score is not None else ""
    ds = f" {c.DIM}score={d_score:.2f}{c.RESET}" if d_score is not None else ""
    print(f"  {c.DIM}label:{c.RESET} {c.BOLD}{teacher_label}{c.RESET}")
    print(f"    {c.GREEN}SURROGATE{c.RESET} {handled[:72]}{hs}")
    print(f"    {c.RED}→ LLM  {c.RESET} {deferred[:72]}{ds}")
    print()


def route_line(text: str, decision: str, label: str, score: Optional[float]) -> None:
    c = _C()
    if decision == "handled":
        tag   = f"{c.BG_BLACK}{c.GREEN} SURR.  {c.RESET}"
        label_str = f"  {c.DIM}→{c.RESET} {c.CYAN}{label}{c.RESET}"
    else:
        tag   = f"{c.BG_BLACK}{c.YELLOW} → LLM  {c.RESET}"
        label_str = ""
    score_str = f" {c.DIM}({score:.2f}){c.RESET}" if score is not None else ""
    print(f"  {tag}{score_str}  {text[:60]}{label_str}")


def cost_table(coverage: float, daily_queries: int = 10_000, cost_per_call: float = 0.002) -> None:
    c = _C()
    llm_full    = daily_queries
    llm_tracer  = int(daily_queries * (1 - coverage))
    day_full    = llm_full   * cost_per_call
    day_tracer  = llm_tracer * cost_per_call
    annual_save = (day_full - day_tracer) * 365

    print(f"  {c.DIM}{'':4}{'LLM calls/day':>15}  {'cost/day':>10}  {'annual':>10}{c.RESET}")
    print(f"  {c.DIM}Without TRACER{c.RESET}  {llm_full:>13,}  ${day_full:>9,.2f}  -")
    print(f"  {c.GREEN}{c.BOLD}With TRACER   {c.RESET}  {llm_tracer:>13,}  ${day_tracer:>9,.2f}  {c.BOLD}{c.GREEN}${annual_save:>9,.0f} saved/yr{c.RESET}")
    reduction = coverage
    print(f"  {c.DIM}({reduction:.0%} fewer LLM calls){c.RESET}")


# ── Spinner ───────────────────────────────────────────────────────────────────

_SPIN_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
_SPIN_FRAMES_PLAIN = ["|", "/", "-", "\\"]


class Spinner:
    """Context manager that displays an animated spinner while work runs."""

    def __init__(self, msg: str, done_msg: str = "done"):
        self.msg = msg
        self.done_msg = done_msg
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _spin(self) -> None:
        if not _is_tty():
            sys.stdout.write(f"  {self.msg}...\n")
            sys.stdout.flush()
            return
        frames = _SPIN_FRAMES if _is_tty() else _SPIN_FRAMES_PLAIN
        c = _C()
        for frame in itertools.cycle(frames):
            if self._stop.is_set():
                break
            sys.stdout.write(f"\r  {c.CYAN}{frame}{c.RESET}  {self.msg}...")
            sys.stdout.flush()
            time.sleep(0.08)

    def __enter__(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=0.5)
        c = _C()
        if _is_tty():
            sys.stdout.write(f"\r  {c.GREEN}✔{c.RESET}  {self.msg}  {c.DIM}{self.done_msg}{c.RESET}\n")
        sys.stdout.flush()


@contextmanager
def step(msg: str, done_msg: str = "done"):
    with Spinner(msg, done_msg):
        yield
