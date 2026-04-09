"""Parada cooperativa (Ctrl+C, teclas, consola)."""

from __future__ import annotations

import os
import signal
import sys
import threading
import time
from typing import Any

_stop_event = threading.Event()


class StopRequested(Exception):
    """Parada cooperativa solicitada por usuario."""


def request_stop(reason: str) -> None:
    if _stop_event.is_set():
        return
    _stop_event.set()
    print(f"\n[STOP] {reason}")


def check_stop_requested() -> None:
    if _stop_event.is_set():
        raise StopRequested("Proceso detenido por solicitud del usuario.")


def sleep_with_stop(total_seconds: float) -> None:
    deadline = time.monotonic() + max(0.0, total_seconds)
    while True:
        check_stop_requested()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(0.25, remaining))


def install_stop_handlers() -> None:
    def _on_sigint(_signum: int, _frame: Any) -> None:
        request_stop("SIGINT recibido (Ctrl+C).")

    try:
        signal.signal(signal.SIGINT, _on_sigint)
    except Exception:
        pass


def _stop_listener_tty_unix() -> None:
    import select
    import termios
    import tty

    fd = sys.stdin.fileno()
    try:
        old = termios.tcgetattr(fd)
    except (OSError, AttributeError, termios.error):
        _stop_listener_line()
        return

    def _char_stops(ch: str) -> bool:
        return len(ch) == 1 and ch.lower() in ("x", "q", "s")

    try:
        tty.setcbreak(fd)
        while not _stop_event.is_set():
            readable, _, _ = select.select([sys.stdin], [], [], 0.2)
            if not readable:
                continue
            ch = sys.stdin.read(1)
            if not ch:
                return
            if _char_stops(ch):
                request_stop("Tecla rápida (x, q o s).")
                return
    except (EOFError, OSError, ValueError):
        return
    finally:
        try:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
        except (OSError, termios.error):
            pass


def _stop_listener_tty_windows() -> None:
    import msvcrt

    while not _stop_event.is_set():
        if msvcrt.kbhit():
            ch = msvcrt.getch()
            if ch.lower() in (b"x", b"q", b"s") or ch == b"\x03":
                request_stop("Tecla rápida (x, q o s) o Ctrl+C (consola).")
                return
        time.sleep(0.05)


def _stop_listener_line() -> None:
    """Sin TTY interactivo: una línea + Enter (p. ej. redirección de stdin)."""
    print("Control (modo línea): escriba 'stop' (o Enter) y pulse Enter para detener.")
    while not _stop_event.is_set():
        try:
            cmd = input().strip().lower()
        except EOFError:
            return
        except Exception:
            return
        if cmd in ("", "stop", "salir", "exit", "quit"):
            request_stop("Parada solicitada desde consola.")
            return
        print("Comando no reconocido. Use 'stop' (o Enter) para detener.")


def start_stop_listener() -> None:
    print(
        "Control activo: pulse x, q o s (sin Enter) para detener; "
        "o Ctrl+C. Sin consola interactiva, use 'stop' + Enter."
    )

    def _runner() -> None:
        if sys.stdin.isatty():
            if os.name == "nt" or sys.platform == "win32":
                _stop_listener_tty_windows()
            else:
                _stop_listener_tty_unix()
        else:
            _stop_listener_line()

    threading.Thread(target=_runner, daemon=True, name="stop-listener").start()
