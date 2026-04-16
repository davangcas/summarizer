"""Progreso global y logging compatible con barra de terminal."""

from __future__ import annotations

import threading

from tqdm import tqdm


class GlobalProgress:
    """Barra global thread-safe para procesos largos en terminal."""

    def __init__(self, total: int, *, desc: str = "Proceso global") -> None:
        self._lock = threading.RLock()
        self._bar = tqdm(
            total=max(0, int(total)),
            desc=desc,
            unit="item",
            dynamic_ncols=True,
            leave=True,
            position=0,
        )

    def log(self, message: str) -> None:
        with self._lock:
            tqdm.write(message)

    def advance(self, step: int = 1) -> None:
        if step <= 0:
            return
        with self._lock:
            self._bar.update(step)

    def set_stage(self, stage: str) -> None:
        with self._lock:
            self._bar.set_postfix_str(stage, refresh=True)

    def set_total(self, total: int) -> None:
        with self._lock:
            new_total = max(self._bar.n, int(total))
            self._bar.total = new_total
            self._bar.refresh()

    def add_total(self, extra: int) -> None:
        if extra <= 0:
            return
        with self._lock:
            next_total = int(self._bar.total or 0) + int(extra)
            self._bar.total = max(self._bar.n, next_total)
            self._bar.refresh()

    def close(self) -> None:
        with self._lock:
            self._bar.close()


_global_progress_lock = threading.RLock()
_global_progress: GlobalProgress | None = None


def init_global_progress(total: int, *, desc: str = "Proceso global") -> GlobalProgress:
    global _global_progress
    with _global_progress_lock:
        if _global_progress is not None:
            _global_progress.close()
        _global_progress = GlobalProgress(total, desc=desc)
        return _global_progress


def get_global_progress() -> GlobalProgress | None:
    with _global_progress_lock:
        return _global_progress


def close_global_progress() -> None:
    global _global_progress
    with _global_progress_lock:
        if _global_progress is not None:
            _global_progress.close()
            _global_progress = None


def progress_log(message: str) -> None:
    progress = get_global_progress()
    if progress is None:
        print(message)
        return
    progress.log(message)
