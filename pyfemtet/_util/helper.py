from __future__ import annotations

from typing import Callable

import string
import secrets
import warnings
from time import time, sleep
from threading import Thread

from pyfemtet._i18n import _

__all__ = [
    'float_',
    'time_counting',
    'generate_random_id',
]


def float_(value: str | None | float) -> str | float | None:
    if value is None:
        return value
    try:
        return float(value)
    except ValueError:
        return value


class _TimeCounting:

    _thread: Thread
    _should_exit: bool
    name: str
    warning_time_sec: float
    warning_message: str | None
    warning_fun: Callable[[], ...] | None

    def __init__(
            self,
            name: str,
            warning_time_sec: float = None,
            warning_message: str = None,
            warning_fun: Callable[[], ...] = None,
    ):
        self.name = name
        self.warning_time_sec = warning_time_sec
        self.warning_message = warning_message
        self.warning_fun = warning_fun
        if self.warning_time_sec is not None:
            if warning_message is None and warning_fun is None:
                self.warning_message = _(
                    '{name} does not finished in {timeout} seconds.',
                    '{name} が {timeout} 秒以内に終了しませんでした。',
                    name=self.name,
                    timeout=self.warning_time_sec
                )

        self._thread = Thread(target=self._time_count, daemon=True)
        self._should_exit = False

    def _time_count(self):
        start = time()

        while True:

            if self._should_exit:
                break

            elif time() - start > self.warning_time_sec:
                if self.warning_fun is None:
                    warnings.warn(self.warning_message)
                else:
                    self.warning_fun()
                break

            sleep(0.5)

    def __enter__(self):
        self._thread.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._should_exit = True
        self._thread.join()


def time_counting(
        name: str,
        warning_time_sec: float = None,
        warning_message: str = None,
        warning_fun: Callable[[], ...] = None,
):

    return _TimeCounting(
        name,
        warning_time_sec,
        warning_message,
        warning_fun,
    )


def generate_random_id(length: int = 16) -> str:
    alphabet = string.ascii_letters + string.digits
    return ''.join(secrets.choice(alphabet) for _ in range(length))
