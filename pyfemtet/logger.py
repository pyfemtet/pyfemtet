import os
import sys
import logging
from colorlog import ColoredFormatter
from dask.distributed import get_worker


__all__ = ['get_logger']


def _get_worker_name_as_prefix():
    name = '(Main) '
    try:
        worker = get_worker()
        # address = worker.address
        name = f'(Worker:{worker.name}) '
    except ValueError:
        pass
    return name


class DaskLogRecord(logging.LogRecord):
    """Generate a log message with dask worker name."""
    # Copyright (C) 2001-2022 Vinay Sajip. All Rights Reserved.
    def getMessage(self):
        msg = str(self.msg)
        if self.args:
            msg = msg % self.args
        msg = _get_worker_name_as_prefix() + msg
        return msg


logging.setLogRecordFactory(DaskLogRecord)  # すべての logging %(message)s の前に prefix を入れる


def _color_supported() -> bool:
    """Detection of color support.

    This function is originated from optuna.logging.

    """

    # MIT License
    #
    # Copyright (c) 2018 Preferred Networks, Inc.
    #
    # Permission is hereby granted, free of charge, to any person obtaining a copy
    # of this software and associated documentation files (the "Software"), to deal
    # in the Software without restriction, including without limitation the rights
    # to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    # copies of the Software, and to permit persons to whom the Software is
    # furnished to do so, subject to the following conditions:
    #
    # The above copyright notice and this permission notice shall be included in all
    # copies or substantial portions of the Software.
    #
    # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    # IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    # FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    # AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    # LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    # OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    # SOFTWARE.


    # NO_COLOR environment variable:
    if os.environ.get("NO_COLOR", None):
        return False

    if not hasattr(sys.stderr, "isatty") or not sys.stderr.isatty():
        return False
    else:
        return True


def _create_formatter() -> logging.Formatter:
    """Create a formatter."""
    header = f"[pyfemtet %(name)s] %(levelname).4s"
    message = "%(message)s"

    if _color_supported():

        formatter = ColoredFormatter(
            f"%(log_color)s{header}%(reset)s {message}",
            datefmt=None,
            reset=True,
            log_colors={
                "DEBUG": "purple",
                "INFO": "cyan",
                "WARNING": "yellow",
                "ERROR": "light_red",
                "CRITICAL": "red",
            },
        )

        return formatter

    return logging.Formatter(f"{header} {message}")


def get_logger(logger_name):
    """Return a logger with a default ColoredFormatter."""

    formatter = _create_formatter()

    logger = logging.getLogger(logger_name)
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)

    return logger
