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
        if isinstance(worker.name, str):  # local なら index, cluster なら tcp address
            name = f'({worker.name}) '
        else:
            name = f'(Sub{worker.name}) '
    except ValueError:
        pass
    return name


class DaskLogRecord(logging.LogRecord):
    """Generate a log message with dask worker name."""

    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)
    #     self.worker = _get_worker_name_as_prefix()

    def getMessage(self):
        """Add worker name to loggin message.

        This function is originated from logging.LogRecord.

            # Copyright (C) 2001-2022 Vinay Sajip. All Rights Reserved.

        """
        msg = str(self.msg)
        if self.args:
            msg = msg % self.args
        msg = _get_worker_name_as_prefix() + msg
        return msg


logging.setLogRecordFactory(DaskLogRecord)  # すべての logging %(message)s の前に prefix を入れる


def _color_supported() -> bool:
    """Detection of color support.

    This function is originated from optuna.logging.

        # Copyright (c) 2018 Preferred Networks, Inc.

    """

    # NO_COLOR environment variable:
    if os.environ.get("NO_COLOR", None):
        return False

    if not hasattr(sys.stderr, "isatty") or not sys.stderr.isatty():
        return False
    else:
        return True


def _create_formatter() -> logging.Formatter:
    """Create a formatter."""
    # header = f"[pyfemtet %(name)s] %(levelname).4s %(worker)s]"
    header = f"[pyfemtet %(name)s %(levelname).4s]"
    message = "%(message)s"

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


def get_logger(logger_name):
    """Return a logger with a default ColoredFormatter."""

    formatter = _create_formatter()

    logger = logging.getLogger(logger_name)
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)

    return logger
