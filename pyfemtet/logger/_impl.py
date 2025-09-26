import logging
import os
import sys
import datetime
import locale
from threading import Lock
import platform

from colorlog import ColoredFormatter
from dask.distributed import get_worker

LOCALE, LOCALE_ENCODING = locale.getlocale()
if platform.system() == 'Windows':
    DATEFMT = '%#m/%#d %#H:%M'
else:
    DATEFMT = '%-m/%-d %-H:%M'

__lock = Lock()  # thread 並列されたタスクがアクセスする場合に備えて

__initialized_root_packages: list[str] = list()


# ===== set dask worker prefix to ``ROOT`` logger =====

def _get_dask_worker_name():
    name = '(Main)'
    try:
        worker = get_worker()
        if isinstance(worker.name, str):  # local なら index, cluster なら tcp address
            name = f'({worker.name})'
        else:
            name = f'(Sub{worker.name})'
    except ValueError:
        pass
    return name


class _DaskLogRecord(logging.LogRecord):
    def getMessage(self):
        msg = str(self.msg)
        if self.args:
            msg = msg % self.args
        msg = _get_dask_worker_name() + ' ' + msg
        return msg


logging.setLogRecordFactory(_DaskLogRecord)


# ===== format config =====

def __create_formatter(colored=True):

    if colored:
        # colorized
        header = "%(log_color)s" + "[%(name)s %(levelname).4s]" + " %(asctime)s" + "%(reset)s"

        formatter = ColoredFormatter(
            f"{header} %(message)s",
            datefmt=DATEFMT,
            reset=True,
            log_colors={
                "DEBUG": "purple",
                "INFO": "cyan",
                "WARNING": "yellow",
                "ERROR": "light_red",
                "CRITICAL": "red",
            },
        )

    else:
        header = "[%(name)s %(levelname).4s]"
        formatter = logging.Formatter(
            f"{header} %(message)s",
            datefmt=DATEFMT,
        )

    return formatter


# ===== handler config =====

STDOUT_HANDLER_NAME = 'stdout-handler'
STDERR_HANDLER_NAME = 'stderr-handler'


def __get_stdout_handler():
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.set_name(STDOUT_HANDLER_NAME)
    stdout_handler.setFormatter(__create_formatter(colored=True))
    return stdout_handler


def __has_stdout_handler(logger):
    return any([handler.get_name() != STDOUT_HANDLER_NAME for handler in logger.handlers])


def __get_stderr_handler():
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.set_name(STDERR_HANDLER_NAME)
    stderr_handler.setFormatter(__create_formatter(colored=True))
    return stderr_handler


def __has_stderr_handler(logger):
    return any([handler.get_name() != STDERR_HANDLER_NAME for handler in logger.handlers])


def set_stdout_output(logger, level=logging.INFO):

    if not __has_stdout_handler(logger):
        logger.addHandler(__get_stdout_handler())

    logger.setLevel(level)


def remove_stdout_output(logger):
    if __has_stdout_handler(logger):
        logger.removeHandler(__get_stdout_handler())


def set_stderr_output(logger, level=logging.INFO):

    if not __has_stderr_handler(logger):
        logger.addHandler(__get_stderr_handler())

    logger.setLevel(level)


def remove_stderr_output(logger):
    if __has_stderr_handler(logger):
        logger.removeHandler(__get_stderr_handler())


def add_file_output(logger, filepath=None, level=logging.INFO) -> str:
    """Add FileHandler to the logger.

    Returns:
        str: THe name of the added handler.
        Its format is 'filehandler-{os.path.basename(filepath)}'

    """

    # certify filepath
    if filepath is None:
        filepath = datetime.datetime.now().strftime('%Y%m%d-%H%M%S') + f'_{logger.name}.log'

    # add file handler
    file_handler = logging.FileHandler(filename=filepath, encoding=LOCALE_ENCODING)
    file_handler.set_name(f'filehandler-{os.path.basename(filepath)}')
    file_handler.setFormatter(__create_formatter(colored=False))
    logger.addHandler(file_handler)

    # set (default) log level
    logger.setLevel(level)

    return file_handler.get_name()


def remove_file_output(logger, filepath=None):
    """Removes FileHandler from the logger.

    If filepath is None, remove all FileHandler.
    """

    if filepath is None:
        for handler in logger.handlers:
            if 'filehandler-' in handler.name:
                logger.removeHandler(handler)

    else:
        handler_name = f'filehandler-{os.path.basename(filepath)}'
        for handler in logger.handlers:
            if handler_name == handler.name:
                logger.removeHandler(handler)


def remove_all_output(logger):
    for handler in logger.handlers:
        logger.removeHandler(handler)

    logger.addHandler(logging.NullHandler())


# ===== root-package logger =====

def setup_package_root_logger(package_name):
    global __initialized_root_packages
    if package_name not in __initialized_root_packages:
        with __lock:
            logger = logging.getLogger(package_name)
            logger.propagate = True
            set_stderr_output(logger)
            logger.setLevel(logging.INFO)
            __initialized_root_packages.append(package_name)
    else:
        logger = logging.getLogger(package_name)
    return logger


# ===== module logger =====

def get_module_logger(name: str, debug=False) -> logging.Logger:
    """Return the module-level logger.

    The format is defined in the package_root_logger.

    Args:
        name (str): The logger name to want.
        debug (bool, optional): Output DEBUG level message or not.

    Returns:
        logging.Logger:
            The logger its name is ``root_package.subpackage.module``.
            child level logger's signal propagates to the parent logger
            and is shown in the parent(s)'s handler(s).

    """

    # check root logger initialized
    name_arr = name.split('.')
    if name_arr[0] not in __initialized_root_packages:
        setup_package_root_logger(name_arr[0])

    # get logger
    logger = logging.getLogger(name)

    # If not root logger, ensure propagate is True.
    if len(name_arr) > 1:
        logger.propagate = True

    # If debug mode, set specific level.
    if debug:
        logger.setLevel(logging.DEBUG)

    return logger


if __name__ == '__main__':

    root_logger = setup_package_root_logger('logger')
    optimizer_logger = get_module_logger('logger.optimizer', False); optimizer_logger.setLevel(logging.INFO)
    interface_logger = get_module_logger('logger.interface', False)

    root_logger.info("This is root logger's info.")
    optimizer_logger.info("This is optimizer logger's info.")

    add_file_output(interface_logger, 'test-module-log.log', level=logging.DEBUG)
    interface_logger.debug('debugging...')
    remove_file_output(interface_logger, 'test-module-log.log')

    interface_logger.debug('debug is finished.')
    root_logger.debug("This message will not be shown "
                      "even if the module_logger's level "
                      "is logging.DEBUG.")
