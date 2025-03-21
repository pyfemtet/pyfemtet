from time import time, sleep
from packaging.version import Version

from win32com.client import CDispatch
from femtetutils import util
import psutil

from pyfemtet._util.process_util import _get_pid
from pyfemtet._util.femtet_version import _version

from pyfemtet.logger import get_module_logger


__all__ = ['_exit_or_force_terminate']


logger = get_module_logger('util.femtet.exit', False)


def _exit_or_force_terminate(timeout, Femtet: CDispatch, force=True):

    # Femtet がすでに終了しているかどうかを判断する
    hwnd = Femtet.hWnd
    pid = _get_pid(hwnd)

    # pid が 0 ならすでに終了しているものと見做して何もしない
    if pid == 0:
        return

    # Exit() メソッドを定義している最小の Femtet バージョン
    minimum_version = Version('2024.0.1')

    # 現在の Femtet のバージョン
    current_version = _version(Femtet=Femtet)

    # 現在のバージョン >= minimum version なら Exit を実行
    if current_version >= minimum_version:
        # gracefully termination method without save project available from 2024.0.1
        try:
            Femtet.Exit(force)

        except AttributeError:
            raise AttributeError('Macro version is not consistent to the one of Femtet.exe. '
                                 'Please consider ``Enable Macros`` to fix it.')

    # そうでなければ強制終了する
    else:
        # terminate
        util.close_femtet(Femtet.hWnd, timeout, force)

        try:
            pid = _get_pid(hwnd)
            start = time()
            while psutil.pid_exists(pid):
                if time() - start > 30:  # 30 秒経っても存在するのは何かおかしい
                    # logger.error(Msg.ERR_CLOSE_FEMTET_FAILED)
                    break
                sleep(1)
            sleep(1)

        # dead
        except (AttributeError, OSError):
            pass
