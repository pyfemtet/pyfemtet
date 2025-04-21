from time import time, sleep
from packaging.version import Version

from win32com.client import CDispatch
from femtetutils import util
import psutil

from pyfemtet._i18n import _

from pyfemtet._util.helper import *
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
            with time_counting(
                name='Femtet.Exit()',
                warning_time_sec=timeout,
                warning_fun=lambda: logger.warning(
                    _(
                        en_message='Femtet.Exit() does not finished in '
                                   '{timeout} sec. Most common reason is '
                                   'that a dialog is opening in Femtet '
                                   'and waiting user input. Please close '
                                   'the dialog if it exists.',
                        jp_message='Femtet.Exit() は {timeout} 秒以内に終了できませんでした。'
                                   '考えられる理由として、Femtet で予期せずダイアログが開いており'
                                   'ユーザーの入力を待っている場合があります。'
                                   'もしダイアログが存在すれば閉じてください。',
                        timeout=timeout
                    ),
                    # f'Femtet.Exit() が {timeout} 以内に終了していません。'
                    # f'Femtet で予期せずダイアログ等が開いている場合、'
                    # f'入力待ちのため処理が終了しない場合があります。'
                    # f'確認してください。'
                )
            ):
                succeeded = Femtet.Exit(force)

        except AttributeError:
            raise AttributeError(
                _(
                    en_message='Macro version is not consistent to '
                               'the one of Femtet.exe. Please consider to'
                               'execute ``Enable Macros`` of current Femtet '
                               'version to fix it.',
                    jp_message='Femtet.exe のバージョンとマクロのバージョンが一致していません。'
                               '使用中の Femtet.exe と同じバージョンの「マクロ機能を有効化する」'
                               'コマンドを実行してください。',
                )
            )

    # そうでなければ強制終了する
    else:
        # terminate
        util.close_femtet(Femtet.hWnd, timeout, force)

        try:
            succeeded = True
            pid = _get_pid(hwnd)
            start = time()
            while psutil.pid_exists(pid):
                if time() - start > 30:  # 30 秒経っても存在するのは何かおかしい
                    logger.error(_('Failed to close Femtet in '
                                   '30 seconds.',
                                   '30 秒以内に Femtet を終了することが'
                                   'できませんでした。'
                                   ))
                    succeeded = False
                    break
                sleep(1)
            sleep(1)

        # dead
        except (AttributeError, OSError):
            pass

    return succeeded
