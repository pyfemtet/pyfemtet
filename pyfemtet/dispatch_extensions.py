from typing import Tuple

import os
from time import time, sleep
from multiprocessing import current_process
import logging
import warnings

import psutil
from dask.distributed import Lock

import win32process
import win32gui
from win32com.client import Dispatch

from femtetutils import util

from multiprocessing.context import BaseContext, SpawnProcess, _concrete_contexts
from multiprocessing.process import _children, _cleanup


logger = logging.getLogger('dispatch_log')
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('[%(process)d] %(message)s'))
logger.addHandler(handler)
# logger.setLevel(logging.INFO)


DISPATCH_TIMEOUT = 60


class DispatchExtensionException(Exception):
    pass


class FemtetNotFoundException(DispatchExtensionException):
    """Raises when (specific) Femtet process doesn't exist."""


class FemtetConnectionTimeoutError(DispatchExtensionException):
    """Raises when connection trials is timed out."""
    pass


class IFemtet:
    """IDispatch object to contact with Femtet.

    Usage:
        >>> from win32com.client import Dispatch
        >>> from femtetutils import const
        >>> Femtet = Dispatch(const.CFemtet)
        >>> print(Femtet.Version)

        or

        >>> from win32com.client import Dispatch
        >>> Femtet = Dispatch('FemtetMacro.Femtet')
        >>> print(Femtet.Version)

    This is just an dummy class for type hint.
    More detail usage, see Femtet Macro Help.

    """
    pass


class _NestableSpawnProcess(SpawnProcess):
    _start_method = 'nestable_spawn'

    def start(self):
        """This method is modified version of multiprocess.process.BaseProcess.start().

        By using this class, it may become a zombie process.

        """

        #
        # multiprocessing/process.py
        #
        # Copyright (c) 2006-2008, R Oudkerk
        # Licensed to PSF under a Contributor Agreement.
        #

        self._check_closed()
        assert self._popen is None, 'cannot start a process twice'
        assert self._parent_pid == os.getpid(), \
               'can only start a process object created by current process'
        # assert not _current_process._config.get('daemon'), \
        #        'daemonic processes are not allowed to have children'
        _cleanup()
        self._popen = self._Popen(self)
        self._sentinel = self._popen.sentinel
        # Avoid a refcycle if the target function holds an indirect
        # reference to the process object (see bpo-30775)
        del self._target, self._args, self._kwargs
        _children.add(self)


class _NestableSpawnContext(BaseContext):

    #
    # multiprocessing/process.py
    #
    # Copyright (c) 2006-2008, R Oudkerk
    # Licensed to PSF under a Contributor Agreement.
    #

    _name = 'nestable_spawn'
    Process = _NestableSpawnProcess


_concrete_contexts.update(
    dict(nestable_spawn=_NestableSpawnContext())
)


def _log_prefix():
    prefix = current_process().name + ' : '
    return prefix


def _get_hwnds(pid):
    """Proces ID から window handle を取得します."""
    def callback(hwnd, _hwnds):
        if win32gui.IsWindowVisible(hwnd) and win32gui.IsWindowEnabled(hwnd):
            _, found_pid = win32process.GetWindowThreadProcessId(hwnd)
            if found_pid == pid:
                _hwnds.append(hwnd)
        return True
    hwnds = []
    win32gui.EnumWindows(callback, hwnds)
    return hwnds


def _get_pid(hwnd):
    """Window handle から process ID を取得します."""
    if hwnd > 0:
        _, pid = win32process.GetWindowThreadProcessId(hwnd)
    else:
        pid = 0
    return pid


def _get_pids(process_name):
    """Process のイメージ名から実行中の process ID を取得します."""
    pids = [p.info["pid"] for p in psutil.process_iter(attrs=["pid", "name"]) if p.info["name"] == process_name]
    return pids


def launch_and_dispatch_femtet(timeout=DISPATCH_TIMEOUT) -> Tuple[IFemtet, int]:
    """Launch Femtet by new process and connect to it.

    Args:
        timeout (int or float, optional): Seconds to wait for connection. Defaults to DISPATCH_TIMEOUT.

    Raises:
        FemtetNotFoundException: Launched Femtet is not found for some reason (i.e. failed to launch Femtet).
        FemtetConnectionTimeoutError: Connection trial takes over `timeout` seconds by some reason.

    Returns:
        Tuple[IFemtet, int]:
    """

    util.execute_femtet()
    pid = util.get_last_executed_femtet_process_id()
    logger.debug(f'Target pid is {pid}.')
    sleep(5)
    Femtet, pid = dispatch_specific_femtet(pid, timeout)
    return Femtet, pid


def dispatch_femtet(timeout=DISPATCH_TIMEOUT) -> Tuple[IFemtet, int]:
    """Connect to existing Femtet process.

    Args:
        timeout (int or float, optional): Seconds to wait for connection. Defaults to DISPATCH_TIMEOUT.

    Raises:
        FemtetConnectionTimeoutError: Couldn't connect Femtet process for some reason (i.e. Femtet.exe is not launched).

    Returns:
        Tuple[IFemtet, int]:
    """

    # Dispatch
    logger.debug(f'{_log_prefix()}Try to connect Femtet.')
    Femtet = Dispatch('FemtetMacro.Femtet')
    logger.debug(f'{_log_prefix()}Dispatched.')

    # timeout 秒以内に接続が確立するか
    start = time()
    while True:
        hwnd = Femtet.hWnd

        # 接続が確立
        if hwnd > 0:
            logger.debug(f'{_log_prefix()}Current hwnd is {hwnd}.')
            break
        else:
            logger.debug(f'{_log_prefix()}Current hwnd is {hwnd}. Wait for established connection.')

        # 接続がタイムアウト
        if time()-start > timeout:
            raise FemtetConnectionTimeoutError(f'Connect trial with Femtet is timed out in {timeout} sec')

        sleep(1)

    # pid を取得
    pid = _get_pid(hwnd)
    logger.info(f'{_log_prefix()}hwnd is {hwnd} and pid is {pid} of connected Femtet.')

    if ('MainProcess' in _log_prefix()) or 'Dask Worker Process' in _log_prefix():
        print(f'INFO: Process id of connected Femtet = {pid}')

    return Femtet, pid


def _block_other_femtets(
        target_pid,
        subprocess_idx,
        connection_flags,
        timeout_main_wait_for_us,
        lock_inter_subproc,
        # lock_main,
):
    """Target 以外の pid を持つ Femtet と接続し、他の（メインの）プロセスが接続できないようにする。

    ただし、Femtet の Dispatch ルールは一意なので、
    ターゲットの Femtet と接続するまでの他のプロセスだけをブロックする。
    接続可能な Femtet がなければ timeout するまで接続を試行しようとするが、
    ひとつでも timeout すれば、それ以降のプロセスは dispatch する必要がない。

    """

    # Dispatch を行う
    if connection_flags[subprocess_idx]:
        # 他のタイムアウトしたプロセスによってスキップフラグを立てられた場合
        my_pid = 0
    else:
        lock_inter_subproc.acquire()
        try:
            logger.debug(f'{_log_prefix()}Try to block Femtet.')
            _, my_pid = dispatch_femtet(2)
        except DispatchExtensionException:
            my_pid = 0
            # ひとつでもここに来れば、それ以降は探しても無駄なのでスキップフラグを立てる
            for i in range(len(connection_flags)-1):
                connection_flags[i] = True
        finally:
            lock_inter_subproc.release()

    # # target なら Dispatch の終了を通知する前に main を lock する
    # if my_pid == target_pid:
    #     logger.info(f'{_log_prefix()}Connected to {my_pid}. This is the target.')
    #     logger.info(f'{_log_prefix()}Start to block main process.')
    #     lock_main.acquire()

    # Dispatch の終了を通知
    connection_flags[subprocess_idx] = True

    # # pid が目的のものなら他の subprocesses すべてが Dispatch 終了後に開放する
    # pid が目的のものなら即刻開放する
    if my_pid == target_pid:
        # while True:
        #     start = time()
        #     if all(connection_flags[:-1]):
        #         break
        #     if time()-start > timeout_main_wait_for_us:
        #         logger.error(f'{_log_prefix()}Timed out while waiting for other subprocesses.')
        #         return -1
        #     sleep(1)
        # logger.debug(f'{_log_prefix()}All subprocesses seems to finish connection. Release {my_pid} and main.')
        logger.debug(f'{_log_prefix()}Release {my_pid}.')
        # lock_main.release()
        return 0

    # my_pid が非正なら何にも関与しないので即刻終了する
    elif my_pid <= 0:
        logger.debug(f'{_log_prefix()}Failed or timed out to connect Femtet. Exit immediately.')
        return 1

    # そうでなければメインプロセスが Dispatch を終えるまで解放しない
    else:
        logger.debug(f'{_log_prefix()}Connected to {my_pid}. Keep connection to block.')
        start = time()
        while True:
            if connection_flags[-1]:
                break
            if time()-start > timeout_main_wait_for_us:
                logger.error(f'{_log_prefix()}Timed out while waiting for the main process.')
                return -1
            sleep(1)
        logger.debug(f'{_log_prefix()}Main process seems to connect target Femtet. Exit immediately.')
        return 1


def dispatch_specific_femtet(pid, timeout=DISPATCH_TIMEOUT) -> Tuple[IFemtet, int]:
    """Connect Femtet whose process id is specified.

    This is a wrapper function of `dispatch_specific_femtet_core`.
    When this function is called by dask worker,
    the connection is processed exclusively.

    Args:
        pid (int): Process id of Femtet that you want to connect.
        timeout (int or float, optional): Seconds to wait for connection. Defaults to DISPATCH_TIMEOUT.

    Raises:
        FemtetNotFoundException: Femtet whose process id is `pid` doesn't exist. 
        FemtetConnectionTimeoutError: Connection trial takes over `timeout` seconds by some reason.

    Returns:
        Tuple[IFemtet, int]:
    """
    try:
        with Lock('dispatch-specific-femtet'):
            return dispatch_specific_femtet_core(pid, timeout)
    except RuntimeError as e:
        if 'object not properly initialized' in str(e):
            pass
        else:
            raise e
    return dispatch_specific_femtet_core(pid, timeout)  # for logger, out of except.


def dispatch_specific_femtet_core(pid, timeout=10) -> Tuple[IFemtet, int]:
    """Connect Femtet whose process id is specified.

    Warnings:
        Once Femtet is connected a python process,
        the python process can only connect it during the process lifetime
        even if the process runs this function.

    Example:

        If you have 2 free Femtet processes (their pid is 1000 and 1001),
        you can only connect to first connected Femtet
        and cannot change the connection.
        >>> from pyfemtet.dispatch_extensions import dispatch_specific_femtet
        >>> Femtet1, pid1 = dispatch_specific_femtet(pid=1000)
        >>> print(pid1)  # 1000
        >>> Femtet2, pid2 = dispatch_specific_femtet(pid=1001)
        >>> print(pid2)  # not 1001, but 1000

        If you want to reconnect another process, please restart python script or interpreter.

        If you want to connect 2 more Femtet processes from single script or interprete,
        please consider parallel processing.

    Code:

        from multiprocessing import Process
        from pyfemtet.dispatch_extensions import dispatch_specific_femtet

        def connect_femtet(pid):
            Femtet, pid = dispatch_specific_femtet(pid=pid)
            ...  # some processing

        if __name__ == '__main__':
            # setup parallel processing
            p1 = Process(target=connect_femtet, args=(1000,))
            p2 = Process(target=connect_femtet, args=(1001,))

            # start parallel processing
            p1.start()
            p2.start()

            # wait for finish processes
            p1.join()
            p2.join()

    Args:
        pid (int): Process id of Femtet that you want to connect.
        timeout (int or float, optional): Seconds to wait for connection. Defaults to 10.

    Raises:
        FemtetNotFoundException: Femtet whose process id is `pid` doesn't exist. 
        FemtetConnectionTimeoutError: Connection trial takes over `timeout` seconds by some reason.

    Returns:
        Tuple[IFemtet, int]:
    """

    # TODO: 安定性を見て lock_main を復活させるか決める

    if timeout < 5:
        raise ValueError(f'Timeout to dispatch specific femtet should equal or be over 5.')

    # 存在する Femtet プロセスの列挙
    pids = _get_pids('Femtet.exe')
    logger.info(f'{_log_prefix()}PID of existing Femtet processes: {pids}')
    if not (pid in pids):
        raise FemtetNotFoundException(f"Femtet (pid = {pid}) doesn't exist.")

    # 子プロセスの準備
    # with Manager() as manager:
    from multiprocessing.managers import SyncManager
    m = SyncManager(ctx=_NestableSpawnContext())
    m.start()
    with m as manager:
        # フラグの準備
        connection_flags = manager.list()
        lock_inter_subproc = manager.Lock()
        # lock_main = manager.Lock()
        for _ in range(len(pids)+1):  # [-1]は自プロセス用のフラグ
            connection_flags.append(False)

        # 目的以外の Femtet をブロックする子プロセスを開始
        processes = []
        for subprocess_id in range(len(pids)):
            p = _NestableSpawnProcess(
                target=_block_other_femtets,
                args=(
                    pid,
                    subprocess_id,
                    connection_flags,
                    10,
                    lock_inter_subproc,
                    # lock_main,
                ),
            )
            p.start()
            processes.append(p)

        # 子プロセスの Dispatch 完了を待つ
        start = time()
        while True:
            if all(connection_flags[:-1]):
                break
            if time()-start > 10:
                raise FemtetConnectionTimeoutError(f'Connect trial with specific Femtet (pid = {pid}) is timed out in {timeout} sec')
            sleep(1)

        # # subprocesses の Dispatch 終了後、target_pid の解放を待つ
        # lock_main.acquire()

        # 子プロセスによるブロックが終了しているので Dispatch する
        try:
            logger.debug(f'{_log_prefix()}All Femtets except to target seem to be blocked. Try Dispatch.')
            Femtet, my_pid = dispatch_femtet()
        except DispatchExtensionException as e:
            # lock_main.release()
            raise e

        # Dispatch 完了を通知
        connection_flags[-1] = True

        # サブプロセスすべての正常終了を待つ
        for p in processes:
            p.join()

    if my_pid != pid:  # pid の結果が違う場合
        txt = f'Target pid is {pid}, but connected pid is {my_pid}. '
        txt += f'The common reason is that this python process once connected {pid}.'
        warnings.warn(txt)

    return Femtet, my_pid


if __name__ == '__main__':
    _Femtet, _my_pid = launch_and_dispatch_femtet()
    # _Femtet, _my_pid = dispatch_specific_femtet(pid=26124)
