from typing import Tuple

import os
from time import time, sleep
from multiprocessing import current_process

from tqdm import tqdm
import psutil
from dask.distributed import Lock

import win32process
import win32gui
from win32com.client import Dispatch

from femtetutils import util

from multiprocessing.context import BaseContext, SpawnProcess, _concrete_contexts
from multiprocessing.process import _children, _cleanup
from multiprocessing.managers import SyncManager

import logging
from pyfemtet.logger import get_logger

from pyfemtet._message import Msg


logger = get_logger('dispatch')
logger.setLevel(logging.INFO)


DISPATCH_TIMEOUT = 120


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


def _get_subprocess_log_prefix():
    return f'({current_process().name}) '


def launch_and_dispatch_femtet(timeout=DISPATCH_TIMEOUT, strictly_pid_specify=True) -> Tuple[IFemtet, int]:
    """Launch Femtet by new process and connect to it.

    The wrapper for Dispatch() but returns PID with IFemtet.

    Args:
        timeout (int, optional): Raises an error if the connection is not established within the specified timeout.
        strictly_pid_specify (bool, optional): Attempts to establish a connection to the launched Femtet strictly.
            This may result in slower processing due to process exclusivity handling.

    Raises:
        FemtetConnectionTimeoutError: Couldn't connect Femtet process for some reason (i.e. Femtet.exe is not launched).

    Returns:
        tuple[IFemtet, int]: An object for controlling Femtet and the PID of the Femtet being controlled.

    """
    # launch femtet
    util.execute_femtet()
    pid = util.get_last_executed_femtet_process_id()
    logger.debug(f'Target pid is {pid}.')
    for _ in tqdm(range(5), 'wait for launch femtet...'):
        sleep(1)

    # dispatch femtet
    if strictly_pid_specify:
        Femtet, pid = dispatch_specific_femtet(pid, timeout)
    else:
        # worker process なら排他処理する
        try:
            with Lock('simply-dispatch-femtet'):
                Femtet, pid = dispatch_femtet()
        except RuntimeError as e:
            if "distributed.lock.Lock" in str(e):
                Femtet, pid = dispatch_femtet()
            else:
                raise e
    return Femtet, pid


def dispatch_femtet(timeout=DISPATCH_TIMEOUT, subprocess_log_prefix='') -> Tuple[IFemtet, int]:
    """Connect to existing Femtet process.

    The wrapper for Dispatch() but returns PID with IFemtet.

    Args:
        timeout (int, optional): Raises an error if the connection is not established within the specified timeout.
        subprocess_log_prefix (str, optional): A prefix to output in logs. Typically used only for internal processing.

    Raises:
        FemtetConnectionTimeoutError: Couldn't connect Femtet process for some reason (i.e. Femtet.exe is not launched).

    Returns:
        tuple[IFemtet, int]: An object for controlling Femtet and the PID of the Femtet being controlled.

    """

    """

    Args:
        timeout (int or float, optional): Seconds to wait for connection. Defaults to DISPATCH_TIMEOUT.
        subprocess_log_prefix (str, optional): The prefix of log message.

    Raises:
        FemtetConnectionTimeoutError: Couldn't connect Femtet process for some reason (i.e. Femtet.exe is not launched).

    Returns:
        Tuple[IFemtet, int]:
    """
    # Dispatch
    if subprocess_log_prefix:
        logger.debug('%s'+'Try to connect Femtet.', subprocess_log_prefix)
    else:
        logger.info('Try to connect Femtet.')
    Femtet = Dispatch('FemtetMacro.Femtet')
    logger.debug('%s'+'Dispatch finished.', subprocess_log_prefix)

    # timeout 秒以内に接続が確立するか
    start = time()
    while True:
        hwnd = Femtet.hWnd

        # 接続が確立
        if hwnd > 0:
            logger.debug('%s'+f'Dispatched hwnd is {hwnd} and its pid is {_get_pid(hwnd)}. Connection established.', subprocess_log_prefix)
            break
        else:
            logger.debug('%s'+f'Dispatched hwnd is {hwnd}. Waiting for establishing connection.', subprocess_log_prefix)

        # 接続がタイムアウト
        if time()-start > timeout:
            raise FemtetConnectionTimeoutError(f'Connection trial with Femtet is timed out in {timeout} sec')

        sleep(1)

    # pid を取得
    pid = _get_pid(hwnd)
    if subprocess_log_prefix:
        logger.debug('%s'+f'Successfully connected. The pid of Femtet is {pid}.', subprocess_log_prefix)
    else:
        logger.info(f'Successfully connected. The pid of Femtet is {pid}.')

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
            logger.debug('%s'+f'Subprocess started.', _get_subprocess_log_prefix())
            _, my_pid = dispatch_femtet(2, _get_subprocess_log_prefix())
        except DispatchExtensionException:
            logger.debug('%s'+f'Connection failed. The other subprocesses will skip.', _get_subprocess_log_prefix())
            my_pid = 0
            # ひとつでもここに来れば、それ以降は探しても無駄なのでスキップフラグを立てる
            for i in range(len(connection_flags)-1):
                connection_flags[i] = True
        finally:
            lock_inter_subproc.release()

    # # target なら Dispatch の終了を通知する前に main を lock する
    # if my_pid == target_pid:
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
        #         return -1
        #     sleep(1)
        logger.debug('%s'+f'Release {my_pid}.', _get_subprocess_log_prefix())
        # lock_main.release()
        return 0

    # my_pid が非正なら何にも関与しないので即刻終了する
    elif my_pid <= 0:
        logger.debug('%s'+'Failed or timed out to connect Femtet.', _get_subprocess_log_prefix())
        return 1

    # そうでなければメインプロセスが Dispatch を終えるまで解放しない
    else:
        logger.debug('%s'+f'Connected to {my_pid}. Keep connection to block.', _get_subprocess_log_prefix())
        start = time()
        while True:
            if connection_flags[-1]:
                logger.debug('%s'+'Main process seems to connect target Femtet.', _get_subprocess_log_prefix())
                return 1
            if time()-start > timeout_main_wait_for_us:
                logger.warning(
                    '%s'+f'Timed out to wait for the main process to {timeout_main_wait_for_us}',
                    _get_subprocess_log_prefix()
                )
                return -1
            sleep(1)


def dispatch_specific_femtet(pid, timeout=DISPATCH_TIMEOUT) -> Tuple[IFemtet, int]:
    """Connect Existing Femtet whose process id is specified.

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

    Args:
        timeout (int, optional): Raises an error if the connection is not established within the specified timeout.

    Raises:
        FemtetConnectionTimeoutError: Couldn't connect Femtet process for some reason (i.e. Femtet.exe is not launched).

    Returns:
        tuple[IFemtet, int]: An object for controlling Femtet and the PID of the Femtet being controlled.


    """
    try:
        with Lock('dispatch-specific-femtet'):
            return _dispatch_specific_femtet_core(pid, timeout)
    except RuntimeError as e:
        if 'object not properly initialized' in str(e):
            pass
        else:
            raise e
    return _dispatch_specific_femtet_core(pid, timeout)  # for logger, out of except.


def _dispatch_specific_femtet_core(pid, timeout=DISPATCH_TIMEOUT) -> Tuple[IFemtet, int]:

    # TODO: 安定性を見て lock_main を復活させるか決める

    if timeout < 5:
        raise ValueError(f'Timeout to dispatch specific femtet should equal or be over 5.')

    # 存在する Femtet プロセスの列挙
    pids = _get_pids('Femtet.exe')
    logger.info(f'PID of existing Femtet processes: {pids}')
    if not (pid in pids):
        raise FemtetNotFoundException(f"Femtet (pid = {pid}) doesn't exist.")

    logger.info('Searching specific Femtet...')

    # 子プロセスの準備
    # with Manager() as manager:
    m = SyncManager(ctx=_NestableSpawnContext())  # このへんが時間のかかる処理
    m.start()
    with m as manager:
        # フラグの準備
        connection_flags = manager.list()
        lock_inter_subproc = manager.Lock()
        # lock_main = manager.Lock()
        for _ in range(len(pids)+1):  # [-1]は自プロセス用のフラグ
            connection_flags.append(False)

        # 目的以外の Femtet をブロックする子プロセスを開始
        logger.debug('Start subprocess to block Femtet other than target pid.')
        processes = []
        for subprocess_id in tqdm(range(len(pids)), 'Specifying connection...'):
            p = _NestableSpawnProcess(
            # p = Process(
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
            if time()-start > timeout:
                # 子プロセスを終了する
                for p in processes:
                    p.terminate()
                    p.join()
                try:
                    lock_inter_subproc.release()
                except RuntimeError:
                    pass
                raise FemtetConnectionTimeoutError(f'Connect trial with specific Femtet (pid = {pid}) is timed out in {timeout} sec')
            sleep(1)

        # # subprocesses の Dispatch 終了後、target_pid の解放を待つ
        # lock_main.acquire()

        # 子プロセスによるブロックが終了しているので Dispatch する
        try:
            logger.debug('Block process seems to be finished. Try Dispatch.')
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
        txt += f'The common reason is that THIS python process once connected {pid}.'
        logger.warn(txt)
        # warnings.warn(txt)

    return Femtet, my_pid


def _debug():
    launch_and_dispatch_femtet(5)


if __name__ == '__main__':
    _Femtet, _my_pid = launch_and_dispatch_femtet(5)
    # _Femtet, _my_pid = dispatch_specific_femtet(pid=26124)
