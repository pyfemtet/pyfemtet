import os
from time import time, sleep
from multiprocessing import current_process
import logging

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


class _NestableSpawnProcess(SpawnProcess):
    _start_method = 'nestable_spawn'

    def start(self):
        """This method is modified version of multiprocess.process.BaseProcess.start().

        By using this class, it may become a zombie process.

        """
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


def launch_and_dispatch_femtet(timeout=DISPATCH_TIMEOUT) -> ['PyIDispatch', int]:
    """Femtet を新しいプロセスで起動し、Dispatch します."""
    util.execute_femtet()
    pid = util.get_last_executed_femtet_process_id()
    sleep(3)
    Femtet, pid = dispatch_specific_femtet(pid, timeout)
    return Femtet, pid


def dispatch_femtet(timeout=DISPATCH_TIMEOUT) -> ['PyIDispatch', int]:
    """Femtet を Dispatch します."""
    # Dispatch
    logger.debug(f'{_log_prefix()}Try to connect Femtet.')
    Femtet = Dispatch('FemtetMacro.Femtet')

    # timeout 秒以内に接続が確立するか
    start = time()
    while True:
        hwnd = Femtet.hWnd
        logger.debug(f'{_log_prefix()}Current hwnd is {hwnd}.')

        # 接続が確立
        if hwnd > 0:
            break

        # 接続がタイムアウト
        if time()-start > timeout:
            raise RuntimeError('Femtet との接続に失敗しました。')

        sleep(1)

    # pid を取得
    pid = _get_pid(hwnd)
    logger.info(f'{_log_prefix()}PID of connected Femtet is {pid}')

    if ('MainProcess' in _log_prefix()) or 'Dask Worker Process' in _log_prefix():
        print(f'INFO: Process id of connected Femtet = {pid}')

    return Femtet, pid


def _block_other_femtets(target_pid, subprocess_idx, connection_flags, timeout, lock):

    # Dispatch を行う
    logger.debug(f'{_log_prefix()}Try to block Femtet.')
    lock.acquire()
    try:
        _, my_pid = dispatch_femtet(timeout)
    finally:
        lock.release()
    # _, my_pid = dispatch_femtet(timeout)

    # Dispatch の終了を通知
    connection_flags[subprocess_idx] = True

    # pid が目的のものなら即刻開放する
    if my_pid == target_pid:
        logger.info(f'{_log_prefix()}Connected to {my_pid}. Release immediately.')
        return 0

    # my_pid が 0 なら何にも関与しないので即刻終了する
    elif my_pid == 0:
        logger.info(f'{_log_prefix()}Failed to connect Femtet. Exit immediately.')
        return -1

    # そうでなければメインプロセスが Dispatch を終えるまで解放しない
    else:
        logger.info(f'{_log_prefix()}Connected to {my_pid}. Keep connection to block.')
        start = time()
        while True:
            if connection_flags[-1]:
                break
            if time()-start > timeout+1:
                logger.error(f'{_log_prefix()}Timed out while waiting for the main process.')
                raise RuntimeError('メインプロセスの Femtet 接続が失敗しています。')
            sleep(1)
        logger.info(f'{_log_prefix()}Main process seems to connect target Femtet. Exit immediately.')
        return 1


def dispatch_specific_femtet(pid, timeout=DISPATCH_TIMEOUT) -> ['PyIDispatch', int]:
    try:
        with Lock('dispatch-specific-femtet'):
            return dispatch_specific_femtet_core(pid, timeout)
    except RuntimeError as e:
        if 'object not properly initialized' in str(e):
            return dispatch_specific_femtet_core(pid, timeout)
        else:
            raise e


def dispatch_specific_femtet_core(pid, timeout=DISPATCH_TIMEOUT) -> ['PyIDispatch', int]:

    # 存在する Femtet プロセスの列挙
    pids = _get_pids('Femtet.exe')
    logger.info(f'{_log_prefix()}PID of existing Femtet processes: {pids}')
    if not (pid in pids):
        raise RuntimeError('指定された pid の Femtet がありません。')

    # 子プロセスの準備
    # with Manager() as manager:
    from multiprocessing.managers import SyncManager
    m = SyncManager(ctx=_NestableSpawnContext())
    m.start()
    with m as manager:
        # フラグの準備
        connection_flags = manager.list()
        lock = manager.Lock()
        for _ in range(len(pids)+1):  # [-1]は自プロセス用のフラグ
            connection_flags.append(False)

        # 目的以外の Femtet をブロックする子プロセスを開始
        processes = []
        for subprocess_id in range(len(pids)):
            p = _NestableSpawnProcess(
                target=_block_other_femtets,
                args=(pid, subprocess_id, connection_flags, timeout, lock),
            )
            p.start()
            processes.append(p)

        # 子プロセスの Dispatch 完了を待つ
        start = time()
        while True:
            if all(connection_flags[:-1]):
                break
            if time()-start > timeout:
                raise RuntimeError('Femtet との接続に失敗しました.')
            sleep(1)

        # 子プロセスによるブロックが終了しているので Dispatch する
        Femtet, my_pid = dispatch_femtet()

        # Dispatch 完了を通知
        connection_flags[-1] = True

        # サブプロセスすべての正常終了を待つ
        for p in processes:
            p.join()

    return Femtet, my_pid


if __name__ == '__main__':
    _Femtet, _my_pid = launch_and_dispatch_femtet()
    # _Femtet, _my_pid = dispatch_specific_femtet(pid=26124)
