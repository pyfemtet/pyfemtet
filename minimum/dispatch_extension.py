from time import time, sleep
from multiprocessing import Process, Manager, current_process

import logging

import psutil

import win32process
import win32gui
from win32com.client import Dispatch

from femtetutils import util


logger = logging.getLogger('dispatch_log')
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('[%(process)d] %(message)s'))
logger.addHandler(handler)
# logger.setLevel(logging.DEBUG)


DISPATCH_TIMEOUT = 10


def _log_prefix():
    prefix = ''
    if current_process().name != 'MainProcess':
        prefix = '(sub) '
    return prefix


def _get_hwnds(pid):
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
    if hwnd > 0:
        _, pid = win32process.GetWindowThreadProcessId(hwnd)
    else:
        pid = 0
    return pid


def _get_pids(process_name):
    pids = [p.info["pid"] for p in psutil.process_iter(attrs=["pid", "name"]) if p.info["name"] == process_name]
    return pids


def launch_and_dispatch_femtet(timeout=DISPATCH_TIMEOUT):
    util.execute_femtet()
    pid = util.get_last_executed_femtet_process_id()
    sleep(3)
    return dispatch_specific_femtet(pid, timeout)


def dispatch_femtet(timeout=DISPATCH_TIMEOUT):
    # Dispatch
    Femtet = Dispatch('FemtetMacro.Femtet')

    # timeout 秒以内に接続が確立するか
    start = time()
    while True:
        hwnd = Femtet.hWnd

        # 接続が確立
        if hwnd > 0:
            break

        # 接続がタイムアウト
        if time()-start > timeout:
            raise RuntimeError('Femtet との接続に失敗しました。')

    # pid を取得
    pid = _get_pid(hwnd)
    logger.info(f'{_log_prefix()}PID of connected Femtet is {pid}')

    if _log_prefix() == '':
        print(f'INFO: Process id of connected Femtet = {pid}')

    return Femtet, pid


def _block_other_femtets(target_pid, subprocess_idx, connection_flags, timeout, lock):

    # Dispatch を行う
    lock.acquire()
    try:
        _, my_pid = dispatch_femtet(timeout)
    finally:
        lock.release()

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
        while True:
            if connection_flags[-1]:
                break
            sleep(1)
        logger.info(f'{_log_prefix()}Main process seems to connect target Femtet. Exit immediately.')
        return 1


def dispatch_specific_femtet(pid, timeout=DISPATCH_TIMEOUT):

    # 存在する Femtet プロセスの列挙
    pids = _get_pids('Femtet.exe')
    logger.info(f'{_log_prefix()}PID of existing Femtet processes: {pids}')
    if not (pid in pids):
        raise RuntimeError('指定された pid の Femtet がありません。')

    # 子プロセスの準備
    with Manager() as manager:
        # フラグの準備
        connection_flags = manager.list()
        lock = manager.Lock()
        for _ in range(len(pids)+1):  # [-1]は自プロセス用のフラグ
            connection_flags.append(False)

        # 目的以外の Femtet をブロックする子プロセスを開始
        processes = []
        for subprocess_id in range(len(pids)):
            p = Process(
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
