import time
from threading import Thread
from multiprocessing import Process, Manager
import psutil

import win32process
from win32com.client import Dispatch

from femtetutils import util


def _get_pid(hwnd):
    if hwnd>0:
        _, pid = win32process.GetWindowThreadProcessId(hwnd)
    else:
        pid = 0
    return pid


def _get_pids(process_name):
    pids = [p.info["pid"] for p in psutil.process_iter(attrs=["pid", "name"]) if p.info["name"] == process_name]
    return pids
# _get_pids('Femtet.exe')


def Dispatch_Femtet_with_pid():
    '''
    Femtet の Dispatch を行う。
    何かで失敗があった場合、Exception を送出する。

    Returns
    -------
    Femtet : pyIDispatch
    pid : int
        process id. if error, pid = 0.

    '''
    Femtet = Dispatch('FemtetMacro.Femtet') # Dispatch は「予約」
    # Dispatch が成功したことを保証する
    timeout = 10
    start = time.time()
    # timeout 又は Femtet の hWnd が 0 超になるまで繰り返す
    while True:
        hwnd = Femtet.hWnd
        if hwnd>0:
            break
        now = time.time()
        duration = now - start
        if duration > timeout:
            raise Exception('Femtet が正常に実行されていません。')
    # hwnd があれば pid が取得できることは保証されるから待たなくていい
    pid = _get_pid(hwnd)
    return Femtet, pid


def _f(pid, subprocess_id, shared_flags):
    # 自分が準備できたことのフラグ
    shared_flags[subprocess_id] = True
    # 他のプロセスが準備できるまで待つ
    while True:
        time.sleep(0.1)
        if all(shared_flags[:-1]):
            break
    # Dispatch して pid を調べる
    Femtet, mypid = Dispatch_Femtet_with_pid()
    # pid が目的のものなら即刻開放する
    if mypid==pid:
        return 1
    # そうでなければメインプロセスが Dispatch を終えるまで解放しない
    else:
        while True:
            time.sleep(0.5)
            if shared_flags[-1]:
                break
        return 0


def Dispatch_Femtet_with_new_process(pp=False):
    '''
    Dispatch Femtet する。
    femtetutils を用いて新しいプロセスを立て、
    そのプロセスと繋ぐことを保証する。

    Parameters
    ----------
    pp : TYPE, optional
        print_progress, if True, show progress info. The default is False.

    Raises
    ------
    Exception
        何らかの失敗

    Returns
    -------
    Femtet : pyIDispatch
    pid : int
        process id. if error, pid = 0.

    '''
    # Femtet 起動
    succeed = util.execute_femtet()
    if not succeed:
        raise Exception('femtetutils を用いた Femtet の起動に失敗しました')

    # pid 取得
    pid = util.get_last_executed_femtet_process_id()
    if pid==0:
        raise Exception('起動された Femtet の認識に失敗しました')
    if pp: print('pid', pid)
    
    #### 目的の pid の Femtet を取得
    # Femtet のプロセスをすべて列挙し、目的のプロセス以外を
    # サブプロセスで Dispatch してブロックする
    # -Femtet プロセスの列挙
    pids = _get_pids('Femtet.exe')
    if pp: print('existing pids', pids)
    # -子プロセスの準備
    with Manager() as manager:
        shared_flags = manager.list()
        for _ in range(len(pids)+1): # [-1]は自プロセス用のフラグ
            shared_flags.append(False)
        if pp: print('initial shared_flags', shared_flags[:])
        processes = []
        for subprocess_id in range(len(pids)):
            p = Process(
                target=_f,
                args=(
                    pid,
                    subprocess_id,
                    shared_flags
                    )
                )
            p.start()
            processes.append(p)
        # 子プロセスの準備完了を待つ
        while True:
            if pp: 
                time.sleep(1)
                print('shared_flags', shared_flags[:])
            time.sleep(.1)
            if all(shared_flags[:-1]):
                print('shared_flags', shared_flags[:])
                break
        # 1.5 秒（timeout+ちょっと）待って Dispatch
        time.sleep(1.5)
        Femtet, mypid = Dispatch_Femtet_with_pid()
        # Dispatch 完了を通知
        shared_flags[-1] = True
        # サブプロセスすべての正常終了を待つ
        for p in processes:
            p.join(timeout=10)
    print(f'pid of Dispatch object of Femtet:{_get_pid(Femtet.hWnd)}')
    return Femtet, _get_pid(Femtet.hWnd)


if __name__=='__main__':
    Femtet = Dispatch_Femtet_with_new_process(True)
    
    
    