
import time
from threading import Thread
from multiprocessing import Process, Manager

import numpy as np
import psutil

import win32process
import win32gui
from win32com.client import Dispatch

from femtetutils import util


pp = True


def _get_hwnds(pid):
    def callback(hwnd, hwnds):
        if win32gui.IsWindowVisible(hwnd) and win32gui.IsWindowEnabled(hwnd):
            _, found_pid = win32process.GetWindowThreadProcessId(hwnd)
            if found_pid == pid:
                hwnds.append(hwnd)
        return True
    hwnds = []
    win32gui.EnumWindows(callback, hwnds)
    return hwnds

def _get_pid(hwnd):
    if hwnd>0:
        _, pid = win32process.GetWindowThreadProcessId(hwnd)
    else:
        pid = 0
    return pid

def _get_pids(process_name):
    pids = [p.info["pid"] for p in psutil.process_iter(attrs=["pid", "name"]) if p.info["name"] == process_name]
    return pids



def Dispatch_Femtet(timeout=10, subprocess_idx=None):
    '''Connect to existing Femtet.

    Args:
        timeout (float): wait second to connect Femtet. 
        subprocess_idx (int or None, optional): subprocess index when called by subprocess. Default to None.

    Returns:
        Femtet (IPyDispatch): COM object of Femtet.
        pid (int): process id of connected Femtet. If error, returns 0.
        
    '''

    #### Dispatch 開始
    if pp:
        if subprocess_idx is not None:
            print(f'  from subprocess{subprocess_idx}; Start to connect Femtet. timeout is {timeout} sec.')
        else:
            print(f'Start to connect Femtet. timeout is {timeout} sec.')
    Femtet = Dispatch('FemtetMacro.Femtet')  # Dispatch は「予約」

    #### Dispatch が成功したことを保証するための処理
    start = time.time()
    # timeout 又は Femtet の hWnd が 0 超になるまで繰り返す
    while True:
        # hwnd を調べる
        hwnd = Femtet.hWnd
        
        # 0 超なら OK
        if hwnd > 0:
            if pp:
                if subprocess_idx is not None:
                    print(f'  from subprocess{subprocess_idx}; Successfully connected.')
                else:
                    print('Successfully connected.')
            break
        
        # そうでなければタイムアウトの判定
        duration = time.time() - start
        if pp:
            if subprocess_idx is not None:
                print(
                    f'  from subprocess{subprocess_idx}; search Femtet ... duration:{int(duration)} < timeout({timeout})')
            else:
                print(f'search Femtet ... duration:{int(duration)} < timeout({timeout})')

        if duration > timeout:
            if pp:
                if subprocess_idx is not None:
                    print(f'  from subprocess{subprocess_idx}; Femtet が正常に実行されていません。')
                else:
                    print('Femtet が正常に実行されていません。')
            return None, 0

        time.sleep(1)

    # hwnd があれば pid が取得できることは保証されるから待たなくていい
    if pp:
        if subprocess_idx is not None:
            print(f'  from subprocess{subprocess_idx};  Connection process finished.')
        else:
            print('Connection process finished.')

    pid = _get_pid(hwnd)
    if pp:
        if subprocess_idx is not None:
            print(f'  from subprocess{subprocess_idx};  pid of connected Femtet is {pid}')
        else:
            print(f'★ pid of connected Femtet is {pid}')

    return Femtet, pid


def _f(pid, subprocess_id, shared_flags):
    """サブプロセスに渡す関数.存在する Femtet と同数の Process が立てられており、この中で Dispatch_Femtet を呼ぶ。"""

    # Dispatch して正常な hwnd が得られるまで待ち、pid を調べる
    if pp: print(f'  from subprocess{subprocess_id}; Start to connect.')
    time.sleep(np.random.rand())  # TODO:排他処理にする
    Femtet, mypid = Dispatch_Femtet(subprocess_idx=subprocess_id)
        
    # Dispatch が終了していることを通知
    shared_flags[subprocess_id] = True

    # pid が目的のものなら即刻開放する
    if mypid==pid:
        if pp: print(f'  from subprocess{subprocess_id}; My pid is it. I will release {pid} and be terminated.')
        return 0
    
    # pid が 0 なら何にも関与しないので即刻終了する
    elif mypid==0:
        if pp: print(f'  from subprocess{subprocess_id}; Failed to connect Femtet. I will be terminated.')
        return 0
    
    # そうでなければメインプロセスが Dispatch を終えるまで解放しない
    else:
        while True:
            if shared_flags[-1]:
                break
            time.sleep(1)
        if pp: print(f'  from subprocess{subprocess_id}; Main process seems to be connected or finished. I will be terminated.')
        return 1


def Dispatch_Femtet_with_new_process():
    """Connect Femtet with new process.

    First, launch new Femtet process and get the pid.
    Next, connect to the launched Femtet using exclusively parallel processing (by Dispatch_Femtet_with_specific_pid).
    Then, 

    Raises:
        RuntimeError: Error raised to be failed to launch Femtet or recognize launched Femtet.
    
    Returns:
        Femtet (IPyDispatch): COM object of Femtet.
        pid (int): process id of connected Femtet. If error, returns 0.

    """

    # Femtet 起動
    if pp: print('Femtet の起動を試行します')
    succeed = util.execute_femtet()
    if not succeed:
        raise RuntimeError('Femtet の起動に失敗しました')

    # pid 取得
    pid = util.get_last_executed_femtet_process_id()
    if pid==0:
        raise RuntimeError('起動された Femtet の認識に失敗しました')
    if pp: print('target pid is', pid)
    
    # ウィンドウが出てくるまで待つ
    timeout = 30
    start = time.time()
    while True:
        # ウィンドウがちゃんと出てきている
        hwnds = _get_hwnds(pid)
        if len(hwnds)>0:
            break
        # タイムアウトした
        duration = time.time() - start
        if duration > timeout:
            if pp: print('Femtet が正常に起動されませんでした。')
            return None, 0
    
    # 目的の pid を持つ Femtet を取得
    Femtet, mypid = Dispatch_Femtet_with_specific_pid(pid)

    # if pp: print(f'pid of connected Femtet is {pid}')
    return Femtet, mypid


def Dispatch_Femtet_with_specific_pid(pid):
    """Connect existing Femtet with specific pid.

    Start Process instances whose number is same to the number of existing Femtet.
    Set the target pid to each Process, and each Process try to connect Femtet.
    The pid of connected Femtet is the target pid, the Process will terminate Immediately.
    Finally, the released Femtet is connected by main process.

    Args:
        pid (int): process id of existing Femtet.
    
    Raises:
        RuntimeError: Error raised to be failed to launch Femtet or recognize launched Femtet.
    
    Returns:
        Femtet (IPyDispatch): COM object of Femtet.
        pid (int): process id of connected Femtet. If error, returns 0.

    """

    #### 目的の pid の Femtet を取得
    # Femtet のプロセスをすべて列挙し、目的のプロセス以外を
    # サブプロセスで Dispatch してブロックする

    # 存在する Femtet プロセスの列挙
    pids = _get_pids('Femtet.exe')
    if pp: print('existing Femtet pids', pids)

    if pid is not None:
        if not (pid in pids):
            raise Exception('指定された pid の Femtet がありません。')

    # 子プロセスの準備
    with Manager() as manager:
        # フラグの準備
        shared_flags = manager.list()
        for _ in range(len(pids)+1): # [-1]は自プロセス用のフラグ
            shared_flags.append(False)
        if pp: print(f'(initial shared_flags are {shared_flags[:]})', )

        # プロセスの準備
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

        # 子プロセスの Dispatch 完了を待つ
        while True:
            # 全てが True していれば次に
            if all(shared_flags[:-1]):
                if pp: print('All subprocesses seem to be connected or finished.')
                break
            if pp: print(f'(shared_flags are {shared_flags[:]} ; wait for signals from subprocess...')
            time.sleep(1)

        # 子プロセスの Dispatch 完了を待って Dispatch
        Femtet, mypid = Dispatch_Femtet()

        # Dispatch 完了を通知
        shared_flags[-1] = True

        # サブプロセスすべての正常終了を待つ
        for p in processes:
            p.join()

        # if pp: print(f'pid of connected Femtet is {mypid}')
        return Femtet, mypid
