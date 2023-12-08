
import time
from threading import Thread
from multiprocessing import Process, Manager
import psutil

import win32process
import win32gui
from win32com.client import Dispatch


from femtetutils import util


pp = False



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



def Dispatch_Femtet(timeout=30):
    '''
    Femtet の Dispatch を行う。

    Returns
    -------
    Femtet : pyIDispatch
    pid : int
        process id. if error, pid = 0.

    '''

    #### Dispatch 開始
    if pp: print('start Dispatch')
    Femtet = Dispatch('FemtetMacro.Femtet') # Dispatch は「予約」

    #### Dispatch が成功したことを保証するための処理
    start = time.time()
    # timeout 又は Femtet の hWnd が 0 超になるまで繰り返す
    while True:
        # hwnd を調べる
        hwnd = Femtet.hWnd
        
        # 0 超なら OK
        if hwnd>0:
            if pp: print('Dispatch finished successfully')
            break
        
        # そうでなければタイムアウトの判定
        duration = time.time() - start
        if pp: print('Dispatch duration', duration)
        if duration > timeout:
            if pp: print('Femtet が正常に実行されていません。')
            return None, 0

        time.sleep(1)

    # hwnd があれば pid が取得できることは保証されるから待たなくていい
    if pp: print('Dispatch finished')
    pid = _get_pid(hwnd)

    return Femtet, pid


def _f(pid, subprocess_id, shared_flags):

    # Dispatch して正常な hwnd が得られるまで待ち、pid を調べる
    if pp: print(f'from subp{subprocess_id}; let us start Dispatch...')
    Femtet, mypid = Dispatch_Femtet()
        
    # Dispatch が終了していることを通知
    shared_flags[subprocess_id] = True

    # pid が目的のものなら即刻開放する
    if mypid==pid:
        if pp: print(f'from subp{subprocess_id}; my pid is hit. i will release {pid}')
        return 0
    
    # pid が 0 なら何にも関与しないので即刻終了する
    elif mypid==0:
        if pp: print(f'from subp{subprocess_id}; failed to connect Femtet. i will finish.')
        return 0
    
    # そうでなければメインプロセスが Dispatch を終えるまで解放しない
    else:
        while True:
            if shared_flags[-1]:
                break
            time.sleep(1)
        if pp: print(f'from subp{subprocess_id}; main process seems to have been Dispatched. i will finish')
        return 1


def Dispatch_Femtet_with_new_process():
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

    print(f'pid of Dispatch object of Femtet:{mypid}')
    return Femtet, mypid


def Dispatch_Femtet_with_specific_pid(pid):
    #### 目的の pid の Femtet を取得
    # Femtet のプロセスをすべて列挙し、目的のプロセス以外を
    # サブプロセスで Dispatch してブロックする

    # 存在する Femtet プロセスの列挙
    pids = _get_pids('Femtet.exe')
    if pp: print('existing pids', pids)

    # 子プロセスの準備
    with Manager() as manager:
        # フラグの準備
        shared_flags = manager.list()
        for _ in range(len(pids)+1): # [-1]は自プロセス用のフラグ
            shared_flags.append(False)
        if pp: print('initial shared_flags', shared_flags[:])

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
                if pp: print('finished: shared_flags', shared_flags[:])
                break
            if pp: print('shared_flags', shared_flags[:])
            time.sleep(1)

        # 子プロセスの Dispatch 完了を待って Dispatch
        Femtet, mypid = Dispatch_Femtet()
        if pp: print('finally Dispatched pid', mypid)

        # Dispatch 完了を通知
        shared_flags[-1] = True

        # サブプロセスすべての正常終了を待つ
        for p in processes:
            p.join()
        
        return Femtet, mypid


if __name__=='__main__':
    Femtet, pid = Dispatch_Femtet_with_new_process()
    start = time.time()
    Femtet.LoadProject(r'C:\Users\mm11592\Documents\myFiles2\working\1_PyFemtetOpt\PyFemtetOptDevelopment\PyFemtetPrj\tests\simple_femtet\simple.femprj', True)
    end = time.time()
    print(end-start)
    
    
    