from time import sleep
from subprocess import run

from femtetutils import util
from pyfemtet.dispatch_extensions import (
dispatch_femtet,
dispatch_specific_femtet,
dispatch_specific_femtet_core,
launch_and_dispatch_femtet,
_get_pid,
_get_pids,
_get_hwnds,
DispatchExtensionException
)
from pyfemtet.opt import FemtetInterface


def taskkill_femtet():
    sleep(3)
    run(['taskkill', '/f', '/im', 'Femtet.exe'])


def test_launch_and_connect_femtet():

    # launch some Femtets
    for _ in range(3):
        util.execute_femtet()

    # launch and dispatch
    Femtet, pid = launch_and_dispatch_femtet()

    # check launched?
    pids = _get_pids('Femtet.exe')

    # Is the Femtet it?
    _pid = _get_pid(Femtet.hWnd)

    # kill Femtet all
    taskkill_femtet()

    assert pid in pids
    assert pid == _pid


def test_dispatch_femtet():
    # launch a Femtet
    util.execute_femtet()
    sleep(10)

    # connect anyway
    Femtet, pid = dispatch_femtet()
    _pid = _get_pid(Femtet.hWnd)

    taskkill_femtet()

    assert pid == _pid


def test_miss_dispatch_femtet():
    # dispatch with no Femtet
    try:
        dispatch_femtet(timeout=3)
    except DispatchExtensionException:
        pass
    else:
        assert False


def test_FemtetInterface():

    # 既存の Femtet を消す
    taskkill_femtet()

    import os
    os.chdir(os.path.dirname(__file__))

    femprj_path = __file__.replace('.py', '.femprj')

    # 1. 既存の Femtet に auto で接続

    # Femtet 起動
    util.execute_femtet()
    sleep(15)

    # 接続
    fem = FemtetInterface(femprj_path=femprj_path, connect_method='auto')

    # 接続確認
    hwnd1 = fem.Femtet.hWnd
    isOK_auto_connect = hwnd1 > 0
    pid1_1 = _get_pid(hwnd1)

    # デストラクト
    del fem

    # 終了していないことを確認
    pid1_2 = _get_pid(hwnd1)
    isOK_not_quit_existed = pid1_1 == pid1_2

    taskkill_femtet()


    # 2. Femtet に new で接続

    # 余分な Femtet 起動
    util.execute_femtet()
    fem = FemtetInterface(femprj_path=femprj_path, connect_method='new')

    # 接続確認
    hwnd2 = fem.Femtet.hWnd
    isOK_new_connect = hwnd2 > 0

    # 元の Femtet とは別の Femtet に接続したことを確認
    isOK_launch_and_connect_another_one = hwnd1 != hwnd2

    # デストラクト
    pid2 = _get_pid(hwnd2)
    del fem

    # quit 確認
    import psutil
    isOK_launched_Femtet_is_quited_on_del = not psutil.pid_exists(pid2)

    # 一応
    taskkill_femtet()

    assert isOK_auto_connect
    assert isOK_not_quit_existed
    assert isOK_launched_Femtet_is_quited_on_del
    assert isOK_new_connect
    assert isOK_launch_and_connect_another_one


def test_empty_femtet():
    try:
        FemtetInterface(connect_method='new')
        raise Exception('RuntimeError should be raised.')
    except RuntimeError:
        pass


def test_allow_empty_femtet():
    taskkill_femtet()
    sleep(3)
    fem = FemtetInterface(allow_without_project=True)
    assert fem.femtet_pid > 0, 'Femtet が起動できていません。'
    assert fem.Femtet.hWnd == _get_hwnds(fem.femtet_pid)[0], '起動された Femtet と接続していません。'
    taskkill_femtet()


if __name__ == '__main__':
    # test_FemtetInterface()
    pass
