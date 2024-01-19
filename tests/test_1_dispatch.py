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
)


def taskkill_femtet():
    sleep(1)
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
        dispatch_femtet(timeout=10)
    except RuntimeError:
        pass
    else:
        assert False


if __name__ == '__main__':
    pass
