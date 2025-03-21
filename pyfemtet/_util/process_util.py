import os

import psutil

import win32process
import win32gui

# noinspection PyUnresolvedReferences
from multiprocessing.context import BaseContext, SpawnProcess, _concrete_contexts
# noinspection PyUnresolvedReferences
from multiprocessing.process import _children, _cleanup
from multiprocessing.managers import SyncManager


__all__ = [
    '_NestableSyncManager',
    '_NestableSpawnProcess',
    '_get_hwnds',
    '_get_pid',
    '_get_pids',
]


# noinspection PyPep8Naming
def _NestableSyncManager():
    m = SyncManager(ctx=_NestableSpawnContext())  # このへんが時間のかかる処理
    m.start()
    return m


def _get_hwnds(pid) -> list[int]:
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


def _get_pid(hwnd) -> int:
    """Window handle から process ID を取得します."""
    if hwnd > 0:
        _, pid = win32process.GetWindowThreadProcessId(hwnd)
    else:
        pid = 0
    return pid


def _get_pids(process_name) -> list[int]:
    """Process のイメージ名から実行中の process ID を取得します."""
    pids = [p.info["pid"] for p in psutil.process_iter(attrs=["pid", "name"]) if p.info["name"] == process_name]
    return pids


# noinspection PyUnresolvedReferences,PyAttributeOutsideInit
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
