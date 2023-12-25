import os
import threading
import ctypes
import ray
from win32com.client import constants


os.environ['RAY_DEDUP_LOGS'] = '0'


class Scapegoat:
    # constants を含む関数を並列化するために
    # メイン処理で一時的に constants への参照を
    # このオブジェクトにして、後で restore する
    pass


def restore_constants_from_scapegoat(function: 'Function'):
    fun = function.fun
    for varname in fun.__globals__:
        if isinstance(fun.__globals__[varname], Scapegoat):
            fun.__globals__[varname] = constants


class TerminatableThread(threading.Thread):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._run = self.run
        self.run = self.set_id_and_run

    def set_id_and_run(self):
        self.id = threading.get_native_id()
        self._run()

    def get_id(self):
        return self.id

    def force_terminate(self):
        res = ctypes.pythonapi.PyThreadState_SetAsyncExc(
            ctypes.c_long(self.get_id()),
            ctypes.py_object(SystemExit)
        )
        if res > 1:
            ctypes.pythonapi.PyThreadState_SetAsyncExc(
                ctypes.c_long(self.get_id()),
                0
            )


class ModelError(Exception):
    pass


class MeshError(Exception):
    pass


class SolveError(Exception):
    pass


class PostError(Exception):
    pass


class FEMCrash(Exception):
    pass


class FemtetAutomationError(Exception):
    pass


class UserInterruption(Exception):
    pass


@ray.remote
class _InterprocessVariables:

    def __init__(self):
        self.state = 'undefined'
        self.history = []  # #16295
        self.allowed_idx = 0

    def set_state(self, state):
        self.state = state

    def get_state(self) -> 'ObjectRef':
        return self.state

    def append_history(self, row):
        self.history.append(row)

    def get_history(self) -> 'ObjectRef':
        return self.history

    def set_allowed_idx(self, idx):
        self.allowed_idx = idx

    def get_allowed_idx(self):
        return self.allowed_idx


class InterprocessVariables:

    def __init__(self):
        self.ns = _InterprocessVariables.remote()

    def set_state(self, state):
        print(f'---{state}---')
        self.ns.set_state.remote(state)

    def get_state(self):
        return ray.get(self.ns.get_state.remote())

    def append_history(self, row):
        self.ns.append_history.remote(row)

    def get_history(self):
        return ray.get(self.ns.get_history.remote())

    def set_allowed_idx(self, idx):
        self.ns.set_allowed_idx.remote(idx)

    def get_allowed_idx(self):
        return ray.get(self.ns.get_allowed_idx.remote())


