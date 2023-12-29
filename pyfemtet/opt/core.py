import os
import threading
import ctypes
import ray
from win32com.client import constants


os.environ['RAY_DEDUP_LOGS'] = '0'


class Scapegoat:
    """Helper class for parallelize Femtet."""
    # constants を含む関数を並列化するために
    # メイン処理で一時的に constants への参照を
    # このオブジェクトにして、後で restore する
    pass


def restore_constants_from_scapegoat(function: 'Function'):
    """Helper function for parallelize Femtet."""
    fun = function.fun
    for varname in fun.__globals__:
        if isinstance(fun.__globals__[varname], Scapegoat):
            fun.__globals__[varname] = constants


class TerminatableThread(threading.Thread):
    """A terminatable class that inherits from :class:`threading.Thread`."""

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
    """Exception raised for errors in the model update.
    
    If this exception is thrown during an optimization calculation, the process attempts to skip that attempt if possible.

    """
    pass


class MeshError(Exception):
    """Exception raised for errors in the meshing.
    
    If this exception is thrown during an optimization calculation, the process attempts to skip that attempt if possible.

    """
    pass


class SolveError(Exception):
    """Exception raised for errors in the solve.
    
    If this exception is thrown during an optimization calculation, the process attempts to skip that attempt if possible.

    """
    pass


# class PostError(Exception):
#     pass


# class FEMCrash(Exception):
#     pass


class FemtetAutomationError(Exception):
    """Exception raised for errors in automating Femtet."""
    pass


class UserInterruption(Exception):
    """Exception raised for errors in interruption by user."""
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

    def set_allowed_idx(self, idx):
        self.allowed_idx = idx

    def get_allowed_idx(self):
        return self.allowed_idx


class InterprocessVariables:
    """An interface for variables shared between parallel processes."""

    def __init__(self):
        self.ns = _InterprocessVariables.remote()

    def set_state(self, state):
        """Sets the state of entire optimization processes."""
        print(f'---{state}---')
        self.ns.set_state.remote(state)

    def get_state(self):
        """Gets the state of entire optimization processes."""
        return ray.get(self.ns.get_state.remote())

    def set_allowed_idx(self, idx):
        """Sets the allowed subprocess index for exclusive process."""
        self.ns.set_allowed_idx.remote(idx)

    def get_allowed_idx(self):
        """Gets the allowed subprocess index for exclusive process."""
        return ray.get(self.ns.get_allowed_idx.remote())


