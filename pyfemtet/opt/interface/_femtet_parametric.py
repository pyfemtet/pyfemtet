import logging

from femtetutils import util, logger
from pyfemtet.dispatch_extensions import _get_pid

import ctypes


logger.setLevel(logging.ERROR)


def _get_dll():
    femtet_exe_path = util.get_femtet_exe_path()
    dll_path = femtet_exe_path.replace('Femtet.exe', 'ParametricIF.dll')
    return ctypes.cdll.LoadLibrary(dll_path)


def _get_dll_with_set_femtet(Femtet):
    dll = _get_dll()
    pid = _get_pid(Femtet.hWnd)
    dll.SetCurrentFemtet.restype = ctypes.c_bool
    dll.SetCurrentFemtet(pid)
    return dll


def _get_prm_result_names(Femtet):
    """Used by pyfemtet-opt-gui"""
    out = []

    # load dll and set target femtet
    dll = _get_dll_with_set_femtet(Femtet)
    n = dll.GetPrmnResult()
    for i in range(n):
        # objective name
        dll.GetPrmResultName.restype = ctypes.c_char_p
        result = dll.GetPrmResultName(i)
        name = result.decode('mbcs')
        # objective value function
        out.append(name)
    return out


def add_parametric_results_as_objectives(femopt, indexes, directions) -> bool:
    # load dll and set target femtet
    dll = _get_dll_with_set_femtet(femopt.fem.Femtet)

    # get objective names
    dll.GetPrmnResult.restype = ctypes.c_int
    n = dll.GetPrmnResult()
    for i, direction in zip(indexes, directions):
        # objective name
        dll.GetPrmResultName.restype = ctypes.c_char_p
        result = dll.GetPrmResultName(i)
        name = result.decode('mbcs')
        # objective value function
        femopt.add_objective(_parametric_objective, name, direction=direction, args=(i,))
    return True  # ここまで来たら成功


def _parametric_objective(Femtet, parametric_result_index):
    # load dll and set target femtet
    dll = _get_dll_with_set_femtet(Femtet)
    dll.GetPrmResult.restype = ctypes.c_double
    return dll.GetPrmResult(parametric_result_index)


def solve_via_parametric_dll(Femtet) -> bool:
    # load dll and set target femtet
    dll = _get_dll_with_set_femtet(Femtet)
    # solve
    dll.PrmCalcExecute.restype = ctypes.c_bool
    succeed = dll.PrmCalcExecute()
    return succeed  # 成功した場合はTRUE、失敗した場合はFALSEを返す
