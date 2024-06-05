from pyfemtet.dispatch_extensions import _get_pid

import winreg
import ctypes


def _get_dll(Femtet):
    femtet_major_version = _get_femtet_major_version(Femtet)
    dll_path = _get_parametric_dll_path(femtet_major_version)
    return ctypes.cdll.LoadLibrary(dll_path)


def _get_femtet_major_version(Femtet):
    from pyfemtet.core import _version
    version_int8 = _version(Femtet=Femtet)
    return str(version_int8)[0:4]


def _get_parametric_dll_path(femtet_major_version) -> str:
    key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'SOFTWARE\FemtetInfo\InstallVersion\x64')
    _, nValues, _ = winreg.QueryInfoKey(key)
    for i in range(nValues):
        name, data, _ = winreg.EnumValue(key, i)
        if name == str(femtet_major_version):
            winreg.CloseKey(key)
            import os
            dll_path = os.path.join(data, 'Program', 'ParametricIF.dll')
            return dll_path
    # ここまで来ていたら失敗
    winreg.CloseKey(key)
    raise RuntimeError('パラメトリック解析機能へのアクセスに失敗しました')


def _get_dll_with_set_femtet(Femtet):
    dll = _get_dll(Femtet)
    pid = _get_pid(Femtet.hWnd)
    dll.SetCurrentFemtet.restype = ctypes.c_bool
    dll.SetCurrentFemtet(pid)
    return dll


def _get_prm_result_names(Femtet):
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


def add_parametric_results_as_objectives(femopt, indexes) -> bool:
    # load dll and set target femtet
    dll = _get_dll_with_set_femtet(femopt.fem.Femtet)

    # get objective names
    dll.GetPrmnResult.restype = ctypes.c_int
    n = dll.GetPrmnResult()
    for i in indexes:
        # objective name
        dll.GetPrmResultName.restype = ctypes.c_char_p
        result = dll.GetPrmResultName(i)
        name = result.decode('mbcs')
        # objective value function
        femopt.add_objective(_parametric_objective, name, args=(i,))
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
