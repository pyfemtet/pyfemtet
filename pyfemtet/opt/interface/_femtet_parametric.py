import os
import ctypes
# from ctypes import wintypes
import logging

import numpy as np
import pandas as pd
from femtetutils import util, logger

from pyfemtet.dispatch_extensions import _get_pid
from pyfemtet.core import SolveError
from pyfemtet._message.messages import encoding

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
    dll.GetPrmResult.restype = ctypes.c_double  # 複素数の場合は実部しか取らない
    return dll.GetPrmResult(parametric_result_index)


def solve_via_parametric_dll(Femtet) -> bool:
    # remove previous csv if exists
    # 消さなくても解析はできるが
    # のちのエラーハンドリングのため
    csv_paths = get_csv_paths(Femtet)
    for path in csv_paths:
        if os.path.exists(path):
            os.remove(path)

    # load dll and set target femtet
    dll = _get_dll_with_set_femtet(Femtet)

    # reset
    dll.ClearPrmSweepTable.restype = ctypes.c_bool
    succeed = dll.ClearPrmSweepTable()
    if not succeed:
        return False

    # solve
    dll.PrmCalcExecute.restype = ctypes.c_bool
    succeed = dll.PrmCalcExecute()
    return succeed  # 成功した場合はTRUE、失敗した場合はFALSEを返す


def get_csv_paths(Femtet):

    # 結果フォルダを取得
    path: str = Femtet.Project
    res_dir_path = path.removesuffix('.femprj') + '.Results'

    # csv を取得
    model_name = Femtet.AnalysisModelName
    csv_path = os.path.join(res_dir_path, f'{model_name}.csv')
    table_csv_path = os.path.join(res_dir_path, f'{model_name}_table.csv')

    return csv_path, table_csv_path


def get_table_data_frame(Femtet):

    # csv path を取得
    _, table_csv_path = get_csv_paths(Femtet)

    # csv の存在を確認
    if not os.path.exists(table_csv_path):
        raise SolveError

    # csv があれば読み込む
    df = pd.read_csv(table_csv_path, encoding=encoding)

    return df


if __name__ == '__main__':
    from win32com.client import Dispatch
    Femtet = Dispatch('FemtetMacro.Femtet')
    dll = _get_dll_with_set_femtet(Femtet)

    # solve
    succeeded = solve_via_parametric_dll(Femtet)
    if not succeeded:
        dll.GetLastErrorMsg.restype = ctypes.c_char_p  # or wintypes.LPCSTR
        error_msg: bytes = dll.GetLastErrorMsg()
        error_msg: str = error_msg.decode(encoding='932')

    # 結果取得：内部的にはエラーになっているはず
    parametric_result_index = 1
    dll = _get_dll_with_set_femtet(Femtet)
    dll.GetPrmResult.restype = ctypes.c_double  # 複素数やベクトルの場合は実部や第一成分しか取らない PIF の仕様
    output = dll.GetPrmResult(parametric_result_index)

    # ... だが、下記のコードでそれは出てこない。
    # 値が実際に 0 である場合と切り分けられないので、
    # csv を見てエラーがあるかどうか判断する。
    # dll.GetLastErrorMsg.restype = ctypes.c_char_p  # or wintypes.LPCSTR
    # error_msg: bytes = dll.GetLastErrorMsg()
    # error_msg: str = error_msg.decode(encoding='932')

    # 現時点では table csv に index の情報がないので、
    # エラーがどの番号のものかわからない。
    # ただし、エラーがそのまま出力されるよりマシなので
    # 安全目に引っ掛けることにする
    df = get_table_data_frame(Femtet)
    if 'エラー' in df.columns:
        error_column_data = df['エラー']
    elif 'error' in df.columns:
        error_column_data = df['error']
    else:
        raise RuntimeError('Internal Error! Error message column not found in table csv.')

    # 全部が空欄でないならエラーあり

    if not np.all(error_column_data.isna().values):
        raise SolveError(f'パラメトリック解析結果出力エラー')  # 本当は PostError にすべき

    # そうでなければ結果を返すなどする
    ...
