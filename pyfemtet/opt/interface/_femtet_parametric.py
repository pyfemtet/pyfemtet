import os
import ctypes
# from ctypes import wintypes
import logging
import warnings
from time import sleep, time

import numpy as np
import pandas as pd
from femtetutils import util
from femtetutils import logger as util_logger

from pyfemtet.dispatch_extensions import _get_pid
from pyfemtet.core import SolveError
from pyfemtet._message.messages import encoding
from pyfemtet.logger import get_module_logger


logger = get_module_logger('opt.fem.ParametricIF', __name__)

util_logger.setLevel(logging.ERROR)


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
    #   消さなくても解析はできるが
    #   エラーハンドリングのため
    csv_paths = _get_csv_paths(Femtet)
    for path in csv_paths:
        if os.path.exists(path):
            os.remove(path)

    # 後で使う
    csv_path, table_csv_path = csv_paths

    # load dll and set target femtet
    dll = _get_dll_with_set_femtet(Femtet)

    # reset existing sweep table
    dll.ClearPrmSweepTable.restype = ctypes.c_bool
    succeed = dll.ClearPrmSweepTable()
    if not succeed:
        logger.error('Failed to remove existing sweep table!')
        return False

    # solve
    dll.PrmCalcExecute.restype = ctypes.c_bool
    succeed = dll.PrmCalcExecute()
    if not succeed:
        logger.error('Failed to solve!')
        return False

    # Check post-processing error
    #   現時点では table csv に index の情報がないので、
    #   エラーがどの番号のものかわからない。
    #   ただし、エラーがそのまま出力されるよりマシなので
    #   安全目に引っ掛けることにする

    # csv が生成されているか
    start = time()
    while not os.path.exists(csv_path):
        # solve が succeeded であるにもかかわらず
        # 3 秒経過しても csv が存在しないのはおかしい
        if time() - start > 3.:
            raise RuntimeError('Internal Error! The result csv of Parametric Analysis is not created.')
        sleep(0.25)

    # csv は存在するが、Femtet が古いと
    # table は生成されない
    if not os.path.exists(table_csv_path):
        warnings.warn('テーブル形式 csv が生成されていないため、'
                      '結果出力エラーチェックが行われません。'
                      'そのため、結果出力にエラーがある場合は'
                      '目的関数が 0 と記録される場合があります。'
                      '結果出力エラーチェック機能を利用するためには、'
                      'Femtet を最新バージョンにアップデートして'
                      'ください。')
        return True  # 最適化自体は OK とする

    # get df and error check
    df = pd.read_csv(table_csv_path, encoding=encoding)
    if 'エラー' in df.columns:
        error_column_data = df['エラー']
    elif 'error' in df.columns:
        error_column_data = df['error']
    else:
        raise RuntimeError('Internal Error! Error message column not found in table csv.')

    # 全部が空欄でないならエラーあり
    if not np.all(error_column_data.isna().values):
        logger.error('Outputs of Parametric Analysis contain some errors!')
        return False

    return succeed  # 成功した場合はTRUE、失敗した場合はFALSEを返す


def _get_csv_paths(Femtet):

    # 結果フォルダを取得
    path: str = Femtet.Project
    res_dir_path = path.removesuffix('.femprj') + '.Results'

    # csv を取得
    model_name = Femtet.AnalysisModelName
    csv_path = os.path.join(res_dir_path, f'{model_name}.csv')
    table_csv_path = os.path.join(res_dir_path, f'{model_name}_table.csv')

    return csv_path, table_csv_path


if __name__ == '__main__':
    from win32com.client import Dispatch
    g_Femtet = Dispatch('FemtetMacro.Femtet')
    g_dll = _get_dll_with_set_femtet(g_Femtet)

    # solve
    g_succeeded = solve_via_parametric_dll(g_Femtet)
    if not g_succeeded:
        g_dll.GetLastErrorMsg.restype = ctypes.c_char_p  # or wintypes.LPCSTR
        g_error_msg: bytes = g_dll.GetLastErrorMsg()
        g_error_msg: str = g_error_msg.decode(encoding='932')

    # 結果取得：内部的にはエラーになっているはず
    g_parametric_result_index = 1
    g_dll = _get_dll_with_set_femtet(g_Femtet)
    g_dll.GetPrmResult.restype = ctypes.c_double  # 複素数やベクトルの場合は実部や第一成分しか取らない PIF の仕様
    g_output = g_dll.GetPrmResult(g_parametric_result_index)

    # ... だが、下記のコードでそれは出てこない。
    # 値が実際に 0 である場合と切り分けられないので、
    # csv を見てエラーがあるかどうか判断せざるを得ない。
    g_dll.GetLastErrorMsg.restype = ctypes.c_char_p  # or wintypes.LPCSTR
    g_error_msg: bytes = g_dll.GetLastErrorMsg()
    g_error_msg: str = g_error_msg.decode(encoding='932')
