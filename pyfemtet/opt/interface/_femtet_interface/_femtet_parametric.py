from __future__ import annotations

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

from pyfemtet._util.process_util import _get_pid
from pyfemtet.opt.exceptions import SolveError
from pyfemtet._i18n import ENCODING, Msg
from pyfemtet.logger import get_module_logger

__all__ = [
    'solve_via_parametric_dll',
    'add_parametric_results_as_objectives'
]


logger = get_module_logger('opt.fem.ParametricIF', False)

util_logger.setLevel(logging.ERROR)

# singleton pattern
_P_CSV: ParametricResultCSVProcessor | None = None


def get_csv_processor(Femtet):
    global _P_CSV
    if _P_CSV is None:
        _P_CSV = ParametricResultCSVProcessor(Femtet)
    return _P_CSV


class ParametricResultCSVProcessor:

    def __init__(self, Femtet):
        self.Femtet = Femtet

    def refresh_csv(self):
        # 存在するならば削除する
        csv_paths = self.get_csv_paths()
        for path in csv_paths:
            if os.path.exists(path):
                os.remove(path)

    def get_csv_paths(self):
        # 結果フォルダを取得
        path: str = self.Femtet.Project
        res_dir_path = path.removesuffix('.femprj') + '.Results'

        # csv を取得
        model_name = self.Femtet.AnalysisModelName
        csv_path = os.path.join(res_dir_path, f'{model_name}.csv')
        table_csv_path = os.path.join(res_dir_path, f'{model_name}_table.csv')

        return csv_path, table_csv_path

    def check_csv_after_succeeded_PrmCalcExecute(self):
        """Parametric Solve の後に呼ぶこと。"""

        csv_path, table_csv_path = self.get_csv_paths()

        # csv が生成されているか
        start = time()
        while not os.path.exists(csv_path):
            # solve が succeeded であるにもかかわらず
            # 数秒経過しても csv が存在しないのはおかしい
            if time() - start > 3.:
                return False
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

        return True

    def is_succeeded(self, parametric_output_index):

        # まず csv 保存が成功しているかどうか。通常あるはず。
        if not self.check_csv_after_succeeded_PrmCalcExecute():
            return False, 'Reason: output csv not found.'

        # 成功しているならば table があるかどうか
        csv_path, table_csv_path = self.get_csv_paths()

        # なければエラーチェックできないので
        # エラーなしとみなす (warning の記載通り)
        if not os.path.exists(table_csv_path):
            return True, None

        # table があれば読み込む
        df = pd.read_csv(table_csv_path, encoding=ENCODING)

        # 結果出力用に行番号を付記する
        df['row_num'] = range(2, len(df) + 2)  # row=0 は header, excel は 1 始まり

        # 「結果出力設定番号」カラムが存在するか
        key = '結果出力設定番号'
        key_en = ' Output setting Number'

        def get_pdf(key_, key_en_):
            if key_ in df.columns:

                # 与えられた output_number に関連する行だけ抜き出し
                # エラーがあるかどうかチェックする
                idx = df[key_] == parametric_output_index + 1
                pdf_ = df[idx]

            elif key_en_ in df.columns:
                idx = df[key_en_] == parametric_output_index + 1
                pdf_ = df[idx]

            # 結果出力設定番号 カラムが存在しない
            else:
                # output_number に関係なくエラーがあればエラーにする
                pdf_ = df

            return pdf_

        pdf = get_pdf(key, key_en)

        # エラーの有無を確認
        key = 'エラー'

        def handle_error(key_):
            if key_ in pdf.columns:
                is_no_error_ = np.all(pdf[key_].isna().values)

                if not is_no_error_:
                    error_message_row_numbers = pdf['row_num'][~pdf[key_].isna()].values.astype(str)
                    error_messages = pdf[key_][~pdf[key_].isna()].values.astype(str)

                    def add_st_or_nd_or_th(n_: int):
                        if n_ == 1:
                            return f'{n_}st'
                        elif n_ == 2:
                            return f'{n_}nd'
                        elif n_ == 3:
                            return f'{n_}rd'
                        else:
                            return f'{n_}th'

                    error_msg_ = f'Error message(s) from {os.path.basename(table_csv_path)}: ' + ', '.join(
                        [f'({add_st_or_nd_or_th(row)} row) {message}'
                         for row, message in zip(error_message_row_numbers, error_messages)])
                else:
                    error_msg_ = None

                return is_no_error_, error_msg_

            else:
                return None, 'NO_KEY'

        is_no_error, error_msg = handle_error(key)

        if error_msg == 'NO_KEY':
            key = ' Error'
            is_no_error, error_msg = handle_error(key)
            if error_msg == 'NO_KEY':
                assert False, ('Internal Error! Parametric Analysis '
                               'output csv has no error column.')

        return is_no_error, error_msg


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


def add_parametric_results_as_objectives(opt, Femtet, indexes, directions) -> bool:
    # load dll and set target femtet
    dll = _get_dll_with_set_femtet(Femtet)

    # get objective names
    dll.GetPrmnResult.restype = ctypes.c_int
    for i, direction in zip(indexes, directions):
        # objective name
        dll.GetPrmResultName.restype = ctypes.c_char_p
        result = dll.GetPrmResultName(i)
        name = result.decode('mbcs')
        # objective value function
        opt.add_objective(name=name, fun=_parametric_objective, direction=direction, args=(i,))
    return True  # ここまで来たら成功


def _parametric_objective(Femtet, parametric_result_index):
    # csv から結果取得エラーの有無を確認する
    # (解析自体は成功していないと objective は呼ばれないはず)
    csv_processor = get_csv_processor(Femtet)
    succeeded, error_msg = csv_processor.is_succeeded(parametric_result_index)
    if not succeeded:
        logger.error(Msg.ERR_PARAMETRIC_CSV_CONTAINS_ERROR)
        logger.error(error_msg)
        raise SolveError

    # load dll and set target femtet
    dll = _get_dll_with_set_femtet(Femtet)
    dll.GetPrmResult.restype = ctypes.c_double  # 複素数の場合は実部しか取らない
    return dll.GetPrmResult(parametric_result_index)


def solve_via_parametric_dll(Femtet) -> bool:
    csv_processor = get_csv_processor(Femtet)

    # remove previous csv if exists
    #   消さなくても解析はできるが
    #   エラーハンドリングのため
    csv_processor.refresh_csv()

    # load dll and set target femtet
    dll = _get_dll_with_set_femtet(Femtet)

    # reset existing sweep table
    dll.ClearPrmSweepTable.restype = ctypes.c_bool
    succeed = dll.ClearPrmSweepTable()
    if not succeed:
        logger.error('Failed to remove existing sweep table!')  # 通常ありえないので error
        return False

    # solve
    dll.PrmCalcExecute.restype = ctypes.c_bool
    succeed = dll.PrmCalcExecute()
    if not succeed:
        logger.warning('Failed to solve!')  # 通常起こりえるので warn
        return False

    # Check post-processing error
    #   現時点では table csv に index の情報がないので、
    #   エラーがどの番号のものかわからない。
    #   ただし、エラーがそのまま出力されるよりマシなので
    #   安全目に引っ掛けることにする
    succeed = csv_processor.check_csv_after_succeeded_PrmCalcExecute()
    if not succeed:
        logger.error('Failed to save parametric result csv!')
        return False  # 通常ありえないので error

    return succeed  # 成功した場合はTRUE、失敗した場合はFALSEを返す


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
