import os
import json
from time import sleep
from multiprocessing import Pool
import subprocess

from win32com.client import Dispatch, DispatchEx
import win32process
import win32api
import win32con

import numpy as np

here, me = os.path.split(__file__)
os.chdir(here)

# ユーザーが案件ごとに設定する変数
path_bas = os.path.abspath('TEST_femprj.bas')
#path_x_t = os.path.abspath('TEST.x_t')
path_prt = os.path.abspath('TEST.prt')
# path_femprj = os.path.abspath('TEST.femprj')

# ユーザーが環境ごとに設定する定数
path_macro = r'C:\Program Files\Femtet_Ver2023_64bit_inside\Program\Macro32\FemtetMacro.dll'
path_ref = r'C:\Program Files\Femtet_Ver2023_64bit_inside\Program\Macro32\FemtetRef.xla'

# ユーザーは設定しない定数
path_hub = os.path.abspath('hub.xlsm')
path_journal = os.path.abspath('update_model_parameter.py')



def _f(functions, arguments): # インスタンスメソッドにしたら動かない クラスメソッドにしても動かない。なんでだろう？
    # 新しいプロセスで呼ぶ関数。
    # 新しい Femtet を作って objectives を計算する
    # その後、プロセスは死ぬので Femtet は解放される
    # TODO:この関数の最後で、Femtet を殺していいかどうか検討する
    Femtet = Dispatch('FemtetMacro.Femtet')
    Femtet.Gaudi.Mesh()
    Femtet.Solve()
    Femtet.OpenCurrentResult(True)
    ret = []
    for func, (args, kwargs) in zip(functions, arguments):
        ret.append(func(Femtet, *args, **kwargs))
    return ret

from PyFemtet.opt.core import FEMSystem

class NX_Femtet(FEMSystem):
    def run(self):
        # 使わないけど FEMSystem が実装を求めるためダミーで作成
        pass
    
    def f(self, df, objectives):
        self._update_model(df)
        self._setup_new_Femtet()
        self._run_new_Femtet(objectives)
        return self.objectiveValues
            
    def _update_model(self, df):
        # run_journal を使って prt から x_t を作る
        exe = r'%UGII_BASE_DIR%\NXBIN\run_journal.exe'
        tmp = dict(zip(df.name.values, df.value.values.astype(str)))
        strDict = json.dumps(tmp)
        subprocess.run([exe, path_journal, '-args', path_prt, strDict])
        # prt と同じ名前の x_t ができる
        
        
    #     pass
    
    def _setup_new_Femtet(self):
        # excel 経由で bas を使って x_t から Femtet のセットアップをする
        # その後、excel は殺す
        # excel の立ち上げ
        self.excel = DispatchEx("Excel.Application")
        self.excel.Visible = False
        self.excel.DisplayAlerts = False
        self.excel.Workbooks.Open(path_hub)
        self.wb = self.excel.Workbooks('hub.xlsm')
        # Femtet マクロの導入
        self.excel.Run('Module1.ImportFemtetMacroBas', path_bas)
        self.excel.Run('Module1.FemtetSetup', path_ref)
        self.excel.Run('Module1.FemtetSetup', path_macro)
        # マクロの破棄
        self.excel.Run('Module1.ReleaseFemtetMacroBas')    
        self.excel.Run('Module1.ReleaseFemtetSetup')
        # 保存せずに閉じる
        self.wb.Saved = True
        self.wb.Close()
        # 終了
        self._close_excel_by_force()

    def _run_new_Femtet(self, objectives):
        # 関数を適用する
        with Pool(processes=1) as p:
            functions = [obj.ptrFunc for obj in objectives]
            arguments = [(obj.args, obj.kwargs) for obj in objectives]
            result = p.apply(_f, (functions, arguments))
            self.objectiveValues = result
        
    def _close_excel_by_force(self):
        # プロセス ID の取得
        hwnd = self.excel.Hwnd
        _, p = win32process.GetWindowThreadProcessId(hwnd)
        # force close
        try:
            handle = win32api.OpenProcess(win32con.PROCESS_TERMINATE, 0, p)
            if handle:
                win32api.TerminateProcess(handle, 0)
                win32api.CloseHandle(handle)
        except:
            pass



# #### サンプル関数
# from win32com.client import constants
# def get_flow(Femtet):
#     Gogh = Femtet.Gogh
#     Gogh.Pascal.Vector = constants.PASCAL_VELOCITY_C
#     _, ret = Gogh.SimpleIntegralVectorAtFace_py([2], [0], constants.PART_VEC_Y_PART_C)
#     flow = ret.Real
#     return flow


# if __name__=='__main__':
#     FEMOpt = NX_Femtet()
#     df = None
#     print(FEMOpt.f(df, [get_flow]))
    