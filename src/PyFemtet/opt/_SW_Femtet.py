import re
import os
from time import sleep

import pandas as pd
from win32com.client import DispatchEx, Dispatch

from .core import Femtet

# 定数の宣言
swThisConfiguration = 1 # https://help.solidworks.com/2023/english/api/swconst/SOLIDWORKS.Interop.swconst~SOLIDWORKS.Interop.swconst.swInConfigurationOpts_e.html
swAllConfiguration = 2
swSpecifyConfiguration = 3 # use with ConfigName argument
swSaveAsCurrentVersion = 0 
swSaveAsOptions_Copy = 2 # 
swSaveAsOptions_Silent = 1 # https://help.solidworks.com/2021/english/api/swconst/solidworks.interop.swconst~solidworks.interop.swconst.swsaveasoptions_e.html
swSaveWithReferencesOptions_None = 0 # https://help-solidworks-com.translate.goog/2023/english/api/swconst/SolidWorks.Interop.swconst~SolidWorks.Interop.swconst.swSaveWithReferencesOptions_e.html?_x_tr_sl=auto&_x_tr_tl=ja&_x_tr_hl=ja&_x_tr_pto=wapp
swDocPART = 1 # https://help.solidworks.com/2023/english/api/swconst/SOLIDWORKS.Interop.swconst~SOLIDWORKS.Interop.swconst.swDocumentTypes_e.html

class SolidWorksAutomationError(Exception):
    pass


class SW_Femtet(Femtet):

    def __init__(self, path_sldprt):
        # 引数の処理
        self.path_sldprt = os.path.abspath(path_sldprt)
        
        # SolidWorks を捕まえ、ファイルを開く
        self.swApp = DispatchEx('SLDWORKS.Application')
        self.swApp.Visible = True

        # open model
        self.swApp.OpenDoc(self.path_sldprt, swDocPART)
        self.swModel = self.swApp.ActiveDoc
        self.swEqnMgr = self.swModel.GetEquationMgr
        self.nEquation = self.swEqnMgr.GetCount

        # 引数の保存
        super().__init__(self.path_sldprt)


    def update(self, df:pd.DataFrame) ->None:
        """Update_model_via_SW と run を実行します.

        Parameters
        ----------
        df

        Returns
        -------

        """

        # df を dict に変換
        user_param_dict = {}
        for i, row in df.iterrows():
            user_param_dict[row['name']] = row['value']

        # 変数の存在チェック
        self.check_parameter(user_param_dict)

        # Femtet 解析モデルを更新
        self.update_model_via_SW(user_param_dict)

        # 解析を実行する
        super().run()


    def check_parameter(self, user_param_dict):
        """与えられた変数セットが .sldprt ファイルに存在するか確認します."""
        swEqnMgr = self.swModel.GetEquationMgr

        # 指定された変数が SW に存在するかどうかをチェック
        for user_param_name in user_param_dict.keys():
            is_in_model = False
            # global 変数に存在するかどうか
            nEquation = swEqnMgr.GetCount
            global_param_names = [self.get_name(swEqnMgr.Equation(i)) for i in range(nEquation) if swEqnMgr.GlobalVariable(i)]
            is_in_model = is_in_model or (user_param_name in global_param_names)
            # 寸法に存在するかどうか
            swParam = self.swModel.Parameter(user_param_name)
            is_in_model = is_in_model or (swParam is not None)
            # is_in_model が False ならエラー
            if not is_in_model:
                raise Exception(f'no such parameter in SW:{user_param_name}')


    def update_model_via_SW(self, user_param_dict):
        """SW を使って parasolid を作り, Femtet 解析モデルを更新します."""
        
        # Femtet が参照している x_t パスを取得する
        path_x_t = self.Femtet.Gaudi.LastXTPath
        
        # 先にそれを消しておく
        if os.path.isfile(path_x_t):
            os.remove(path_x_t)

        # sldprt モデルの変数の更新
        self.update_SW_parameter(user_param_dict)

        # export parasolid
        self.swModel.SaveAs(path_x_t)

        # 30 秒待っても parasolid ができてなければエラー(COM なので)
        timeout = 30
        waiting = 0
        while True:
            sleep(1)
            waiting += 1
            if os.path.isfile(path_x_t):
                break
            else:
                if waiting<timeout:
                    continue
                else:
                    # from .core import ModelError
                    # 再構築できていれば保存はできているはず
                    raise SolidWorksAutomationError('parasolid 書き出しに失敗しました')

        # Femtet モデルの更新 by ReExecute Femtet History
        self.Femtet.Gaudi.ReExecute()
        self.Femtet.Redraw()


    def update_SW_parameter(self, user_param_dict):
        # プロパティを退避
        buffer_aso = self.swEqnMgr.AutomaticSolveOrder
        buffer_ar = self.swEqnMgr.AutomaticRebuild
        self.swEqnMgr.AutomaticSolveOrder = False
        self.swEqnMgr.AutomaticRebuild = False
        
        from tqdm import trange
        for i in trange(self.nEquation):
            # name, equation の取得
            current_equation = self.swEqnMgr.Equation(i)
            current_name = self.get_name(current_equation)
            # 対象なら処理
            if current_name in list(user_param_dict.keys()):
                new_equation = f'"{current_name}" = {user_param_dict[current_name]}'
                self.swEqnMgr.Equation(i, new_equation)

        # 式の計算
        self.swEqnMgr.EvaluateAll # always returns -1
        
        # プロパティをもとに戻す
        self.swEqnMgr.AutomaticSolveOrder = buffer_aso
        self.swEqnMgr.AutomaticRebuild = buffer_ar

        # 更新する（ここで失敗はしうる）
        result = self.swModel.EditRebuild3 # モデル再構築
        if result==False:
            from .core import ModelError
            raise ModelError('モデル再構築に失敗しました')
        
    def get_name(self, equation:str):
        pattern = r'^\s*"(.+?)"\s*$'
        matched = re.match(pattern, equation.split('=')[0])
        if matched:
            return matched.group(1)
        else:
            return None        
