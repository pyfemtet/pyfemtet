import re
import os
from time import sleep

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

    def __init__(self, path_sldprt, strategy=None):
        # 引数の処理
        self.path_sldprt = os.path.abspath(path_sldprt)
        
        # SolidWroks を捕まえ、ファイルを開く
        self.swApp = DispatchEx('SLDWORKS.Application')
        self.swApp.Visible = True
        # open model
        self.swApp.OpenDoc(self.path_sldprt, swDocPART)
        self.swModel = self.swApp.ActiveDoc
        
        # Femtet を捕まえる
        if strategy is None:
            super().setFemtet()
        else:
            super().setFemtet(strategy)

    def __del__(self):
        self.swApp.ExitApp()


    def run(self, df):
        # Femtet が参照しているパスの x_t をアップデートする
        self.update_model_via_SW(df)
        super().run(df)
    
    def update_model_via_SW(self, df):
        # SW を使って parasolid を作る
        
        # Femtet が参照している x_t パスを取得する
        path_x_t = self.Femtet.Gaudi.LastXTPath
        
        # 先にそれを消しておく
        if os.path.isfile(path_x_t):
            os.remove(path_x_t)

        # 変数ひとつひとつ処理していく（エラー箇所同定のため）
        # self.initialize_parameter()
        for i, row in df.iterrows():
            name = row['name']
            value = row['value']
            self.update_parameter(name, value)

        # export parasolid
        self.swModel.SaveAs(path_x_t)
        
        sleep(1)
        print(os.path.isfile(path_x_t))

        # ReExecute Femtet History
        self.Femtet.Gaudi.ReExecute()

    # def initialize_parameter(self):
    #     # evaluate とか rebuild がマクロ経由の関係式しか見ないらしく
    #     # 一度すべての関係式を消して作り直す
    #     swEqnMgr = self.swModel.GetEquationMgr
    #     swEqnMgr.AutomaticSolveOrder = True
    #     swEqnMgr.AutomaticRebuild = True
    #     nEquation = swEqnMgr.GetCount
    #     for i in range(nEquation):
    #         existing_equation:str = swEqnMgr.Equation(i)
    #         name:str = self.get_parameter_name(existing_equation)
        
    #         # delete
    #         error_code = swEqnMgr.Delete(i)
    #         if error_code==-1:
    #             raise SolidWorksAutomationError(f'failed to initialize/delete {name} in SW')
        
    #         # add
    #         added_index = swEqnMgr.Add2(
    #             i,
    #             existing_equation,
    #             False
    #             )
    #         if added_index==-1:
    #             raise SolidWorksAutomationError(f'failed to initialize/update {name} in SW')
    #     swEqnMgr.EvaluateAll
        
    def update_parameter(self, name, value):
        # global parameter
        status = self.update_parameter_as_global_parameter(name, value)
        if status==0:
            # dimension
            status2 = self.update_parameter_as_dimension(name, value)
            if status2==0:
                raise Exception(f'no such parameter in SW: "{name}"')

    def update_parameter_as_dimension(self, name, value):
        status = 0 # 0：処理した、1：処理なかった
        # name = 'D1@ｽｹｯﾁ1' # @以降まで必要、エクスポート機能を活用のこと（その際、「ファイルへリンク」を外すこと。）

        # 寸法の取得       
        swParameter = self.swModel.Parameter(name)
        if swParameter is None:
            return status

        # 寸法に値を更新
        error_code = swParameter.SetValue3(value, swThisConfiguration)
        from .core import ModelError
        if error_code>0:
            raise ModelError(f'変数の更新に失敗しました：変数名{name}, 値{value}')
        status = 1
        
        result = self.swModel.EditRebuild3 # モデル再構築
        if result==False:
            raise ModelError(f'モデル再構築に失敗しました：変数名{name}, 値{value}')
        
        return status
        
        
            

    def update_parameter_as_global_parameter(self, name, value):
        status = 0 # 0:処理なし、1：処理した
        swEqnMgr = self.swModel.GetEquationMgr
        swEqnMgr.AutomaticSolveOrder = True
        swEqnMgr.AutomaticRebuild = True
        nEquation = swEqnMgr.GetCount
        # evaluate とか rebuild がマクロ経由の関係式しか見ないらしく
        # 一度すべての関係式を消して作り直す
        for i in range(nEquation):
            existing_equation = swEqnMgr.Equation(i)
            existing_name = self.get_parameter_name(existing_equation)
            
            # 対象であれば式を更新する
            if name==existing_name:
                new_equation = f'"{existing_name}"={value}'
                status = 1
            else:
                new_equation = existing_equation
        
            # delete
            error_code = swEqnMgr.Delete(i)
            if error_code==-1:
                raise Exception(f'failed to delete {existing_name} in SW')
        
            # add
            added_index = swEqnMgr.Add2(
                i,
                new_equation,
                False
                )
            if added_index==-1:
                raise Exception(f'failed to update {existing_name} in SW')
 
        
        result = swEqnMgr.EvaluateAll
        # if result == -1:
        #     from .core import ModelError
        #     raise ModelError(f'関係式の更新に失敗しました：変数名{existing_name}, 値{value}')
        
        result2 = self.swModel.EditRebuild3 # OK, but...What's True/False?
        if result2==False:
            from .core import ModelError
            # raise ModelError(f'モデル再構築に失敗しました：変数名{name}, 値{value}')

        return status
        
    def get_parameter_name(self, equation:str):
        pattern = r'^\s*"(.+?)"\s*$'
        matched = re.match(pattern, equation.split('=')[0])
        if matched:
            return matched.group(1)
        else:
            return None        
