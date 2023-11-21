from PyFemtet.opt._SW_Femtet import SW_Femtet



FEM = SW_Femtet(r'C:\Users\mm11592\Documents\myFiles2\working\PyFemtetOpt2\PyFemtetPrj\tests\SWFemtetWithProcessMonitor\SWTEST.SLDPRT')

# FEM.initialize_parameter()
FEM.update_parameter('A_y', 5) # OK
FEM.update_parameter('D1@ﾎﾞｽ - 押し出し1', 20) # OK

import pandas as pd
df = pd.DataFrame([['A_x', 5], ['A_y', 5]], columns=['name', 'value'])

FEM.run(df)

del FEM









# from win32com.client import Dispatch, DispatchEx

# import re



# def get_parameter_name_from_equation(strEquation):
#     pattern = r'^"(.+)"$'
#     match = re.match(pattern, strEquation)
#     if match:
#         return match.group(1)
#     else:
#         return None


# # 定数の宣言
# swThisConfiguration = 1 # https://help.solidworks.com/2023/english/api/swconst/SOLIDWORKS.Interop.swconst~SOLIDWORKS.Interop.swconst.swInConfigurationOpts_e.html
# swAllConfiguration = 2
# swSpecifyConfiguration = 3 # use with ConfigName argument
# swSaveAsCurrentVersion = 0 
# swSaveAsOptions_Copy = 2 # 
# swSaveAsOptions_Silent = 1 # https://help.solidworks.com/2021/english/api/swconst/solidworks.interop.swconst~solidworks.interop.swconst.swsaveasoptions_e.html
# swSaveWithReferencesOptions_None = 0 # https://help-solidworks-com.translate.goog/2023/english/api/swconst/SolidWorks.Interop.swconst~SolidWorks.Interop.swconst.swSaveWithReferencesOptions_e.html?_x_tr_sl=auto&_x_tr_tl=ja&_x_tr_hl=ja&_x_tr_pto=wapp
# swDocPART = 1 # https://help.solidworks.com/2023/english/api/swconst/SOLIDWORKS.Interop.swconst~SOLIDWORKS.Interop.swconst.swDocumentTypes_e.html


# # dispatch swApp
# swApp = DispatchEx('SLDWORKS.Application')
# swApp.Visible = True

# # open model
# filepath = r'C:\Users\mm11592\Documents\myFiles2\working\PyFemtetOpt2\PyFemtetPrj\tests\SWFemtetWithProcessMonitor\SWTEST.SLDPRT'
# swApp.OpenDoc(filepath, swDocPART)
# swModel = swApp.ActiveDoc

# # Global Variables
# name = 'B_y'
# new_value = 40

# swEqnMgr = swModel.GetEquationMgr
# swEqnMgr.AutomaticSolveOrder = True
# swEqnMgr.AutomaticRebuild = True
# nEquation = swEqnMgr.GetCount
# for i in range(nEquation):

#     # This method modifies only equations added using IEquationMgr::Add3.
#     # error_code = swEqnMgr.SetEquationAndConfigurationOption(
#     #     i,
#     #     f'"{name}"={new_value}',
#     #     swAllConfiguration,
#     #   # ConfigNames        
#     #     )
#     # if error_code!=1:
#     #     raise Exception(f'failed to update {name}')

#     # 左辺のうち、最初と最後の「"」に挟まれた内容が変数
#     target_name = get_parameter_name_from_equation(swEqnMgr.Equation(i).split('=')[0])
#     target_equation = swEqnMgr.Equation(i)

#     # delete
#     error_code = swEqnMgr.Delete(i)
#     if error_code==-1:
#         raise Exception(f'failed to delete {name}')

#     # add
#     if name==target_name:
#         new_equation = f'"{name}"={new_value}'
#     else:
#         new_equation = target_equation
#     # configuration がないと Add3 が動かない。
#     # configuration がある場合も Add2 で動作はする。
#     new_index = swEqnMgr.Add2(
#         i,
#         new_equation,
#         False
#         )
#     # new_index = swEqnMgr.Add3(
#     #     i,
#     #     new_equation,
#     #     False,
#     #     swAllConfiguration,
#     #     # ConfigNames        
#     #     )
#     if new_index==-1:
#         raise Exception(f'failed to update {target_name}')
    
# swEqnMgr.EvaluateAll
# swModel.EditRebuild3 # モデル再構築





# # Dimensions
# name = 'D1@ｽｹｯﾁ1' # @以降まで必要なのはとても面倒だがエクスポート機能でなんとかしてもらおう
# new_value = 20

# swParameter = swModel.Parameter(name)
# if swParameter is None:
#     raise Exception(f'no such parameter: {name}')
# # swParameter.SystemValue # 0.01 # [m]
# # swParameter.Value # 10 [mm]
# # swParameter.GetValue3(swThisConfiguration)
# error_code = swParameter.SetValue3(
#     new_value,
#     swThisConfiguration,
#     # Config_count=1,
#     # Config_names=
#     )
# if error_code>0:
#     raise Exception('error!')

# swModel.EditRebuild3 # モデル再構築

# # export parasolid
# swModel.SaveAs(path_x_t)

# # quit
# swApp.QuitDoc(Name)
# swApp.ExitApp()


# def main():
    
#     # Femtet の準備
#     Femtet = Dispatch('FemtetMacro.Femtet')
    
#     # インポートコマンドでインポートされた x_t の所在を確認
#     path_x_t = Femtet.Gaudi.LastXTPath
    
#     # export parasolid from SW to path_x_t
#     ...
    
#     # 履歴再実行（モデルの変更が反映される）
#     Femtet.Gaudi.ReExecute()
    
#     # mesh, solve
#     Femtet.Gaudi.mesh()
#     Femtet.solve()
    
