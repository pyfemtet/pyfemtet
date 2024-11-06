import os
import re
from time import sleep, time

import pandas as pd
from dask.distributed import get_worker

from win32com.client import DispatchEx
from pythoncom import CoInitialize, CoUninitialize

from pyfemtet.core import ModelError
from pyfemtet.opt.interface import FemtetInterface, logger
from pyfemtet._message import Msg


class FemtetWithSolidworksInterface(FemtetInterface):
    """Control Femtet and Solidworks.

    Using this class, you can import CAD files created
    in Solidworks through the Parasolid format into a
    Femtet project. It allows you to pass design
    variables to Solidworks, update the model, and
    perform analysis using the updated model in Femtet.


    Args:
        sldprt_path (str):
            The path to .sldprt file containing the
            CAD data from which the import is made.
        **kwargs:
            For other arguments, please refer to the
            :class:`FemtetInterface` class.

    """


    # 定数の宣言
    swThisConfiguration = 1  # https://help.solidworks.com/2023/english/api/swconst/SOLIDWORKS.Interop.swconst~SOLIDWORKS.Interop.swconst.swInConfigurationOpts_e.html
    swAllConfiguration = 2
    swSpecifyConfiguration = 3  # use with ConfigName argument
    swSaveAsCurrentVersion = 0
    swSaveAsOptions_Copy = 2  #
    swSaveAsOptions_Silent = 1  # https://help.solidworks.com/2021/english/api/swconst/solidworks.interop.swconst~solidworks.interop.swconst.swsaveasoptions_e.html
    swSaveWithReferencesOptions_None = 0  # https://help-solidworks-com.translate.goog/2023/english/api/swconst/SolidWorks.Interop.swconst~SolidWorks.Interop.swconst.swSaveWithReferencesOptions_e.html?_x_tr_sl=auto&_x_tr_tl=ja&_x_tr_hl=ja&_x_tr_pto=wapp
    swDocPART = 1  # https://help.solidworks.com/2023/english/api/swconst/SOLIDWORKS.Interop.swconst~SOLIDWORKS.Interop.swconst.swDocumentTypes_e.html

    def __init__(
            self,
            sldprt_path,
            **kwargs
    ):
        # 引数の処理
        # dask サブプロセスのときは space 直下の sldprt_path を参照する
        try:
            worker = get_worker()
            space = worker.local_directory
            self.sldprt_path = os.path.join(space, os.path.basename(sldprt_path))
        except ValueError:  # get_worker に失敗した場合
            self.sldprt_path = os.path.abspath(sldprt_path)

        # FemtetInterface の設定 (femprj_path, model_name の更新など)
        # + restore 情報の上書き
        super().__init__(
            sldprt_path=self.sldprt_path,
            **kwargs
        )

    def initialize_sldworks_connection(self):
        # SolidWorks を捕まえ、ファイルを開く
        self.swApp = DispatchEx('SLDWORKS.Application')
        self.swApp.Visible = True

        # open model
        self.swApp.OpenDoc(self.sldprt_path, self.swDocPART)
        self.swModel = self.swApp.ActiveDoc
        self.swEqnMgr = self.swModel.GetEquationMgr
        self.nEquation = self.swEqnMgr.GetCount

    def check_param_value(self, param_name):
        """Override FemtetInterface.check_param_value().

        Do nothing because the parameter can be registered
        to not only .femprj but also .SLDPRT.

        """
        pass

    def _setup_before_parallel(self, client):
        client.upload_file(
            self.kwargs['sldprt_path'],
            False
        )
        super()._setup_before_parallel(client)

    def _setup_after_parallel(self):
        CoInitialize()
        self.initialize_sldworks_connection()

    def update_model(self, parameters: pd.DataFrame, with_warning=False):
        """Update .x_t"""

        self.parameters = parameters.copy()

        # Femtet が参照している x_t パスを取得する
        x_t_path = self.Femtet.Gaudi.LastXTPath

        # 前のが存在するならば消しておく
        if os.path.isfile(x_t_path):
            os.remove(x_t_path)

        # solidworks のモデルの更新
        self.update_sw_model(parameters)

        # export as x_t
        self.swModel.SaveAs(x_t_path)

        # 30 秒待っても x_t ができてなければエラー(COM なので)
        timeout = 30
        start = time()
        while True:
            if os.path.isfile(x_t_path):
                break
            if time() - start > timeout:
                raise ModelError(Msg.ERR_MODEL_UPDATE_FAILED)
            sleep(1)

        # モデルの再インポート
        self._call_femtet_api(
            self.Femtet.Gaudi.ReExecute,
            False,
            ModelError,  # 生きてるのに失敗した場合
            error_message=Msg.ERR_RE_EXECUTE_MODEL_FAILED,
            is_Gaudi_method=True,
        )

        # 処理を確定
        self._call_femtet_api(
            self.Femtet.Redraw,
            False,  # 戻り値は常に None なのでこの変数に意味はなく None 以外なら何でもいい
            ModelError,  # 生きてるのに失敗した場合
            error_message=Msg.ERR_MODEL_REDRAW_FAILED,
            is_Gaudi_method=True,
        )

        # femprj モデルの変数も更新
        super().update_model(parameters)

    def update_sw_model(self, parameters: pd.DataFrame):
        """Update .sldprt"""
        # df を dict に変換
        user_param_dict = {}
        for i, row in parameters.iterrows():
            user_param_dict[row['name']] = row['value']

        # プロパティを退避
        buffer_aso = self.swEqnMgr.AutomaticSolveOrder
        buffer_ar = self.swEqnMgr.AutomaticRebuild
        self.swEqnMgr.AutomaticSolveOrder = False
        self.swEqnMgr.AutomaticRebuild = False

        for i in range(self.nEquation):
            # name, equation の取得
            current_equation = self.swEqnMgr.Equation(i)
            current_name = self._get_name_from_equation(current_equation)
            # 対象なら処理
            if current_name in list(user_param_dict.keys()):
                new_equation = f'"{current_name}" = {user_param_dict[current_name]}'
                self.swEqnMgr.Equation(i, new_equation)

        # 式の計算
        # noinspection PyStatementEffect
        self.swEqnMgr.EvaluateAll  # always returns -1

        # プロパティをもとに戻す
        self.swEqnMgr.AutomaticSolveOrder = buffer_aso
        self.swEqnMgr.AutomaticRebuild = buffer_ar

        # 更新する（ここで失敗はしうる）
        result = self.swModel.EditRebuild3  # モデル再構築
        if not result:
            raise ModelError(Msg.ERR_UPDATE_SOLIDWORKS_MODEL_FAILED)

    def _get_name_from_equation(self, equation: str):
        pattern = r'^\s*"(.+?)"\s*$'
        matched = re.match(pattern, equation.split('=')[0])
        if matched:
            return matched.group(1)
        else:
            return None
