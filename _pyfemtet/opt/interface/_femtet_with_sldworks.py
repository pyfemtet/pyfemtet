import os
import re
from time import sleep, time

import pandas as pd
from dask.distributed import get_worker, Lock

from win32com.client import DispatchEx
# noinspection PyUnresolvedReferences
from pythoncom import CoInitialize, CoUninitialize, com_error

from pyfemtet.core import ModelError
from pyfemtet.opt.interface._femtet import FemtetInterface
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
            quit_sldworks_on_terminate=False,
            **kwargs
    ):
        # 引数の処理
        self._orig_sldprt_basename = os.path.basename(sldprt_path)

        # # dask サブプロセスのときは space 直下の sldprt_path を参照する
        # try:
        #     worker = get_worker()
        #     space = worker.local_directory
        #     name_ext = os.path.basename(sldprt_path)
        #     name, ext = os.path.splitext(name_ext)
        #     self.sldprt_path = os.path.join(space, name_ext)
        #
        #     # ただし solidworks は 1 プロセスで同名のファイルを開けないので
        #     # 名前を更新する
        #     new_sldprt_path = os.path.join(
        #         space,
        #         f'{name}'
        #         f'_{os.path.basename(space)}'  # worker に対し一意
        #         f'{ext}'  # ext は . を含む
        #     )
        #     os.rename(
        #         self.sldprt_path,
        #         new_sldprt_path
        #     )
        #     self.sldprt_path = new_sldprt_path
        #
        # except ValueError:  # get_worker に失敗した場合
        #     self.sldprt_path = os.path.abspath(sldprt_path)
        self.sldprt_path = os.path.abspath(sldprt_path)
        self.quit_sldworks_on_terminate = quit_sldworks_on_terminate

        # FemtetInterface の設定 (femprj_path, model_name の更新など)
        # + restore 情報の上書き
        super().__init__(
            sldprt_path=self.sldprt_path,
            quit_sldworks_on_terminate=self.quit_sldworks_on_terminate,
            **kwargs
        )

    def initialize_sldworks_connection(self):
        # SolidWorks を捕まえ、ファイルを開く

        self.swApp = DispatchEx('SLDWORKS.Application')
        self.swApp.Visible = True

        # solidworks は単一プロセスなので開くファイルはひとつだけ
        try:
            get_worker()
        except ValueError:
            self.swApp.OpenDoc(self.sldprt_path, self.swDocPART)

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

    def _setup_after_parallel(self, *args, **kwargs):
        CoInitialize()
        self.initialize_sldworks_connection()

    def update_model(self, parameters: pd.DataFrame, with_warning=False):
        """Update .x_t"""

        self.parameters = parameters.copy()

        # Femtet が参照している x_t パスを取得する
        x_t_path = self.Femtet.Gaudi.LastXTPath

        # dask サブプロセスならば競合しないよう保存先を scratch 直下にしておく
        try:
            get_worker()
            x_t_path = os.path.splitext(self.sldprt_path)[0] + '.x_t'

        except ValueError:  # No worker found
            pass

        # 前のが存在するならば消しておく
        if os.path.isfile(x_t_path):
            os.remove(x_t_path)

        # solidworks のモデルの更新
        try:
            with Lock('update-model-sldworks'):
                sleep(0.5)  # 並列処理でクラッシュすることが多かったため試験的に導入
                self.update_sw_model(parameters, x_t_path)

        # femopt を使わない場合
        except RuntimeError:  # <class 'distributed.lock.Lock'> object not properly initialized. ...
            self.update_sw_model(parameters, x_t_path)

        # dask サブプロセスならば LastXTPath を更新する
        try:
            get_worker()
            try:
                self.Femtet.Gaudi.LastXTPath = x_t_path
            except (KeyError, AttributeError, com_error):
                raise RuntimeError('This feature is available from Femtet version 2023.2. Please update Femtet.')

        # dask を使わない場合
        except ValueError:  # No worker found
            pass

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

    def update_sw_model(self, parameters: pd.DataFrame, x_t_path):
        """Update .sldprt"""

        # df を dict に変換
        user_param_dict = {}
        for i, row in parameters.iterrows():
            user_param_dict[row['name']] = row['value']

        while True:

            try:

                # ===== model を取得 =====
                swModel = get_model_by_basename(self.swApp, os.path.basename(self.sldprt_path))

                # ===== equation manager を取得 =====
                swEqnMgr = swModel.GetEquationMgr
                nEquation = swEqnMgr.GetCount

                # プロパティを退避
                buffer_aso = swEqnMgr.AutomaticSolveOrder
                buffer_ar = swEqnMgr.AutomaticRebuild
                swEqnMgr.AutomaticSolveOrder = False
                swEqnMgr.AutomaticRebuild = False

                for i in range(nEquation):
                    # name, equation の取得
                    current_equation = swEqnMgr.Equation(i)
                    current_name = self._get_name_from_equation(current_equation)
                    # 対象なら処理
                    if current_name in list(user_param_dict.keys()):
                        new_equation = f'"{current_name}" = {user_param_dict[current_name]}'
                        swEqnMgr.Equation(i, new_equation)

                # 式の計算
                # noinspection PyStatementEffect
                swEqnMgr.EvaluateAll  # always returns -1

                # プロパティをもとに戻す
                swEqnMgr.AutomaticSolveOrder = buffer_aso
                swEqnMgr.AutomaticRebuild = buffer_ar

                # 更新する（ここで失敗はしうる）
                result = swModel.EditRebuild3  # モデル再構築
                if not result:
                    raise ModelError(Msg.ERR_UPDATE_SOLIDWORKS_MODEL_FAILED)

                # export as x_t
                swModel.SaveAs(x_t_path)

                # 30 秒待っても x_t ができてなければエラー(COM なのでありうる)
                timeout = 30
                start = time()
                while True:
                    if os.path.isfile(x_t_path):
                        break
                    if time() - start > timeout:
                        raise ModelError(Msg.ERR_MODEL_UPDATE_FAILED)
                    sleep(1)

            except AttributeError as e:
                if 'SLDWORKS.Application.' in str(e):
                    # re-launch solidworks
                    self.swApp = DispatchEx('SLDWORKS.Application')
                    self.swApp.Visible = True
                    self.swApp.OpenDoc(self.sldprt_path, self.swDocPART)
                    continue

                else:
                    raise e

            break

    def _get_name_from_equation(self, equation: str):
        pattern = r'^\s*"(.+?)"\s*$'
        matched = re.match(pattern, equation.split('=')[0])
        if matched:
            return matched.group(1)
        else:
            return None

    def quit(self, timeout=1, force=True):
        if self.quit_sldworks_on_terminate:
            try:
                get_worker()
            except ValueError:
                try:
                    self.swApp.ExitApp()
                except AttributeError:
                    pass

        super().quit(timeout, force)


def get_model_by_basename(swApp, basename):
    swModel = swApp.GetFirstDocument
    while swModel is not None:
        pathname = swModel.GetPathName
        if os.path.basename(pathname) == basename:
            from win32com.client import Dispatch
            # swModel_ = swApp.ActivateDoc3(
            #     basename,
            #     False,
            #     1,  # swRebuildOnActivation_e.swDontRebuildActiveDoc,
            #     Dispatch("Scripting.List"),
            # )
            swApp.OpenDoc(pathname, 1)
            swModel_ = swApp.ActiveDoc
            return swModel_
        else:
            swModel = swModel.GetNext
    raise ModuleNotFoundError(f'No model named {basename}')
