from __future__ import annotations

from time import sleep
from typing import TYPE_CHECKING

import os
import re

from win32com.client import DispatchEx, CDispatch
# noinspection PyUnresolvedReferences
from pythoncom import CoInitialize, CoUninitialize, com_error

from pyfemtet._util.dask_util import *
from pyfemtet.opt.exceptions import *
from pyfemtet.opt.interface.interface import COMInterface
from pyfemtet._i18n import Msg
from pyfemtet.opt.variable_manager import SupportedVariableTypes
from pyfemtet.logger import get_module_logger

if TYPE_CHECKING:
    from pyfemtet.opt.optimizer import AbstractOptimizer


logger = get_module_logger('opt.interface', False)


# 定数の宣言
swThisConfiguration = 1  # https://help.solidworks.com/2023/english/api/swconst/SOLIDWORKS.Interop.swconst~SOLIDWORKS.Interop.swconst.swInConfigurationOpts_e.html
swAllConfiguration = 2
swSpecifyConfiguration = 3  # use with ConfigName argument
swSaveAsCurrentVersion = 0
swSaveAsOptions_Copy = 2  #
swSaveAsOptions_Silent = 1  # https://help.solidworks.com/2021/english/api/swconst/solidworks.interop.swconst~solidworks.interop.swconst.swsaveasoptions_e.html
swSaveWithReferencesOptions_None = 0  # https://help-solidworks-com.translate.goog/2023/english/api/swconst/SolidWorks.Interop.swconst~SolidWorks.Interop.swconst.swSaveWithReferencesOptions_e.html?_x_tr_sl=auto&_x_tr_tl=ja&_x_tr_hl=ja&_x_tr_pto=wapp
swDocPART = 1  # https://help.solidworks.com/2023/english/api/swconst/SOLIDWORKS.Interop.swconst~SOLIDWORKS.Interop.swconst.swDocumentTypes_e.html


class FileNotOpenedError(Exception):
    pass


class SolidworksInterface(COMInterface):

    swApp: CDispatch
    com_members = {'swApp': 'SLDWORKS.Application'}

    def __init__(
            self,
            sldprt_path,
            close_solidworks_on_terminate=False,
            visible=True,
    ):
        self.sldprt_path = os.path.abspath(sldprt_path)
        self.quit_solidworks_on_terminate = close_solidworks_on_terminate
        self.solidworks_visible = visible

        assert os.path.isfile(self.sldprt_path)
        self._original_sldprt_path = self.sldprt_path

    def connect_sw(self):
        logger.info('Solidworks を起動、または接続しています...')
        self.swApp = DispatchEx('SLDWORKS.Application')
        self.swApp.Visible = self.solidworks_visible

    def _setup_before_parallel(self):
        self._distribute_files([self.sldprt_path])

    def _setup_after_parallel(self, opt: AbstractOptimizer = None):

        # get suffix
        suffix = self._get_worker_index_from_optimizer(opt)

        # rename and get worker path
        self.sldprt_path = self._rename_and_get_path_on_worker_space(
            self.sldprt_path,
            suffix,
        )

        # connect solidworks
        CoInitialize()
        self.connect_sw()

        # open it
        self.swApp.OpenDoc(self.sldprt_path, swDocPART)

    @property
    def swModel(self) -> CDispatch:
        return _get_model_by_basename(self.swApp, os.path.basename(self.sldprt_path))

    def update(self) -> None:
        raise NotImplementedError

    def update_parameter(self, x: dict[str, SupportedVariableTypes]) -> None:

        COMInterface.update_parameter(self, x)

        # sw はプロセスが一つなので Lock
        with Lock('update-sw-model'):

            # ===== model を取得 =====
            swModel = self.swModel

            # ===== equation manager を取得 =====
            swEqnMgr = swModel.GetEquationMgr
            nEquation = swEqnMgr.GetCount

            # プロパティを退避
            buffer_aso = swEqnMgr.AutomaticSolveOrder
            buffer_ar = swEqnMgr.AutomaticRebuild
            swEqnMgr.AutomaticSolveOrder = False
            swEqnMgr.AutomaticRebuild = False

            # 値を更新
            for i in range(nEquation):
                # name, equation の取得
                eq = swEqnMgr.Equation(i)
                prm_name = _get_name_from_equation(eq)
                # 対象なら処理
                if prm_name in self.current_prm_values:
                    new_equation = f'"{prm_name}" = {self.current_prm_values[prm_name]}'
                    swEqnMgr.Equation(i, new_equation)

            # 式の計算
            # noinspection PyStatementEffect
            swEqnMgr.EvaluateAll  # always returns -1

            # プロパティをもとに戻す
            swEqnMgr.AutomaticSolveOrder = buffer_aso
            swEqnMgr.AutomaticRebuild = buffer_ar

    def update_model(self):
        """Update .sldprt"""

        # sw はプロセスが一つなので Lock
        with Lock('update-sw-model'):

            # ===== model を取得 =====
            swModel = self.swModel

            # モデル再構築
            result = swModel.EditRebuild3  # モデル再構築
            if not result:
                raise ModelError(Msg.ERR_UPDATE_SOLIDWORKS_MODEL_FAILED)

    def close(self):
        # TODO: 他の worker の終了を待つ手段がないので実施保留
        # if self.quit_solidworks_on_terminate:
        #
        #     # メインプロセスでのみ終了する
        #     if get_worker() is None:
        #
        #         # 他の worker の終了を待つ
        #
        #         self.swApp.ExitApp()
        logger.info('Solidworks モデルを閉じています...')
        self.swApp.CloseDoc(os.path.basename(self.sldprt_path))
        # self.swApp.QuitDoc(os.path.basename(self.sldprt_path))  # COM の動作が変になる
        logger.info('Solidworks モデルを閉じました。')

    def __del__(self):
        # FIXME: テスト実装
        if self.quit_solidworks_on_terminate:
            logger.info('Solidworks を終了しています...')
            self.swApp.CloseAllDocuments(True)
            self.swApp.ExitApp()
            sleep(3)
            logger.info('Solidworks を終了しました。')


def _get_model_by_basename(swApp, basename):
    swModel = swApp.GetFirstDocument
    while swModel is not None:
        pathname = swModel.GetPathName
        if os.path.basename(pathname) == basename:
            swApp.OpenDoc(pathname, 1)
            swModel_ = swApp.ActiveDoc
            return swModel_
        else:
            swModel = swModel.GetNext
    raise FileNotOpenedError(f'Model {basename} is not opened.')


def _get_name_from_equation(equation: str):
    pattern = r'^\s*"(.+?)"\s*$'
    matched = re.match(pattern, equation.split('=')[0])
    if matched:
        return matched.group(1)
    else:
        return None
