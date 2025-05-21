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
from pyfemtet.opt.interface._base_interface import COMInterface
from pyfemtet._i18n import _
from pyfemtet.opt.problem.variable_manager import SupportedVariableTypes
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


# noinspection PyPep8Naming
class SolidworksInterface(COMInterface):

    swApp: CDispatch
    com_members = {'swApp': 'SLDWORKS.Application'}
    _access_sw_lock_name = 'access_sw'

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
        logger.info(_(
            en_message='Connecting to Solidworks...',
            jp_message='Solidworks に接続しています...'
        ))
        try:
            self.swApp = DispatchEx('SLDWORKS.Application')
        except com_error:
            raise Exception(_(
                en_message='Failed to instantiate Solidworks. '
                           'Please check installation and enabling macro.',
                jp_message='Solidworks のインスタンス化に失敗しました。'
                           'Solidworks がインストールされており、'
                           'Solidworks マクロが有効であることを確認してください。'))
        self.swApp.Visible = self.solidworks_visible

    def _setup_before_parallel(self):
        self._distribute_files([self.sldprt_path])

    def _setup_after_parallel(self, opt: AbstractOptimizer = None):

        # get suffix
        suffix = self._get_worker_index_from_optimizer(opt)

        # rename and get worker path
        self.sldprt_path = self._rename_and_get_path_on_worker_space(
            self._original_sldprt_path,
            suffix,
        )

        # connect solidworks
        CoInitialize()
        with Lock(self._access_sw_lock_name):
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
        with Lock(self._access_sw_lock_name):

            sleep(0.2)

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
        with Lock(self._access_sw_lock_name):

            sleep(0.2)

            # ===== model を取得 =====
            swModel = self.swModel

            # モデル再構築
            result = swModel.EditRebuild3  # モデル再構築
            if not result:
                raise ModelError(_(
                    en_message='Failed to update the model on Solidworks.',
                    jp_message='Solidworks モデルの更新に失敗しました。'
                ))

    def close(self):
        if not hasattr(self, 'swApp'):
            return

        if self.swApp is None:
            return

        with Lock(self._access_sw_lock_name):
            model_name = os.path.basename(self.sldprt_path)
            logger.info(_(
                en_message='Closing {model_name} ...',
                jp_message='モデル {model_name} を閉じています...',
                model_name=model_name,
            ))

            # 最後の Doc ならばプロセスを落とす仕様？
            self.swApp.QuitDoc(os.path.basename(self.sldprt_path))
            # logger.info(Msg.F_SW_MODEL_CLOSED(model_name))
            logger.info(_(
                en_message='Successfully closed {model_name}.',
                jp_message='モデル {model_name} を閉じました。',
                model_name=model_name,
            ))
            sleep(3)


# noinspection PyPep8Naming
def _get_model_by_basename(swApp, basename):
    swModel = swApp.ActivateDoc(basename)
    if swModel is None:
        raise FileNotOpenedError(f'Model {basename} is not opened.')
    return swModel


def _get_name_from_equation(equation: str):
    pattern = r'^\s*"(.+?)"\s*$'
    matched = re.match(pattern, equation.split('=')[0])
    if matched:
        return matched.group(1)
    else:
        return None
