from __future__ import annotations
from typing import TYPE_CHECKING

import os
from time import sleep, time

# noinspection PyUnresolvedReferences
from pywintypes import com_error

from pyfemtet._i18n import _

from pyfemtet.opt.problem.variable_manager import SupportedVariableTypes
from pyfemtet.opt.interface._base_interface import AbstractFEMInterface
from pyfemtet.opt.interface._femtet_interface import FemtetInterface
from pyfemtet.opt.interface._solidworks_interface import SolidworksInterface
from pyfemtet.opt.exceptions import *

if TYPE_CHECKING:
    from pyfemtet.opt.optimizer import AbstractOptimizer


class FemtetWithSolidworksInterface(FemtetInterface, SolidworksInterface, AbstractFEMInterface):

    def __init__(
            self,
            sldprt_path: str,
            femprj_path: str = None,
            model_name: str = None,
            connect_method: str = "auto",
            save_pdt: str = "all",
            strictly_pid_specify: bool = True,
            allow_without_project: bool = False,
            open_result_with_gui: bool = True,
            parametric_output_indexes_use_as_objective: dict[int, str or float] = None,
            always_open_copy=False,
            close_solidworks_on_terminate=False,
            solidworks_visible=True,
    ):
        SolidworksInterface.__init__(
            self,
            sldprt_path=sldprt_path,
            close_solidworks_on_terminate=close_solidworks_on_terminate,
            visible=solidworks_visible,
        )

        FemtetInterface.__init__(
            self,
            femprj_path=femprj_path,
            model_name=model_name,
            connect_method=connect_method,
            save_pdt=save_pdt,
            strictly_pid_specify=strictly_pid_specify,
            allow_without_project=allow_without_project,
            open_result_with_gui=open_result_with_gui,
            parametric_output_indexes_use_as_objective=parametric_output_indexes_use_as_objective,
            always_open_copy=always_open_copy,
        )

        self._warn_if_undefined_variable = False

    def _check_param_and_raise(self, param_name) -> None:
        pass

    def _setup_before_parallel(self):
        SolidworksInterface._setup_before_parallel(self)
        FemtetInterface._setup_before_parallel(self)

    def _setup_after_parallel(self, opt: AbstractOptimizer = None):
        SolidworksInterface._setup_after_parallel(self, opt)
        FemtetInterface._setup_after_parallel(self, opt)

    def update_parameter(self, x: dict[str, SupportedVariableTypes], with_warning=False) -> None:
        SolidworksInterface.update_parameter(self, x)
        FemtetInterface.update_parameter(self, x, with_warning)

    def _export_xt(self, xt_path):

        # 前のが存在するならば消しておく
        if os.path.isfile(xt_path):
            os.remove(xt_path)

        # export as x_t
        self.swModel.SaveAs(xt_path)

        # 30 秒待っても x_t ができてなければエラー(COM なのでありうる)
        timeout = 30
        start = time()
        while True:
            if os.path.isfile(xt_path):
                break
            if time() - start > timeout:
                raise ModelError(_(
                    en_message='Failed to update Solidworks model.',
                    jp_message='Solidworks モデルの更新に失敗しました。',
                ))
            sleep(1)

    def update_model(self):

        # solidworks のモデルの更新
        SolidworksInterface.update_model(self)

        # 競合しないよう保存先を temp にしておく
        worker_space = self._get_worker_space()
        xt_path = os.path.join(worker_space, 'temp.x_t')

        # export parasolid
        self._export_xt(xt_path)

        # LastXTPath を更新する
        try:
            self.Femtet.Gaudi.LastXTPath = xt_path
        except (KeyError, AttributeError, com_error):
            raise RuntimeError('This feature is available from Femtet version 2023.2. Please update Femtet.')

        # update_parameter で変数は更新されているので
        # ここでモデルを完全に再構築できる
        FemtetInterface.update_model(self)

    def close(self, timeout=15, force=True):
        SolidworksInterface.close(self)
        FemtetInterface.close(self, timeout, force)
