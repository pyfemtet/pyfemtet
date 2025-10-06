from __future__ import annotations

from time import sleep
from typing import TYPE_CHECKING, Any
import os

from win32com.client import Dispatch
from pythoncom import com_error, CoInitialize

from pyfemtet._util.dask_util import *
from pyfemtet.opt.exceptions import *
from pyfemtet.opt.interface._base_interface import COMInterface
from pyfemtet._i18n import _
from pyfemtet.opt.problem.problem import *
from pyfemtet.logger import get_module_logger

from pyfemtet._util.solidworks_variable import SolidworksVariableManager, is_assembly, SWVariables

if TYPE_CHECKING:
    from pyfemtet.opt.optimizer import AbstractOptimizer

logger = get_module_logger('opt.interface', False)
asm_logger = get_module_logger('opt.interface.sldasm', False)

# 定数の宣言
swThisConfiguration = 1  # https://help.solidworks.com/2023/english/api/swconst/SOLIDWORKS.Interop.swconst~SOLIDWORKS.Interop.swconst.swInConfigurationOpts_e.html
swAllConfiguration = 2
swSpecifyConfiguration = 3  # use with ConfigName argument
swSaveAsCurrentVersion = 0
swSaveAsOptions_Copy = 2  #
swSaveAsOptions_Silent = 1  # https://help.solidworks.com/2021/english/api/swconst/solidworks.interop.swconst~solidworks.interop.swconst.swsaveasoptions_e.html
swSaveWithReferencesOptions_None = 0  # https://help-solidworks-com.translate.goog/2023/english/api/swconst/SolidWorks.Interop.swconst~SolidWorks.Interop.swconst.swSaveWithReferencesOptions_e.html?_x_tr_sl=auto&_x_tr_tl=ja&_x_tr_hl=ja&_x_tr_pto=wapp
swDocPART = 1  # https://help.solidworks.com/2023/english/api/swconst/SOLIDWORKS.Interop.swconst~SOLIDWORKS.Interop.swconst.swDocumentTypes_e.html
swDocASSEMBLY = 2


class FileNotOpenedError(Exception):
    pass


# noinspection PyPep8Naming
class SolidworksInterface(COMInterface):
    """
    Interface class for interacting with SolidWorks through COM automation.

    This class manages the connection and interaction with SolidWorks using its COM interface.
    It handles initialization, visibility, and clean termination of the SolidWorks application.

    Attributes:
        swApp (CDispatch): The COM dispatch object for SolidWorks application.
        com_members (dict): Mapping of COM member names to their interface strings.
        sldprt_path (str): Absolute path to the SolidWorks part file (.sldprt).
        quit_solidworks_on_terminate (bool): Whether to close SolidWorks upon object destruction.
        solidworks_visible (bool): Whether the SolidWorks application window is visible.

    Args:
        sldprt_path (str): Path to the SolidWorks part file (.sldprt).
        close_solidworks_on_terminate (bool, optional): If True, SolidWorks will close when this object is destroyed. Defaults to False.
        visible (bool, optional): If True, SolidWorks will be started in visible mode. Defaults to True.

    Raises:
        AssertionError: If the specified part file does not exist.
    """

    swApp: Any
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

        if not os.path.isfile(self.sldprt_path):
            raise FileNotFoundError(self.sldprt_path)
        self._original_sldprt_path = self.sldprt_path

    def connect_sw(self):
        logger.info(_(
            en_message='Connecting to Solidworks...',
            jp_message='Solidworks に接続しています...'
        ))
        try:
            self.swApp = Dispatch('SLDWORKS.Application')
        except com_error:
            raise RuntimeError(_(
                en_message='Failed to instantiate Solidworks. '
                           'Please check installation and enabling macro.',
                jp_message='Solidworks のインスタンス化に失敗しました。'
                           'Solidworks がインストールされており、'
                           'Solidworks マクロが有効であることを確認してください。'))
        self.swApp.Visible = self.solidworks_visible

    def _setup_before_parallel(self, scheduler_address=None):
        if not is_assembly(self.sldprt_path):
            self._distribute_files([self.sldprt_path], scheduler_address)

    def _setup_after_parallel(self, opt: AbstractOptimizer):

        # validation
        if is_assembly(self.sldprt_path) and get_worker() is not None:
            # 現在の仕様だと sldprt_path だけが
            # worker_space に保存される。
            # 並列処理に対応するためには
            # すべてのファイルを distribute したうえで
            # 構成部品の置換を実行する必要がある。
            raise RuntimeError(_(
                en_message='Parallel processing is not supported when handling assembly parts with SolidworksInterface.',
                jp_message='SolidworksInterfaceでアセンブリパーツを対象とする場合、並列処理はサポートされていません。'
            ))

        if not is_assembly(self.sldprt_path):
            # get suffix
            suffix = self._get_file_suffix(opt)

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
            if is_assembly(self.sldprt_path):
                self.swApp.OpenDoc(self.sldprt_path, swDocASSEMBLY)
            else:
                self.swApp.OpenDoc(self.sldprt_path, swDocPART)

    @property
    def swModel(self):
        return _get_model_by_basename(self.swApp, os.path.basename(self.sldprt_path))

    def update(self) -> None:
        raise NotImplementedError

    def update_parameter(self, x: TrialInput) -> None:

        COMInterface.update_parameter(self, x)

        # sw はプロセスが一つなので Lock
        with Lock(self._access_sw_lock_name):
            sleep(0.2)
            swModel = self.swModel
            mgr = SolidworksVariableManager(asm_logger)
            sw_variables: SWVariables = {
                name: str(param.value) + param.properties.get('unit', '')
                for name, param in
                self.current_prm_values.items()
                if isinstance(param.value, float | int)
            }
            mgr.update_global_variables_recourse(swModel, sw_variables)

    def update_model(self):
        """Update .sldprt"""

        # sw はプロセスが一つなので Lock
        with Lock(self._access_sw_lock_name):
            sleep(0.2)

            swModel = self.swModel
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
