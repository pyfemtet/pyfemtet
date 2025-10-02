from __future__ import annotations

from time import sleep
from typing import TYPE_CHECKING, Any
import os

from win32com.client import Dispatch, DispatchEx
from pythoncom import com_error, CoInitialize

from pyfemtet._util.dask_util import *
from pyfemtet.opt.exceptions import *
from pyfemtet.opt.interface._base_interface import COMInterface
from pyfemtet._i18n import _
from pyfemtet.opt.problem.problem import *
from pyfemtet.logger import get_module_logger

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


class EquationContext:
    def __init__(self, swModel) -> None:
        self.swModel = swModel
        self.swEqnMgr = None

    def __enter__(self):
        # プロパティを退避
        self.swEqnMgr = self.swModel.GetEquationMgr
        self.buffer_aso = self.swEqnMgr.AutomaticSolveOrder
        self.buffer_ar = self.swEqnMgr.AutomaticRebuild
        self.swEqnMgr.AutomaticSolveOrder = False
        self.swEqnMgr.AutomaticRebuild = False
        return self.swEqnMgr

    def __exit__(self, exc_type, exc_val, exc_tb):
        # プロパティをもとに戻す
        assert self.swEqnMgr is not None
        self.swEqnMgr.AutomaticSolveOrder = self.buffer_aso
        self.swEqnMgr.AutomaticRebuild = self.buffer_ar


class EditPartContext:
    def __init__(self, swModel, component) -> None:
        self.swModel = swModel
        self.component = component

    def __enter__(self):
        swSelMgr = self.swModel.SelectionManager
        swSelData = swSelMgr.CreateSelectData
        swSelMgr.AddSelectionListObject(self.component, swSelData)
        # self.swModel.EditPart()  # 対象がアセンブリの場合動作しない
        self.swModel.AssemblyPartToggle()  # Obsolete だが代わりにこれを使う

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.swModel.EditAssembly()


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
        if not _is_assembly(self.sldprt_path):
            self._distribute_files([self.sldprt_path], scheduler_address)

    def _setup_after_parallel(self, opt: AbstractOptimizer):

        # validation
        if _is_assembly(self.sldprt_path) and get_worker() is not None:
            # 現在の仕様だと sldprt_path だけが
            # worker_space に保存される。
            # 並列処理に対応するためには
            # すべてのファイルを distribute したうえで
            # 構成部品の置換を実行する必要がある。
            raise RuntimeError(_(
                en_message='Parallel processing is not supported when handling assembly parts with SolidworksInterface.',
                jp_message='SolidworksInterfaceでアセンブリパーツを対象とする場合、並列処理はサポートされていません。'
            ))

        if not _is_assembly(self.sldprt_path):
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
            if _is_assembly(self.sldprt_path):
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
            mgr = _UpdateVariableManager()
            mgr._update_global_variables(swModel, self.current_prm_values)

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


def _is_assembly(swModel_or_name):
    if isinstance(swModel_or_name, str):
        return swModel_or_name.lower().endswith('.sldasm')
    else:
        return swModel_or_name.GetPathName.lower().endswith('.sldasm')


def _iter_parts(swModel):
    components = swModel.GetComponents(
        False  # TopOnly
    )
    return components


class _UpdateVariableManager:

    def __init__(self):
        self.updated_variables = set()

    def _update_global_variables(self, swModel, x: TrialInput):
        # まず自身のパラメータを更新
        asm_logger.debug(f'Processing `{swModel.GetPathName}`')
        self._update_global_variables_core(swModel, x)

        # アセンブリならば、構成部品のパラメータを更新
        if _is_assembly(swModel):
            components = _iter_parts(swModel)
            for component in components:
                swPartModel = component.GetModelDoc2
                asm_logger.debug(f'Checking `{swPartModel.GetPathName}`')
                if swPartModel.GetPathName.lower() not in self.updated_variables:
                    asm_logger.debug(f'Processing `{swPartModel.GetPathName}`')
                    with EditPartContext(swModel, component):
                        self._update_global_variables_core(swPartModel, x)
                    self.updated_variables.add(swPartModel.GetPathName.lower())

    def _update_global_variables_core(self, swModel, x: TrialInput):
        with EquationContext(swModel) as swEqnMgr:
            # txt にリンクされている場合は txt を更新
            if swEqnMgr.LinkToFile:
                self._update_global_variables_linked_txt(swEqnMgr, x)
            self._update_global_variables_simple(swEqnMgr, x)
            # noinspection PyStatementEffect
            swEqnMgr.EvaluateAll

    def _update_global_variables_linked_txt(self, swEqnMgr, x: TrialInput):
        txt_path = swEqnMgr.FilePath
        if txt_path in self.updated_variables:
            return
        with open(txt_path, 'r', encoding='utf_8_sig') as f:
            equations = [line.strip() for line in f.readlines() if line.strip() != '']
        for i, eq in enumerate(equations):
            equations[i] = self._update_equation(eq, x)
        with open(txt_path, 'w', encoding='utf_8_sig') as f:
            f.writelines([eq + '\n' for eq in equations])
        asm_logger.debug(f'`{txt_path}` is updated.')
        self.updated_variables.add(txt_path)

    def _update_global_variables_simple(self, swEqnMgr, x: TrialInput):
        nEquation = swEqnMgr.GetCount

        # equation を列挙
        asm_logger.debug(f'{nEquation} equations detected.')
        for i in range(nEquation):
            # name, equation の取得
            eq = swEqnMgr.Equation(i)
            prm_name = self._get_left(eq)
            # COM 経由なので必要な時以外は触らない
            asm_logger.debug(f'Checking `{prm_name}`')
            if (prm_name in x) and (prm_name not in self.updated_variables):
                asm_logger.debug(f'Processing `{prm_name}`')
                # 特定の Equation がテキストリンク有効か
                # どうかを判定する術がないので、一旦更新する
                new_eq = self._update_equation(eq, x)
                swEqnMgr.Equation(i, new_eq)
                # テキストリンクの場合、COM インスタンスに
                # 更新された値が残ってしまうのでテキストを再読み込み
                if swEqnMgr.LinkToFile:
                    # noinspection PyStatementEffect
                    swEqnMgr.UpdateValuesFromExternalEquationFile
                self.updated_variables.add(prm_name)

    def _update_equation(self, equation: str, x: TrialInput):
        prm_name = self._get_left(equation)
        if prm_name not in x:
            return equation
        prm = x[prm_name]
        right = str(prm.value) + prm.properties.get('unit', '')
        new_eq = f'"{prm_name}" = {right}'
        asm_logger.debug(f'New eq.: `{new_eq}`')
        return new_eq

    @staticmethod
    def _get_left(equation: str):
        return equation.split('=')[0].strip('" ')

    @staticmethod
    def _load(swModel):
        # テスト用関数
        out = set()
        swEqnMgr = swModel.GetEquationMgr
        for i in range(swEqnMgr.GetCount):
            eq = swEqnMgr.Equation(i)
            out.add(eq)
        if _is_assembly(swModel):
            components = _iter_parts(swModel)
            for component in components:
                swPartModel = component.GetModelDoc2
                swEqnMgr = swPartModel.GetEquationMgr
                for i in range(swEqnMgr.GetCount):
                    eq = swEqnMgr.Equation(i)
                    out.add(eq)
        return out


# noinspection PyPep8Naming
def _get_model_by_basename(swApp, basename):
    swModel = swApp.ActivateDoc(basename)
    if swModel is None:
        raise FileNotOpenedError(f'Model {basename} is not opened.')
    return swModel
