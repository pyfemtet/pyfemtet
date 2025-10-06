from typing import TypeAlias

SWVariables: TypeAlias = dict[str, str]
"""<prm_name>: <expression>. Note that the <prm_name> does not contain `"`."""


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


def is_assembly(swModel_or_name):
    if isinstance(swModel_or_name, str):
        return swModel_or_name.lower().endswith('.sldasm')
    else:
        return swModel_or_name.GetPathName.lower().endswith('.sldasm')


def _iter_parts(swModel):
    components = swModel.GetComponents(
        False  # TopOnly
    )
    return components


# Used by pyfemtet-opt-gui
class SolidworksVariableManager:

    def __init__(self, logger=None):
        # Used by pyfemtet-opt-gui
        self.updated_objects = set()
        """Updated variable names, sldprt file paths and linked .txt equation file paths."""
        if logger is None:
            from logging import getLogger
            logger = getLogger('solidworks_variable_manager')
        self.logger = logger

    # Used by pyfemtet-opt-gui
    def update_global_variables_recourse(self, swModel, x: SWVariables):
        # まず自身のパラメータを更新
        self.logger.debug(f'Processing `{swModel.GetPathName}`')
        self._update_global_variables_core(swModel, x)

        # アセンブリならば、構成部品のパラメータを更新
        if is_assembly(swModel):
            components = _iter_parts(swModel)
            for component in components:
                swPartModel = component.GetModelDoc2
                self.logger.debug(f'Checking `{swPartModel.GetPathName}`')
                if swPartModel.GetPathName.lower() not in self.updated_objects:
                    self.logger.debug(f'Processing `{swPartModel.GetPathName}`')
                    with EditPartContext(swModel, component):
                        self._update_global_variables_core(swPartModel, x)
                    self.updated_objects.add(swPartModel.GetPathName.lower())

    def _update_global_variables_core(self, swModel, x: SWVariables):
        with EquationContext(swModel) as swEqnMgr:
            # txt にリンクされている場合は txt を更新
            if swEqnMgr.LinkToFile:
                self._update_global_variables_linked_txt(swEqnMgr, x)
            self._update_global_variables_simple(swEqnMgr, x)
            # noinspection PyStatementEffect
            swEqnMgr.EvaluateAll

    def _update_global_variables_linked_txt(self, swEqnMgr, x: SWVariables):
        txt_path = swEqnMgr.FilePath
        if txt_path in self.updated_objects:
            return
        with open(txt_path, 'r', encoding='utf_8_sig') as f:
            equations = [line.strip() for line in f.readlines() if line.strip() != '']
        for i, eq in enumerate(equations):
            equations[i] = self._update_equation(eq, x)
        with open(txt_path, 'w', encoding='utf_8_sig') as f:
            f.writelines([eq + '\n' for eq in equations])
        self.logger.debug(f'`{txt_path}` is updated.')
        self.updated_objects.add(txt_path)

    def _update_global_variables_simple(self, swEqnMgr, x: SWVariables):
        nEquation = swEqnMgr.GetCount

        # equation を列挙
        self.logger.debug(f'{nEquation} equations detected.')
        for i in range(nEquation):
            # name, equation の取得
            eq = swEqnMgr.Equation(i)
            prm_name = self._get_left(eq)
            # COM 経由なので必要な時以外は触らない
            self.logger.debug(f'Checking `{prm_name}`')
            if (prm_name in x) and (prm_name not in self.updated_objects):
                self.logger.debug(f'Processing `{prm_name}`')
                # 特定の Equation がテキストリンク有効か
                # どうかを判定する術がないので、一旦更新する
                new_eq = self._update_equation(eq, x)
                swEqnMgr.Equation(i, new_eq)
                # テキストリンクの場合、COM インスタンスに
                # 更新された値が残ってしまうのでテキストを再読み込み
                if swEqnMgr.LinkToFile:
                    # noinspection PyStatementEffect
                    swEqnMgr.UpdateValuesFromExternalEquationFile
                self.updated_objects.add(prm_name)

    def _update_equation(self, equation: str, x: SWVariables):
        prm_name = self._get_left(equation)
        if prm_name not in x:
            return equation
        new_eq = f'"{prm_name}" = {x[prm_name]}'
        self.logger.debug(f'New eq.: `{new_eq}`')
        return new_eq

    @staticmethod
    def _get_left(equation: str):
        tmp = equation.split('=')
        if len(tmp) == 0:
            raise RuntimeError(f'Invalid solidworks equation: {equation} (no `=` contained)')
        return tmp[0].strip('" ')

    # Used by pyfemtet-opt-gui
    @staticmethod
    def get_equations_recourse(swModel, global_variables_only=False) -> list[str]:
        out = list()
        swEqnMgr = swModel.GetEquationMgr
        for i in range(swEqnMgr.GetCount):
            if global_variables_only and not swEqnMgr.GlobalVariable(i):
                continue
            eq = swEqnMgr.Equation(i)
            out.append(eq)
        if is_assembly(swModel):
            components = _iter_parts(swModel)
            for component in components:
                swPartModel = component.GetModelDoc2
                swEqnMgr = swPartModel.GetEquationMgr
                for i in range(swEqnMgr.GetCount):
                    if global_variables_only and not swEqnMgr.GlobalVariable(i):
                        continue
                    eq = swEqnMgr.Equation(i)
                    out.append(eq)
        return out
