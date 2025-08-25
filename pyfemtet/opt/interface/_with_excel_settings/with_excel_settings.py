from __future__ import annotations

from typing import TYPE_CHECKING

import re

from pyfemtet.opt.problem.problem import *
from pyfemtet.opt.interface._base_interface import AbstractFEMInterface
from pyfemtet.opt.interface._excel_interface import ExcelInterface
from pyfemtet.opt.interface._femtet_interface import FemtetInterface

if TYPE_CHECKING:
    from pyfemtet.opt.optimizer import AbstractOptimizer


def _get_name(FEMClass: type):
    return FEMClass.__name__.removesuffix('Interface') + 'WithExcelSettingsInterface'


def _is_parametric_output(obj_name: str) -> bool:
    if obj_name.startswith('パラメトリック結果出力'):
        return True

    return False


class ParametricNumberNotFoundError(Exception):
    pass


def _get_parametric_output_number(obj_name: str) -> int:
    # 最後のマッチを取得
    matches = re.findall(r'\d+', obj_name)

    if matches:
        last_match = matches[-1]
        return int(last_match)

    else:
        raise ParametricNumberNotFoundError(
            'Excel の設定シートからパラメトリック結果出力機能を'
            '使うことを意図した目的名が検出されましたが、'
            '出力番号が取得できませんでした。目的名には、'
            'パラメトリック結果出力番号を表す自然数を含めてください。'
            '例：「パラメトリック結果出力 1 番」')


def _class_factory(FEMClass: type[AbstractFEMInterface]) -> type[AbstractFEMInterface]:

    class _WithExcelSettingsInterface(FEMClass, ExcelInterface):

        __name__ = _get_name(FEMClass=FEMClass)

        # 構造が複雑で型ヒントが働かないため
        _excel_initialized: bool
        _load_problem_from_fem: bool

        def init_excel(self, *args, **kwargs):
            ExcelInterface.__init__(self, *args, **kwargs)
            self._excel_initialized = True
            self._load_problem_from_fem = True

        def _setup_before_parallel(self, scheduler_address=None):
            assert self._excel_initialized, '最初に init_excel() を呼び出してください。'
            ExcelInterface._setup_before_parallel(self, scheduler_address)
            FEMClass._setup_before_parallel(self, scheduler_address)

        def _setup_after_parallel(self, opt: AbstractOptimizer):
            assert self._excel_initialized, '最初に init_excel() を呼び出してください。'
            ExcelInterface._setup_after_parallel(self, opt)
            FEMClass._setup_after_parallel(self, opt)

        def load_variable(self, opt: AbstractOptimizer, raise_if_no_keyword=True) -> None:
            assert self._excel_initialized, '最初に init_excel() を呼び出してください。'
            ExcelInterface.load_variables(self, opt, raise_if_no_keyword)
            FEMClass.load_variables(self, opt)

        def load_objectives(self, opt: AbstractOptimizer, raise_if_no_keyword=False) -> None:
            assert self._excel_initialized, '最初に init_excel() を呼び出してください。'
            ExcelInterface.load_objectives(self, opt, raise_if_no_keyword)

            # FEMClass が FemtetInterface なら
            #   この時点での objectives を列挙して
            #   「パラメトリック」から始まる目的関数を
            #   削除して parametric 結果出力の
            #   それに置き換える
            if isinstance(self, FemtetInterface):

                # 削除すべき obj を取得
                obj_names_to_replace = []
                for obj_name in opt.objectives.keys():
                    if _is_parametric_output(obj_name):
                        obj_names_to_replace.append(obj_name)

                # 削除と置換の予約
                for old_obj_name in obj_names_to_replace:

                    # 削除
                    old_obj = opt.objectives.pop(old_obj_name)

                    # 追加すべき内容の決定
                    self: FemtetInterface
                    self.use_parametric_output_as_objective(
                        number=_get_parametric_output_number(old_obj_name),
                        direction=old_obj.direction,
                    )

                # parametric 結果出力の追加の実行
                FemtetInterface.load_objectives(self, opt)

            else:
                FEMClass.load_objectives(self, opt)

        def load_constraints(self, opt: AbstractOptimizer, raise_if_no_keyword=False) -> None:
            assert self._excel_initialized, '最初に init_excel() を呼び出してください。'
            ExcelInterface.load_constraints(self, opt, raise_if_no_keyword)
            FEMClass.load_constraints(self, opt)

        def update_parameter(self, x: TrialInput) -> None:
            assert self._excel_initialized, '最初に init_excel() を呼び出してください。'
            ExcelInterface.update_parameter(self, x)
            FEMClass.update_parameter(self, x)

        def update(self) -> None:
            assert self._excel_initialized, '最初に init_excel() を呼び出してください。'
            ExcelInterface.update(self)
            FEMClass.update(self)

        def close(self):
            assert self._excel_initialized, '最初に init_excel() を呼び出してください。'
            ExcelInterface.close(self)
            FEMClass.close(self)

    return _WithExcelSettingsInterface
