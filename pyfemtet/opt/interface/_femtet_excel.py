import re
from pathlib import Path

from pyfemtet.opt.interface._base import FEMInterface
from pyfemtet.opt.interface._femtet import FemtetInterface
from pyfemtet.opt.interface._excel_interface import (
    ExcelInterface, is_cell_value_empty, ParseAsObjective, ScapeGoatObjective,
    ParseAsConstraint, search_c, search_r
)


PARAMETRIC_PREFIX = 'パラメトリック'


def get_number(name):
    numbers = re.findall(r'\d+', name)
    if len(numbers) == 0:
        raise ValueError('パラメトリック結果出力の番号指定が検出できませんでした。')
    else:
        return int(numbers[0])



class FemtetWithExcelSettingsInterface(FemtetInterface, ExcelInterface, FEMInterface):

    def __init__(
            self,

            # FemtetInterface arguments
            femprj_path: str = None, model_name: str = None, connect_method: str = 'auto',
            save_pdt: str = 'all', strictly_pid_specify: bool = True, allow_without_project: bool = False,
            open_result_with_gui: bool = True,
            parametric_output_indexes_use_as_objective: dict[int, str or float] = None,

            # ExcelInterface arguments
            input_xlsm_path: str or Path = None, input_sheet_name: str = None, output_xlsm_path: str or Path = None,
            output_sheet_name: str = None, constraint_xlsm_path: str or Path = None,
            constraint_sheet_name: str = None, procedure_name: str = None, procedure_args: list or tuple = None,
            procedure_timeout: float or None = None,
            setup_xlsm_path: str or Path = None, setup_procedure_name: str = None,
            setup_procedure_args: list or tuple = None, teardown_xlsm_path: str or Path = None,
            teardown_procedure_name: str = None, teardown_procedure_args: list or tuple = None,
            related_file_paths: list[str or Path] = None, visible: bool = False, display_alerts: bool = False,
            terminate_excel_when_quit: bool = None, interactive: bool = True, use_named_range: bool = True,

    ):

        excel_connect_method = 'new'

        ExcelInterface.__init__(
            self, input_xlsm_path, input_sheet_name, output_xlsm_path, output_sheet_name, constraint_xlsm_path,
            constraint_sheet_name, procedure_name, procedure_args, excel_connect_method, procedure_timeout,
            setup_xlsm_path, setup_procedure_name, setup_procedure_args, teardown_xlsm_path,
            teardown_procedure_name, teardown_procedure_args, related_file_paths, visible, display_alerts,
            terminate_excel_when_quit, interactive, use_named_range)

        FemtetInterface.__init__(
            self,
            femprj_path, model_name, connect_method, save_pdt, strictly_pid_specify, allow_without_project,
            open_result_with_gui, parametric_output_indexes_use_as_objective
        )


    def load_objective(self, opt, raise_if_no_keyword=True):
        from pyfemtet.opt.optimizer import AbstractOptimizer
        from pyfemtet.opt._femopt_core import Objective
        opt: AbstractOptimizer

        df = ParseAsObjective.parse(
            self.output_xlsm_path,
            self.output_sheet_name,
            raise_if_no_keyword,
        )

        for i, row in df.iterrows():

            # use(optional)
            use = True
            if ParseAsObjective.use in df.columns:
                _use = row[ParseAsObjective.use]
                use = False if is_cell_value_empty(_use) else bool(_use)  # bool or NaN

            # name
            name = str(row[ParseAsObjective.name])

            # direction
            direction = row[ParseAsObjective.direction]
            assert not is_cell_value_empty(direction), 'direction is empty.'
            try:
                direction = float(direction)
            except ValueError:
                direction = str(direction).lower()
                assert direction in ['minimize', 'maximize']

            if use:

                # name が「パラメトリック」から始まっていたら
                # パラメトリック解析の結果を目的関数にする
                if name.startswith(PARAMETRIC_PREFIX):
                    number = get_number(name)
                    self.use_parametric_output_as_objective(number, direction)

                # そうでなければ通常の Excel objective を作る
                else:
                    opt.objectives[name] = Objective(
                        fun=ScapeGoatObjective(),
                        name=name,
                        direction=direction,
                        args=(name,),
                        kwargs=dict(),
                    )

    def _setup_before_parallel(self, client):
        FemtetInterface._setup_before_parallel(self, client)
        ExcelInterface._setup_before_parallel(self, client)

    def _setup_after_parallel(self, *args, **kwargs):
        FemtetInterface._setup_after_parallel(self, *args, **kwargs)
        ExcelInterface._setup_after_parallel(self, *args, **kwargs)

    def update(self, parameters) -> None:
        FemtetInterface.update(self, parameters)
        ExcelInterface.update(self, parameters)

    def quit(self, timeout=1, force=True):
        FemtetInterface.quit(self, timeout, force)
        ExcelInterface.quit(self)

    # noinspection PyMethodOverriding
    def objective_from_excel(self, _, name: str):
        r = 1 + search_r(self.output_xlsm_path, self.output_sheet_name, name)
        c = 1 + search_c(self.output_xlsm_path, self.output_sheet_name, ParseAsObjective.value)
        v = self.sh_output.Cells(r, c).value
        return float(v)

    # noinspection PyMethodOverriding
    def constraint_from_excel(self, _, name: str):
        r = 1 + search_r(self.constraint_xlsm_path, self.constraint_sheet_name, name)
        c = 1 + search_c(self.constraint_xlsm_path, self.constraint_sheet_name, ParseAsConstraint.value)
        v = self.sh_constraint.Cells(r, c).value
        return float(v)
