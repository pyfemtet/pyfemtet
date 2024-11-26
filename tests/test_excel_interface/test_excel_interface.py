import os
os.chdir(os.path.dirname(__file__))

from optuna.samplers import QMCSampler
from pyfemtet.opt import FEMOpt, OptunaOptimizer
from pyfemtet.opt.interface._excel_interface import ExcelInterface



if __name__ == '__main__':

    fem = ExcelInterface(
        input_xlsm_path='io_and_solve.xlsm',
        input_sheet_name='input',
        output_sheet_name='output',
        procedure_name='FemtetMacro.FemtetMain',
        procedure_args=None,
    )

    opt = OptunaOptimizer(
        sampler_class=QMCSampler,
    )

    femopt = FEMOpt(
        fem=fem,
        opt=opt,
        # history_path='history.csv'
    )

    femopt.optimize(
        n_trials=30,
        confirm_before_exit=False,
        n_parallel=3,
    )

