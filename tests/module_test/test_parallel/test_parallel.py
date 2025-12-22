import os
from time import sleep
from optuna.samplers import RandomSampler
from pyfemtet.opt import FEMOpt
from pyfemtet.opt.interface import NoFEM, FemtetInterface
from pyfemtet.opt.optimizer import OptunaOptimizer
from tests.utils.reccsv_processor import RECCSV

import pytest

here = os.path.dirname(__file__)


def obj(_, opt: OptunaOptimizer):
    sleep(7)
    x = opt.get_variables('values')
    return (x ** 2).sum()


def test_parallel_nofem(record=False):

    base_path = os.path.join(here, 'test_parallel_record')
    reccsv = RECCSV(base_path, record)

    os.environ['DEBUG_FEMOPT_PARALLEL'] = '1'

    fem = NoFEM()
    opt = OptunaOptimizer(
        sampler_class=RandomSampler
    )
    femopt = FEMOpt(fem=fem, opt=opt)
    femopt.add_parameter('x1', 1, 0, 1)
    femopt.add_parameter('x2', 1, 0, 1)
    femopt.add_objective('y', obj, args=(opt,))
    femopt.set_random_seed(42)
    df = femopt.optimize(
        with_monitor=False,
        confirm_before_exit=False,
        n_parallel=3,
        n_trials=6,
        history_path=reccsv.csv_path,
    )

    check_columns_float = ['x1']
    check_columns_str = ['messages']
    reccsv.check(check_columns_float, check_columns_str, df)


@pytest.mark.fem
def test_parallel_femtet(record=False):

    base_path = os.path.join(here, 'test_parallel_femtet')
    reccsv = RECCSV(base_path, record)

    os.environ['DEBUG_FEMOPT_PARALLEL'] = '1'

    fem = FemtetInterface(
        femprj_path=os.path.join(here, 'test_parallel_femtet.femprj'),
        # strictly_pid_specify=False,
    )
    opt = OptunaOptimizer(
        sampler_class=RandomSampler
    )
    femopt = FEMOpt(fem=fem, opt=opt)
    femopt.add_parameter('x1', 1, 0, 1)
    femopt.add_parameter('x2', 1, 0, 1)
    femopt.add_objective('y', obj, args=(opt,))
    femopt.set_random_seed(42)
    df = femopt.optimize(
        with_monitor=False,
        confirm_before_exit=False,
        n_parallel=3,
        n_trials=6,
        history_path=reccsv.csv_path,
    )

    check_columns_float = ['x1']
    check_columns_str = ['messages']
    reccsv.check(check_columns_float, check_columns_str, df)


if __name__ == '__main__':
    test_parallel_nofem(record=False)
    test_parallel_femtet(record=False)
