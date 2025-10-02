from pyfemtet.opt import FEMOpt
from pyfemtet.opt.interface import NoFEM, FemtetInterface
from pyfemtet.opt.optimizer import OptunaOptimizer
from optuna.samplers import RandomSampler
from time import sleep
import os


here = os.path.dirname(__file__)


def obj(_, opt: OptunaOptimizer):
    sleep(3)
    x = opt.get_variables('values')
    return (x ** 2).sum()


def test_parallel_nofem(record=False):

    base_path = os.path.join(here, 'test_parallel_record')

    csv_path = base_path + '.csv'
    db_path = base_path + '.db'
    reccsv_path = base_path + '.reccsv'
    if record:
        if os.path.isfile(reccsv_path):
            os.remove(reccsv_path)
    if os.path.isfile(csv_path):
        os.remove(csv_path)
    if os.path.isfile(db_path):
        os.remove(db_path)

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
    dif_df = femopt.optimize(
        with_monitor=False,
        confirm_before_exit=False,
        n_parallel=3,
        n_trials=6,
        history_path=csv_path,
    )

    check_columns_float = ['x1']
    check_columns_str = ['messages']
    if record:
        os.rename(csv_path, reccsv_path)
    else:
        from pyfemtet.opt.history import History
        reference = History()
        reference.load_csv(reccsv_path, with_finalize=True)
        ref_df = reference.get_df()
        for col in check_columns_float:
            ref = [f'{v: .4e}' for v in ref_df[col].values]
            dif = [f'{v: .4e}' for v in dif_df[col].values]
            is_ok = [r == d for r, d in zip(ref, dif)]
            assert all(is_ok), f'{col} error.'
        for col in check_columns_str:
            ref = ref_df[col].values
            dif = dif_df[col].values
            is_ok = [r == d for r, d in zip(ref, dif)]
            assert all(is_ok), f'{col} error.'


if __name__ == '__main__':
    test_parallel_nofem(record=False)
