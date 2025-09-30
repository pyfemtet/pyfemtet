import os
from optuna.samplers import RandomSampler
from pyfemtet.opt.optimizer import OptunaOptimizer, PoFBoTorchSampler
from pyfemtet.opt.interface import NoFEM
from pyfemtet.opt import FEMOpt


here = os.path.dirname(__file__)


def obj(_, opt: OptunaOptimizer):
    x = opt.get_variables('values')
    return (x ** 2).sum()


def bounds_x2(opt: OptunaOptimizer):
    variables = opt.get_variables()
    # add_parameter した順に変数が決まっていきます！
    # 順番を間違えると、前回の最適化試行での変数値を参照することになります。
    x1 = variables['x1']
    return 0, x1


def simple_run(record=False):

    history_base_path = os.path.join(here, 'test_dynamic')
    csv_path = history_base_path + '.csv'
    db_path = history_base_path + '.db'
    reccsv_path = history_base_path + '.reccsv'
    if os.path.isfile(csv_path):
        os.remove(csv_path)
    if os.path.isfile(db_path):
        os.remove(db_path)

    opt = OptunaOptimizer(
        sampler_class=RandomSampler
    )
    opt.fem = NoFEM()
    opt.add_parameter('x1', 1, 0, 1)
    opt.add_parameter(
        'x2', 0.5,
        properties={
            'dynamic_bounds_fun': bounds_x2,
        },
    )
    opt.add_objective('y', obj, args=(opt,))
    opt.n_trials = 10
    opt.history.path = csv_path
    opt.seed = 42
    opt.run()

    if record:
        if os.path.isfile(reccsv_path):
            os.remove(reccsv_path)
        os.rename(csv_path, reccsv_path)
    else:
        import numpy as np
        from pyfemtet.opt.history import History
        history_ref = History()
        history_ref.load_csv(reccsv_path, with_finalize=True)

        df_ref = history_ref.get_df()
        df_dif = opt.history.get_df()

        cols = ['x1', 'x2', 'x2_upper_bound']
        for col in cols:
            ref = df_ref[col].values
            dif = df_dif[col].values
            assert all(np.abs((dif - ref) / ref) < 0.1)


def simple_run_femopt(record=False, confirm_before_exit=False):

    history_base_path = os.path.join(here, 'test_dynamic')
    csv_path = history_base_path + '.csv'
    db_path = history_base_path + '.db'
    reccsv_path = history_base_path + '.reccsv'
    if os.path.isfile(csv_path):
        os.remove(csv_path)
    if os.path.isfile(db_path):
        os.remove(db_path)

    opt = OptunaOptimizer(
        sampler_class=PoFBoTorchSampler,
        sampler_kwargs=dict(
            n_startup_trials=2,
        ),
    )
    opt.fem = NoFEM()
    opt.add_parameter('x1', 1, 0, 1)
    opt.add_parameter(
        name='x2',
        initial_value=0.5,
        properties={
            'dynamic_bounds_fun': bounds_x2,
        },
    )
    opt.add_objective('y', obj, args=(opt,))
    femopt = FEMOpt(fem=opt.fem, opt=opt)
    femopt.set_random_seed(42)
    femopt.optimize(
        history_path=csv_path,
        n_trials=10,
        confirm_before_exit=confirm_before_exit,
    )

    if record:
        if os.path.isfile(reccsv_path):
            os.remove(reccsv_path)
        os.rename(csv_path, reccsv_path)
    else:
        import numpy as np
        from pyfemtet.opt.history import History
        history_ref = History()
        history_ref.load_csv(reccsv_path, with_finalize=True)

        df_ref = history_ref.get_df()
        df_dif = opt.history.get_df()

        cols = ['x1', 'x2', 'x2_upper_bound']
        for col in cols:
            ref = df_ref[col].values
            dif = df_dif[col].values
            assert all(np.abs((dif - ref) / ref) < 0.1)


def check_graph():
    ...


if __name__ == '__main__':
    # simple_run()
    simple_run_femopt(confirm_before_exit=True)
