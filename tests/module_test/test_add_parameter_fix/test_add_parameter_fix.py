import os
from optuna.samplers import RandomSampler
from pyfemtet.opt.optimizer import OptunaOptimizer
from pyfemtet.opt.interface import NoFEM, BoTorchInterface
from pyfemtet.opt.history import History
from pyfemtet.opt.visualization.plotter.pm_graph_creator import plot3d
from pyfemtet.opt.prediction import PyFemtetModel, SingleTaskGPModel

here = os.path.dirname(__file__)
history_path = os.path.join(here, 'test_add_parameter_fix.reccsv')
optimize_history_path = os.path.join(here, 'test_add_parameter_fix_optimize.reccsv')


def objective(_, opt_: OptunaOptimizer):
    return opt_.get_variables('values').astype(float).sum()


def _generate_new_history():
    if os.path.isfile(history_path):
        os.remove(history_path)
        os.remove(history_path + '.db')

    opt = OptunaOptimizer(
        sampler_class=RandomSampler
    )

    opt.add_parameter('fix_x0', 0, None, None, fix=True)
    opt.add_parameter('x1', 0, 0, 1)
    opt.add_parameter('x2', 0, 0, 1)
    # opt.add_parameter('fix_x3', 0, 0, 1, fix=True)
    # opt.add_parameter('fix_x3', 0, None, None, fix=True)
    opt.add_parameter('fix_x4', 0, 0, 1, fix=True)
    opt.add_categorical_parameter('c1', 0, [0, 1, 2])
    opt.add_categorical_parameter('fix_c2', 0, [0, 1, 2], fix=True)
    # opt.add_categorical_parameter('c1', '0', ['0', '1', '2'])
    # opt.add_categorical_parameter('fix_c2', '0', ['0', '1', '2'], fix=True)
    opt.add_objective('objective', objective, args=(opt,))

    opt.fem = NoFEM()
    opt.n_trials = 10
    opt.seed = 42
    opt.history.path = history_path
    opt.run()


def test_load_fixed_parameter():
    history = History()
    history.load_csv(history_path, True)
    print(history.prm_names)
    assert history.prm_names == ['fix_x0', 'x1', 'x2', 'fix_x4', 'c1', 'fix_c2']


def test_show_surrogate_model():
    history = History()
    history.load_csv(history_path, True)

    # 上下限のない変数が plot に入ることは UI 上で禁止する
    # test_prm_names = ['fix_x3', 'fix_c2']
    test_prm_names = ['x1', 'x2']
    params = {name: 0. for name in history.prm_names}

    model = SingleTaskGPModel()
    pyfemtet_model = PyFemtetModel()
    pyfemtet_model.update_model(model)
    pyfemtet_model.fit(
        history,
        history.get_df(),
        **{}
    )

    plot3d(
        history=history,
        prm_name1=test_prm_names[0],
        prm_name2=test_prm_names[1],
        params=params,
        obj_name=history.obj_names[0],
        df=history.get_df(),
        pyfemtet_model=pyfemtet_model,
        n=4,
    ).show()


def test_using_surrogate_model_interface():
    if os.path.isfile(optimize_history_path):
        os.remove(optimize_history_path)
        os.remove(optimize_history_path + '.db')

    opt = OptunaOptimizer(
        sampler_kwargs=dict(
            n_startup_trials=5,
        )
    )

    opt.add_parameter('x1', 0, 0, 1)
    opt.add_parameter('x2', 0, 0, 1)
    opt.add_parameter('fix_x0', 0, 0, 1, fix=True)
    opt.add_parameter('fix_x4', 0, 0, 1, fix=True)
    opt.add_categorical_parameter('c1', 0, [0, 1, 2])
    opt.add_categorical_parameter('fix_c2', 0, [0, 1, 2], fix=True)
    opt.add_objective('objective', None, args=(opt,))

    opt.fem = BoTorchInterface(history_path=history_path)
    opt.n_trials = 10
    opt.seed = 42
    opt.history.path = optimize_history_path
    opt.run()


if __name__ == '__main__':
    _generate_new_history()
    test_load_fixed_parameter()
    test_show_surrogate_model()
    test_using_surrogate_model_interface()
