import os
from pyfemtet.opt.history import History
from pyfemtet.opt.optimizer import AbstractOptimizer, OptunaOptimizer
from pyfemtet.opt.interface import AbstractSurrogateModelInterfaceBase, NoFEM, BoTorchInterface


def test_output_directions():
    history = History()
    history.obj_names = ['obj1', 'obj2', 'obj3']
    opt = AbstractOptimizer()

    def check_fail(_output_directions):

        fem = AbstractSurrogateModelInterfaceBase(
            train_history=history,
            _output_directions=_output_directions
        )
        try:
            fem.load_objectives(opt)
        except AssertionError as e:
            if '_output_directions' in str(e):
                print('正しいエラー', str(e))
            else:
                assert False, 'エラーメッセージが不正'
        else:
            assert False, '起こるはずのエラーが起こらない'

    def check(_output_directions):
        opt_ = AbstractOptimizer()

        fem = AbstractSurrogateModelInterfaceBase(
            train_history=history,
            _output_directions=_output_directions
        )
        fem.load_objectives(opt_)

    check_fail(('minimize',))
    check_fail({'obj4': 'minimize',})
    check_fail({0: 0, 'obj2': 'minimize',})
    check(('minimize', 0, 'maximize',))
    check({'obj1': 'minimize',})
    check({0: 0, 1: 'minimize',})


def parabola(_, opt: OptunaOptimizer):
    return (opt.get_variables('values') ** 2).sum()


def test_surrogate_optimize_output_directions():
    fem_train = NoFEM()
    opt_train = OptunaOptimizer()
    opt_train.fem = fem_train
    opt_train.add_parameter('x1', -1, -1, 1)
    opt_train.add_parameter('x2', -1, -1, 1)
    opt_train.add_parameter('x3', -1, -1, 1, 0.25)
    opt_train.add_objective('obj', parabola, args=(opt_train,))
    opt_train.n_trials = 10
    opt_train.history.path = 'tmp_training_test_surrogate_optimize_output_directions.csv'
    if os.path.isfile(opt_train.history.path):
        os.remove(opt_train.history.path)
        os.remove(opt_train.history.path.removesuffix('.csv') + '.db')
    opt_train.run()
    opt_train.history.save()

    fem = BoTorchInterface(
        history_path=opt_train.history.path,
        _output_directions=('minimize',)
    )
    opt = OptunaOptimizer()
    opt.fem = fem
    opt.add_parameter('x1', -1, -1, 1)
    opt.add_parameter('x2', -1, -1, 1)
    opt.add_parameter('x3', -1, -1, 1, 0.25)
    opt.n_trials = 20
    opt.run()

    df = opt.history.get_df()
    assert df[opt.history.obj_names].values.min() < 3
    print('test_surrogate_optimize_output_directions passed!')


def test_surrogate_optimize_normal():
    fem_train = NoFEM()
    opt_train = OptunaOptimizer()
    opt_train.fem = fem_train
    opt_train.add_parameter('x1', -1, -1, 1)
    opt_train.add_parameter('x2', -1, -1, 1)
    opt_train.add_parameter('x3', -1, -1, 1, 0.25)
    opt_train.add_objective('obj', parabola, args=(opt_train,))
    opt_train.n_trials = 10
    opt_train.history.path = 'tmp_training_test_surrogate_optimize_normal.csv'
    if os.path.isfile(opt_train.history.path):
        os.remove(opt_train.history.path)
        os.remove(opt_train.history.path.removesuffix('.csv') + '.db')
    opt_train.run()
    opt_train.history.save()

    fem = BoTorchInterface(
        history_path=opt_train.history.path,
    )
    opt = OptunaOptimizer()
    opt.fem = fem
    opt.add_parameter('x1', -1, -1, 1)
    opt.add_parameter('x2', -1, -1, 1)
    opt.add_parameter('x3', -1, -1, 1, 0.25)
    opt.add_objective('obj', lambda: 100.,)
    opt.n_trials = 20
    opt.run()

    df = opt.history.get_df()
    assert df[opt.history.obj_names].values.min() < 3
    print('test_surrogate_optimize_normal passed!')


if __name__ == '__main__':
    # test_output_directions()
    # test_surrogate_optimize_output_directions()
    test_surrogate_optimize_normal()
