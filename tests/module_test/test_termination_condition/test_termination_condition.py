from pyfemtet.opt.optimizer import ScipyOptimizer, OptunaOptimizer
from pyfemtet.opt.interface import NoFEM
from pyfemtet.opt.history import History
from time import sleep


def termination_condition(history: History):
    """y の値が 10 以下になったら終了する。"""
    df = history.get_df()
    optimal_values = df[df['optimality']]
    if len(optimal_values) == 0:
        return False
    obj_values = optimal_values[history.obj_names].values
    obj_value = obj_values[-1]
    obj = obj_value[0]
    if obj <= 10.0:
        return True
    return False


def test_termination_condition_scipy():
    """
    y = x**2 の最適化を行う。
    手法は Scipy の trust-constr とする。
    初期値は 10。
    """
    opt = ScipyOptimizer(
        method="trust-constr",
    )
    opt.fem = NoFEM()

    opt.add_parameter('x', initial_value=10.0)
    opt.set_termination_condition(termination_condition)
    opt.add_objective('y', lambda _, opt_: opt_.get_variables('values') ** 2, args=(opt,))
    opt.n_trials = 10
    opt.run()

    assert len(opt.history.get_df()) < 10


def test_termination_condition_optuna():
    """
    y = x**2 の最適化を行う。
    手法は optuna の TPE とする。
    初期値は 10。
    """
    # 1 秒以内にふたつの最適化が始まると
    # csv 名が重複してエラーになる
    sleep(2)

    opt = OptunaOptimizer()
    opt.fem = NoFEM()

    def objective(_, opt_):
        y = opt_.get_variables('values') ** 2
        return y

    opt.add_parameter('x', initial_value=10.0, lower_bound=-10.0, upper_bound=10.0)
    opt.set_termination_condition(termination_condition)
    opt.add_objective('y', objective, args=(opt,))
    opt.n_trials = 10
    opt.seed = 42
    opt.run()

    assert len(opt.history.get_df()) < 10


if __name__ == '__main__':
    test_termination_condition_scipy()
    test_termination_condition_optuna()
