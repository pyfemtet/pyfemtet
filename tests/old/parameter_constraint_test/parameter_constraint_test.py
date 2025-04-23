"""parameter_constraint のサンプルです。

parameter_constraint は PyFemtet の新機能で、
OptunaOptimizer かつ BoTorchSampler を使う場合に限り
拘束を満たさない変数提案を生成しないようにする機能です。

従来の constraint は、生成された提案を
単に skip するものなので BoTorchSampler では
似たような提案値を繰り返し最適化が実質
進まなくなる現象が発生しました。
これを回避することができます。

"""

from numpy import sin, cos, pi
from optuna_integration import BoTorchSampler
from pyfemtet.opt import FEMOpt, OptunaOptimizer, NoFEM


def x(opt: OptunaOptimizer):
    """r, theta から x 座標を返します。"""
    r, theta = opt.get_parameter("values")
    return r * cos(theta)


def y(opt: OptunaOptimizer):
    """r, theta から y 座標を返します。"""
    r, theta = opt.get_parameter("values")
    return r * sin(theta)


def constraint_1(opt: OptunaOptimizer):
    """r, theta の関係式を返します。"""
    _r, _theta = opt.get_parameter("values")
    return pi * _r - _theta  # >= 0


def constraint_2(opt: OptunaOptimizer):
    """r, theta の関係式を返します。"""
    _r, _theta = opt.get_parameter("values")
    return _theta - pi / 2 * _r  # >= 0


def main():

    fem = NoFEM()  # サンプルのため Femtet を使わない
    opt = OptunaOptimizer(
        sampler_class=BoTorchSampler,
        sampler_kwargs=dict(
            consider_running_trials=True,
            n_startup_trials=10,  # 最初の 10 回が解析できるまではランダムサンプリングを行います
        )
    )

    femopt = FEMOpt(fem, opt, history_path='拘束なし.csv')

    femopt.add_parameter("r", 0.5, 0, 1)
    femopt.add_parameter("theta", 3 / 8 * pi, 0, 2 * pi)

    femopt.add_objective(x, "x 座標", args=(femopt.opt,), direction="maximize")
    femopt.add_objective(y, "y 座標", args=(femopt.opt,), direction="maximize")

    femopt.add_constraint(constraint_1, lower_bound=0, kwargs=dict(opt=opt))
    femopt.add_constraint(constraint_2, lower_bound=0, kwargs=dict(opt=opt))

    femopt.set_random_seed(42)
    ret = femopt.optimize(
        n_trials=20,
        n_parallel=1,
        wait_setup=True,
        confirm_before_exit=False,
    )


if __name__ == "__main__":
    main()
