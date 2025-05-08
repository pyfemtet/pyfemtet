"""ScipyScalarOptimizer の使用方法です。

"""

from numpy import sin, cos, pi
from pyfemtet.opt import FEMOpt, ScipyScalarOptimizer, NoFEM
import pytest


def x(opt: ScipyScalarOptimizer):
    """r, theta から x 座標を返します。"""
    theta = opt.get_parameter("values")
    return cos(theta)


def y(opt: ScipyScalarOptimizer):
    """r, theta から y 座標を返します。"""
    theta = opt.get_parameter("values")
    return sin(theta)


@pytest.mark.nofem
def test_scipy_scalar_optimizer():

    fem = NoFEM()  # Femtet を使わない設定です。
    opt = ScipyScalarOptimizer()

    femopt = FEMOpt(fem, opt, history_path='ScipyScalar による最適化結果.csv')

    femopt.add_parameter("theta", pi)  # 上下限の設定が必要ありません。

    femopt.add_objective(x, "x 座標", args=(opt,))

    femopt.set_random_seed(42)
    ret = femopt.optimize(
        confirm_before_exit=False,
    )

    for e in femopt._opt_exceptions:
        if e is not None:
            raise e


if __name__ == "__main__":
    test_scipy_scalar_optimizer()
