"""ScipyOptimizer の使用方法です。

"""

from numpy import sin, cos, pi
from pyfemtet.opt import FEMOpt, ScipyOptimizer, NoFEM
import pytest


def x(opt: ScipyOptimizer):
    """r, theta から x 座標を返します。"""
    r, theta = opt.get_parameter("values")
    return r * cos(theta)


@pytest.mark.nofem
def test_scipy_optimizer():

    fem = NoFEM()  # Femtet を使わない設定です。
    opt = ScipyOptimizer()

    femopt = FEMOpt(fem, opt, history_path='Scipyによる最適化結果.csv')

    femopt.add_parameter("r", 0.5, 0, 1)
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
    test_scipy_optimizer()
