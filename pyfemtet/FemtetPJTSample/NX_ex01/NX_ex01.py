import os
from pyfemtet.opt import FemtetWithNXInterface, OptimizerOptuna


here, me = os.path.split(__file__)
os.chdir(here)


def disp(Femtet):
    _, _, ret = Femtet.Gogh.Galileo.GetMaxDisplacement_py()
    return ret


def volume(Femtet):
    _, ret = Femtet.Gogh.CalcVolume_py([0])
    return ret


def C_minus_B(_, femopt):
    A, B, C = femopt.get_parameter('values')
    return C - B


if __name__ == '__main__':

    # NX-Femtet 連携クラスのインスタンス化
    fem = FemtetWithNXInterface(
        prt_path='NX_ex01.prt',
        femprj_path='NX_ex01.femprj',
    )

    # 最適設計クラスのインスタンス化
    femopt = OptimizerOptuna(fem)

    # 乱数シードの設定
    femopt.set_random_seed(42)

    # 変数設定
    femopt.add_parameter('A', 10, lower_bound=1, upper_bound=59)
    femopt.add_parameter('B', 10, lower_bound=1, upper_bound=40)
    femopt.add_parameter('C', 20, lower_bound=5, upper_bound=59)
    femopt.add_constraint(C_minus_B, 'C>B', lower_bound=1, args=femopt)

    # 目的関数設定
    femopt.add_objective(disp, name='変位', direction=0)
    femopt.add_objective(volume, name='体積', direction='minimize')

    # 最適化の実行
    femopt.main(n_trials=20, use_lhs_init=False)
