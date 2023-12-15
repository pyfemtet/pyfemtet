import os
from pyfemtet.opt import FemtetWithNXInterface, OptimizerOptuna


def disp(Femtet):
    '''評価指標を定義する関数は、第一引数に Femtet のインスタンスを取るようにしてください。'''
    # Femtet = Dispatch('FemtetMacro.Femtet')
    # Femtet.OpenCurrentResult(True)
    Gogh = Femtet.Gogh
    _, _, ret = Gogh.Galileo.GetMaxDisplacement_py()
    return ret


def volume(Femtet):
    '''評価指標を定義する関数は、第一引数に Femtet のインスタンスを取るようにしてください。'''
    Gogh = Femtet.Gogh
    _, ret = Gogh.CalcVolume_py([0])
    return ret


def Ax_is_greater_than_Bx(Femtet, femopt):
    '''評価指標を定義する関数は、第一引数に Femtet のインスタンスを取るようにしてください。'''
    value_dict = femopt.get_parameter('dict')
    return value_dict['A_x'] - value_dict['B_x']


def Ay_is_greater_than_By(Femtet, femopt):
    '''評価指標を定義する関数は、第一引数に Femtet のインスタンスを取るようにしてください。'''
    value_dict = femopt.get_parameter('dict')
    return value_dict['A_y'] - value_dict['B_x']


if __name__ == '__main__':

    # ワーキングディレクトリを設定
    here, me = os.path.split(__file__)
    os.chdir(here)

    # NX-Femtet 連携クラスのインスタンス化
    fem = FemtetWithNXInterface(
        prt_path='4_FemtetNX/NXTEST.prt',
        femprj_path='4_FemtetNX/NXTEST.femprj'
    )

    # 最適設計クラスのインスタンス化
    femopt = OptimizerOptuna(fem)

    # 乱数シードの設定
    femopt.set_random_seed(42)

    # 変数設定
    femopt.add_parameter('A_x', 50, lower_bound=25, upper_bound=95)
    femopt.add_parameter('A_y', 45, lower_bound=5, upper_bound=45)
    femopt.add_parameter('B_x', 30, lower_bound=25, upper_bound=95)
    femopt.add_parameter('B_y', 12, lower_bound=5, upper_bound=45)
    femopt.add_parameter('C_x', 90, lower_bound=25, upper_bound=95)
    femopt.add_parameter('C_y', 45, lower_bound=5, upper_bound=45)
    femopt.add_parameter('Cut_x', 10, lower_bound=5, upper_bound=45)
    femopt.add_parameter('Cut_y', 20, lower_bound=5, upper_bound=45)

    # 目的関数設定
    femopt.add_objective(disp, name='変位', direction=0)
    femopt.add_objective(volume, name='体積', direction='minimize')

    # 拘束設定
    femopt.add_constraint(
        Ax_is_greater_than_Bx,
        lower_bound=0,
        args=femopt
        )
    femopt.add_constraint(
        Ay_is_greater_than_By,
        lower_bound=0,
        args=femopt
        )

    # 最適化の実行
    femopt.main(n_trials=5)

    # 結果
    print(femopt.history.data)
