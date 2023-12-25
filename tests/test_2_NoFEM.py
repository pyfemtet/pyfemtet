from time import sleep
import numpy as np
import pandas as pd
from pyfemtet.opt import OptimizerOptuna, NoFEM


should_sleep = True

def objective_x(femopt):
    if should_sleep: sleep(1)
    r, theta, fai = femopt.get_parameter('values')
    return r * np.cos(theta) * np.cos(fai)


def objective_y(femopt):
    r, theta, fai = femopt.get_parameter('values')
    return r * np.cos(theta) * np.sin(fai)


def objective_z(femopt):
    r, theta, fai = femopt.get_parameter('values')
    return r * np.sin(theta)


def constraint_y(femopt):
    y = objective_y(femopt)
    return y


def constraint_z(femopt):
    z = objective_z(femopt)
    return z


def test_2_1():
    """
    テストしたい状況
        FEM なしで一通りの機能が動くか
    パラメータ
        手法
    結果
        結果が保存したものと一致するか
    """

    fem = NoFEM()
    femopt = OptimizerOptuna(fem)
    femopt.set_random_seed(42)
    femopt.add_parameter('r', .5, 0, 1)
    femopt.add_parameter('theta', np.pi/3, -np.pi/2, np.pi/2)  # 空間上で xy 平面となす角
    femopt.add_parameter('fai', (7/6)*np.pi, 0, 2*np.pi)  # xy 平面上で x 軸となす角
    femopt.add_objective(objective_x, args=femopt)  # 名前なし目的変数（obj_0 になる）
    femopt.add_objective(objective_x, args=femopt)  # 名前なし目的変数（obj_1 になる）
    femopt.add_objective(objective_y, 'y(mm)', args=femopt)
    femopt.add_objective(objective_y, 'y(mm)', args=femopt, direction='maximize')  # 上書きかつ direction 指定
    femopt.add_objective(objective_z, 'z(mm)', args=femopt, direction=-1)  # direction 数値指定
    femopt.add_constraint(constraint_y, 'y<=0', upper_bound=-1, args=femopt)
    femopt.add_constraint(constraint_y, 'y<=0', upper_bound=0, args=femopt)  # 上書き
    femopt.add_constraint(constraint_z, 'z<=0', upper_bound=0, args=femopt, strict=False)
    femopt.main(n_trials=30, n_parallel=3)
    femopt.terminate_monitor()

    # データの取得
    ref_df = pd.read_csv(f'test2/test2_TPE.csvdata').replace(np.nan, None)
    def_df = femopt.history.data.copy()

    # 並べ替え（並列しているから順番は違いうる）
    ref_df = ref_df.iloc[:, 1:].sort_values('r').sort_values('theta').sort_values('fai').select_dtypes(include='number')
    def_df = def_df.iloc[:, 1:].sort_values('r').sort_values('theta').sort_values('fai').select_dtypes(include='number')

    assert np.sum(np.abs(def_df.values - ref_df.values)) < 0.001



def simple():
    """シンプルな動作確認用"""

    fem = NoFEM()
    femopt = OptimizerOptuna(fem)
    femopt.set_random_seed(42)
    femopt.add_parameter('r', .5, 0, 1)
    femopt.add_parameter('theta', -np.pi/3, -np.pi/2, np.pi/2)  # 空間上で xy 平面となす角
    femopt.add_parameter('fai', (7/6)*np.pi, 0, 2*np.pi)  # xy 平面上で x 軸となす角
    femopt.add_objective(objective_x, 'x', args=femopt)
    femopt.add_objective(objective_y, 'y', args=femopt)
    femopt.add_objective(objective_z, 'z', args=femopt)
    femopt.add_constraint(objective_z, 'z<=0', upper_bound=0, args=femopt)
    femopt.main(n_trials=30, n_parallel=3)
    femopt.terminate_monitor()





if __name__ == '__main__':
    simple()


