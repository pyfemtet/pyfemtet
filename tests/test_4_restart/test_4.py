import os
import subprocess
from time import sleep
import numpy as np
import pandas as pd
import psutil

from pyfemtet.opt import OptimizerOptuna, FemtetInterface


FEMTET_EXE_PATH = r'C:\Program Files\Femtet_Ver2023_64bit_inside\Program\Femtet.exe'

here, me = os.path.split(__file__)
record = False


def destruct(femopt):
    femopt.fem.quit()


def lunch_femtet(femprj):
    # 後片付けのために Femtet の pid を取得する
    pid_list_before = [p.pid for p in psutil.process_iter(attrs=["name"]) if p.info["name"] == 'Femtet.exe']

    # Femtet の起動, 普通に起動するのと同じやり方でやらないと ProjectPath が '' になる
    subprocess.Popen([FEMTET_EXE_PATH, os.path.abspath(femprj)], cwd=os.path.dirname(FEMTET_EXE_PATH))

    # pid_list の差から起動した Femtet の pid を取得する
    pid_list_after = [p.pid for p in psutil.process_iter(attrs=["name"]) if p.info["name"] == 'Femtet.exe']
    pid = [x for x in pid_list_after if x not in pid_list_before][-1]

    # 一応 Femtet の起動を待つ
    sleep(5)

    return pid


def max_disp(Femtet):
    return Femtet.Gogh.Galileo.GetMaxDisplacement_py()[1]


def volume(Femtet, femopt):
    d, h, _ = femopt.get_parameter('values')
    w = Femtet.GetVariableValue('w')
    return d * h * w


def bottom_surface(_, femopt):
    d, h, w = femopt.get_parameter('values')
    return d * w


def test_restart_femtet():
    """
    テストしたい状況
        Femtet で一通りの機能が動くか
    結果
        結果が保存したものと一致するか
    """

    femprj = os.path.join(here, f'{me.replace(".py", ".femprj")}')
    csvdata = os.path.join(here, f'{me.replace(".py", ".csvdata")}')
    csv = os.path.join(here, f'{me.replace(".py", "_specified_history.csv")}')

    if os.path.isfile(csv): os.remove(csv)
    if os.path.isfile(f'{csv.replace(".csv", ".db")}'): os.remove(f'{csv.replace(".csv", ".db")}')

    for i in range(3):

        fem = FemtetInterface(femprj, connect_method='new')

        femopt = OptimizerOptuna(fem, history_path=csv)
        femopt.add_parameter('d', 5, 1, 10)
        femopt.add_parameter('h', 5, 1, 10)
        femopt.add_parameter('w', 5, 1, 10)
        femopt.add_objective(max_disp)  # 名前なし目的変数（obj_1 になる）
        femopt.add_objective(volume, 'volume(mm3)', args=femopt)  # 上書き
        femopt.add_constraint(bottom_surface, 'surf<=20', upper_bound=20, args=femopt)

        femopt.set_random_seed(42)
        femopt.main(n_trials=5)

        femopt.terminate_monitor()
        try:
            femopt.fem.quit()
        except:
            pass
    #
    # if record:
    #     # データの保存
    #     os.rename(femopt.history_path, csvdata)
    #
    # else:
    #     # データの取得
    #     ref_df = pd.read_csv(csvdata).replace(np.nan, None)
    #     def_df = pd.read_csv(csv).replace(np.nan, None)
    #
    #     # 並べ替え（並列しているから順番は違いうる）
    #     ref_df = ref_df.iloc[:, 1:].sort_values('d').sort_values('h').sort_values('w').select_dtypes(include='number')
    #     def_df = def_df.iloc[:, 1:].sort_values('d').sort_values('h').sort_values('w').select_dtypes(include='number')
    #
    #     assert np.sum(np.abs(def_df.values - ref_df.values)) < 0.001


if __name__ == '__main__':
    # record = True
    # test_restart_femtet()
    record = False
    test_restart_femtet()


