import os
from time import sleep

from optuna.samplers import RandomSampler

from pyfemtet.opt import FEMOpt, FemtetInterface, OptunaOptimizer


def get_res_freq(Femtet):
    Galileo = Femtet.Gogh.Galileo
    Galileo.Mode = 0
    sleep(0.01)
    return Galileo.GetFreq().Real


if __name__ == '__main__':

    os.chdir(os.path.dirname(__file__))

    # Femtet との接続を行います。
    fem = FemtetInterface(
        femprj_path='gal_ex13_parametric.femprj',
    )

    # 最適化用オブジェクトの設定を行います。
    # ただしこのスクリプトでは最適化ではなく
    # 学習データ作成を行うので、 optuna の
    # ランダムサンプリングクラスを用いて
    # 設計変数の選定を行います。
    opt = OptunaOptimizer(
        sampler_class=RandomSampler,
    )

    # FEMOpt オブジェクトを設定します。
    # 最適化スクリプトで history_path を参照するため、
    # わかりやすい csv ファイル名を指定します。
    femopt = FEMOpt(
        fem=fem,
        opt=opt,
        history_path='training_data.csv'
    )

    # 設計変数を設定します。
    femopt.add_parameter('length', 0.1, 0.02, 0.2)
    femopt.add_parameter('width', 0.01, 0.001, 0.02)
    femopt.add_parameter('base_radius', 0.008, 0.006, 0.01)
    # 目的関数を設定します。ランダムサンプリングなので
    # direction は指定してもサンプリングに影響しません。
    femopt.add_objective(fun=get_res_freq, name='第一共振周波数(Hz)')

    # 学習データ作成を行います。
    # 終了条件を指定しない場合、手動で停止するまで
    # 学習データ作成を続けます。
    femopt.set_random_seed(42)
    femopt.optimize(
        # n_trials=100
    )
