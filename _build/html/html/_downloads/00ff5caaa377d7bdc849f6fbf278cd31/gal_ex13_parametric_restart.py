import os
from time import sleep

from optuna.samplers import RandomSampler, NSGAIISampler, GPSampler, BaseSampler

from pyfemtet.opt import FEMOpt, FemtetInterface, OptunaOptimizer

os.chdir(os.path.dirname(__file__))


def get_res_freq(Femtet):
    Galileo = Femtet.Gogh.Galileo
    Galileo.Mode = 0
    sleep(0.01)
    return Galileo.GetFreq().Real


def main(n_trials, sampler_class: type[BaseSampler], sampler_kwargs: dict):
    """メイン関数

    このサンプルは最適化を途中で中断して続きから行う場合で、
    各リスタートで異なるアルゴリズムを使用して
    最適化を再開する方法を示しています。

    このメイン関数は n_trials と sampler_class を与えると
    その回数、アルゴリズムに応じて最適化を行います。

    Args:

        n_trials (int):
            最適化を終了するために必要な追加の成功した試行回数。

        sampler_class (type[optuna.samplers.BaseSampler]):
            使用するアルゴリズム。
        
        sampler_kwargs (dict):
            アルゴリズムの引数。

    """


    # Femtet に接続します。
    fem = FemtetInterface(
        femprj_path='gal_ex13_parametric.femprj',
    )

    # 最適化オブジェクトを初期化します。
    opt = OptunaOptimizer(
        sampler_class=sampler_class,
        sampler_kwargs=sampler_kwargs,
    )

    # リスタートするためには以前の最適化の履歴を
    # 新しい最適化プログラムに知らせる必要があります。
    # FEMOpt の `history_path` 引数に csv を指定すると、
    # それが存在しない場合、新しい csv ファイルを作り、
    # それが存在する場合、その csv ファイルの続きから
    # 最適化を実行します。
    #
    # 注意:
    #   リスタートする場合、変数の数と名前、目的関数の数と
    #   名前、および拘束関数の数と名前が一貫している必要が
    #   あります。
    #   ただし、変数の上下限や目的関数の方向、拘束関数の内容
    #   などは変更できます。
    #
    # 注意:
    #   OptunaOptimizer を使用する場合、csv と同名の
    #   .db ファイル (ここでは restarting-sample.db) が
    #   csv ファイルと同じフォルダにある必要があります。
    femopt = FEMOpt(
        fem=fem,
        opt=opt,
        history_path='restarting-sample.csv'
    )

    # 設計パラメータを指定します。
    femopt.add_parameter('length', 0.1, 0.02, 0.2)
    femopt.add_parameter('width', 0.01, 0.001, 0.02)
    femopt.add_parameter('base_radius', 0.008, 0.006, 0.01)

    # 目的関数を指定します。
    femopt.add_objective(fun=get_res_freq, name='First Resonant Frequency (Hz)', direction=800)

    # 最適化を実行します。
    femopt.set_random_seed(42)
    femopt.optimize(n_trials=n_trials, confirm_before_exit=False)


if __name__ == '__main__':
    # 最初に、RandomSampler を使用して 3 回計算を行います。
    main(3, RandomSampler, {})

    # 次に、NSGAIISampler を使用して 3 回計算を行います。
    main(3, NSGAIISampler, {})

    #  最後に、GPSampler を使用して 3 回計算を行います。
    main(3, GPSampler, {'n_startup_trials': 0, 'deterministic_objective': True})

    # このプログラム終了後、
    # restarting-sample.csv と同名の .db ファイルを用いて
    # さらに続きの最適化を行うことができます。
