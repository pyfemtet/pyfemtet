"""単目的最適化: 有限長ヘリカルコイルの自己インダクタンス

Femtet の磁場解析ソルバを利用して、
有限長ヘリカルコイルの自己インダクタンスを
目標の値にする設計を行います。

対応プロジェクト: gau_ex08_parametric_jp.femprj
"""
from optuna.integration.botorch import BoTorchSampler
from pyfemtet.opt import FEMOpt, OptunaOptimizer


def inductance(Femtet):
    """自己インダクタンスを取得します。

    Note:
        目的関数または制約関数は、
        第一引数としてFemtetを受け取り、
        戻り値としてfloat型を返す必要があります。

    Params:
        Femtet: Femtet をマクロで操作するためのインスタンスです。詳細な情報については、「Femtet マクロヘルプ」をご覧ください。

    Returns:
        float: 自己インダクタンスです。
    """
    Gogh = Femtet.Gogh

    coil_name = Gogh.Gauss.GetCoilList()[0]
    return Gogh.Gauss.GetL(coil_name, coil_name)  # 単位: F


if __name__ == '__main__':

    # 数値最適化問題の初期化 (最適化手法を決定します)
    opt = OptunaOptimizer(
        sampler_class=BoTorchSampler,
        sampler_kwargs=dict(
            n_startup_trials=5,
        )
    )

    # FEMOpt オブジェクトの初期化 (最適化問題とFemtetとの接続を行います)。
    femopt = FEMOpt(opt=opt)

    # 設計変数を最適化問題に追加 (femprj ファイルに登録されている変数を指定してください)。
    femopt.add_parameter("helical_pitch", 6, lower_bound=4.2, upper_bound=8)
    femopt.add_parameter("coil_radius", 10, lower_bound=1, upper_bound=10)
    femopt.add_parameter("n_turns", 5, lower_bound=1, upper_bound=5)

    # 目的関数を最適化問題に追加
    # 目標の自己インダクタンスは 0.1 μF です。
    femopt.add_objective(inductance, name='自己インダクタンス (F)', direction=1e-7)

    # 最適化を実行
    femopt.set_random_seed(42)
    femopt.optimize(n_trials=20)
