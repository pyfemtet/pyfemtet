"""有限長ソレノイドコイルの自己インダクタンス

gau_ex08_parametric.femprj に対し磁場解析を行い、
自己インダクタンスを特定の値にする
有限長さソレノイドコイルの寸法を探索します。
"""
from optuna.integration.botorch import BoTorchSampler
from pyfemtet.opt import FEMOpt, OptunaOptimizer


def inductance(Femtet):
    """Femtet の解析結果から自己インダクタンスを取得します。

    Femtet : マクロを使用するためのインスタンスです。詳しくは "Femtet マクロヘルプ / CFemtet クラス" をご覧ください。
        目的関数は第一引数に Femtet インスタンスを取る必要があります。

    l : 計算された自己インダクタンスです。
        目的関数は単一の float を返す必要があります。

    """
    Gogh = Femtet.Gogh

    # インダクタンスの取得
    cName = Gogh.Gauss.GetCoilList()[0]
    l = Gogh.Gauss.GetL(cName, cName)
    return l  # F


if __name__ == '__main__':

    # 最適化手法を定義するオブジェクトを用意
    opt = OptunaOptimizer(
        sampler_class=BoTorchSampler,
        sampler_kwargs=dict(
            n_startup_trials=5,
        )
    )

    # 最適化処理を行うオブジェクトを用意
    femopt = FEMOpt(opt=opt)  # ここで起動している Femtet が紐づけされます

    # 設計変数の登録
    femopt.add_parameter("h", 3, lower_bound=1.5, upper_bound=6, memo='1巻きピッチ')
    femopt.add_parameter("r", 5, lower_bound=1, upper_bound=10, memo='コイル半径')
    femopt.add_parameter("n", 3, lower_bound=1, upper_bound=5, memo='コイル巻き数')

    # インダクタンスが 0.1 uF に近づくようにゴールを設定
    femopt.add_objective(
        inductance, name='自己インダクタンス', direction=0.1e-06
        )

    # 最適化の実行
    femopt.set_random_seed(42)
    femopt.main(n_trials=20)
    femopt.terminate_all()
