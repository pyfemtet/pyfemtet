"""有限長ソレノイドコイルの自己インダクタンス

gau_ex08_parametric.femprj に対し磁場解析を行い、
自己インダクタンスを特定の値にする
有限長さソレノイドコイルの寸法を探索します。
"""

from pyfemtet.opt import OptimizerOptuna


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

    # 最適化処理を行うオブジェクトを用意
    femopt = OptimizerOptuna()

    # 設計変数の登録
    femopt.add_parameter("h", 3, lower_bound=1.5, upper_bound=6, memo='1巻きピッチ')
    femopt.add_parameter("r", 5, lower_bound=3, upper_bound=12, memo='コイル半径')
    femopt.add_parameter("n", 5, lower_bound=1, upper_bound=20, memo='コイル巻き数')

    # インダクタンスが 0.44 uF に近づくようにゴールを設定
    femopt.add_objective(
        inductance, name='自己インダクタンス', direction=4.4e-07
        )

    # 最適化の実行
    femopt.main(n_trials=30, method='botorch', n_parallel=2)
