"""基板上の発熱体

wat_ex14_parametric.femprj に対し熱伝導解析を行い、
チップの発熱による温度上昇を最小にしつつ
基板寸法を最小にする
基板寸法・チップ配置寸法を探索します。
"""

from pyfemtet.opt import FEMOpt


def max_temperature(Femtet, body_name):
    """Femtet の解析結果からチップの最高温度を取得します。

    Femtet : マクロを使用するためのインスタンスです。詳しくは "Femtet マクロヘルプ / CFemtet クラス" をご覧ください。
        目的関数は第一引数に Femtet インスタンスを取る必要があります。

    max_temp : 計算された最高温度です。
        目的関数は単一の float を返す必要があります。

    """
    Gogh = Femtet.Gogh

    temp, _, _ = Gogh.Watt.GetTemp_py(body_name)

    return temp  # degree


def substrate_size(Femtet):
    """Femtet の設計変数から基板サイズを取得します。

    Femtet : マクロを使用するためのインスタンスです。詳しくは "Femtet マクロヘルプ / CFemtet クラス" をご覧ください。
        目的関数は第一引数に Femtet インスタンスを取る必要があります。

    subs_w * subs_d : XY 平面における基板の専有面積です。
        目的関数は単一の float を返す必要があります。

    """
    subs_w = Femtet.GetVariableValue('substrate_w')
    subs_d = Femtet.GetVariableValue('substrate_d')

    return subs_w * subs_d  # mm2


if __name__ == '__main__':

    # 最適化処理を行うオブジェクトを用意
    femopt = FEMOpt()

    # 設計変数の設定
    femopt.add_parameter("substrate_w", 40, lower_bound=22, upper_bound=40, memo='基板サイズ X')
    femopt.add_parameter("substrate_d", 60, lower_bound=33, upper_bound=60, memo='基板サイズ Y')

    # 目的関数の設定
    femopt.add_objective(max_temperature, name='メインチップ温度', args=('MAINCHIP',))
    femopt.add_objective(max_temperature, name='サブチップ温度', args=('SUBCHIP',))
    femopt.add_objective(substrate_size, name='基板サイズ')

    # 最適化の実行
    femopt.set_random_seed(42)
    femopt.main(n_trials=20)
    femopt.terminate_all()
