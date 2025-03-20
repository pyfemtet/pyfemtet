"""多目的の最適化: プリント基板上ICの空冷（強制対流）

Femtet の簡易熱流体解析ソルバを利用して、強制対流を受ける
プリント基板上のICチップについて、チップ温度を低減しつつ
基板サイズを小さくする設計を行います。

対応プロジェクト：paswat_ex1_parametric_jp.femprj
"""
from pyfemtet.opt import FEMOpt


def chip_temp(Femtet, chip_name):
    """チップの最高温度を取得します。

    Note:
        目的関数または制約関数は、
        第一引数としてFemtetを受け取り、
        戻り値としてfloat型を返す必要があります。

    Params:
        Femtet: Femtet をマクロで操作するためのインスタンスです。詳細な情報については、「Femtet マクロヘルプ」をご覧ください。
        chip_name (str): femprj 内で定義されているボディ属性名です。有効な値は 'MAINCHIP' 又は 'SUBCHIP' です。

    Returns:
        float: 指定されたボディ属性名のボディの最高温度です。
    """
    Gogh = Femtet.Gogh

    max_temperature, min_temperature, mean_temperature = Gogh.Watt.GetTemp(chip_name)

    return max_temperature  # 単位: 度


def substrate_size(Femtet):
    """基板のXY平面上での専有面積を計算します。"""
    substrate_w = Femtet.GetVariableValue('substrate_w')
    substrate_d = Femtet.GetVariableValue('substrate_d')
    return substrate_w * substrate_d  # 単位: mm2


if __name__ == '__main__':

    # FEMOpt オブジェクトの初期化 (最適化問題とFemtetとの接続を行います)
    femopt = FEMOpt()

    # 設計変数を最適化問題に追加 (femprj ファイルに登録されている変数を指定してください)
    femopt.add_parameter("substrate_w", 40, lower_bound=22, upper_bound=60)
    femopt.add_parameter("substrate_d", 60, lower_bound=34, upper_bound=60)
    femopt.add_parameter("rot", 0, lower_bound=0, upper_bound=180)

    # 目的関数を最適化問題に追加
    femopt.add_objective(chip_temp, name='MAINCHIP<br>最高温度（度）', direction='minimize', args=('MAINCHIP',))
    femopt.add_objective(chip_temp, name='SUBCHIP<br>最高温度（度）', direction='minimize', args=('SUBCHIP',))
    femopt.add_objective(substrate_size, name='基板サイズ（mm2）', direction='minimize')

    # 最適化を実行
    femopt.set_random_seed(42)
    # femopt.optimize(n_trials=15)
    femopt.optimize(n_trials=30, n_parallel=3)  # 並列計算しない場合との差はこの行のみです。
