"""単目的最適化: スプリングバックを加味した曲げ

Femtet の応力解析ソルバを利用して、スプリングバックを考慮した
目標の材料曲げ角度を達成するために必要な曲げ角度を決定します。
※ 弾塑性解析は特別オプション機能です。

対応プロジェクト：gal_ex58_parametric_jp.femprj
"""
import numpy as np
from win32com.client import constants
from optuna.integration.botorch import BoTorchSampler

from pyfemtet.opt import FEMOpt, OptunaOptimizer


def bending(Femtet):
    """材料の曲げ角度を取得します。

    Note:
        目的関数または制約関数は、
        第一引数としてFemtetを受け取り、
        戻り値としてfloat型を返す必要があります。

    Params:
        Femtet: Femtet をマクロで操作するためのインスタンスです。詳細な情報については、「Femtet マクロヘルプ」をご覧ください。

    Returns:
        float: 曲げ。
    """
    Gogh = Femtet.Gogh

    # モードを除荷後に設定
    Gogh.Galileo.Mode = Gogh.Galileo.nMode - 1

    # 計測対象点の変位を取得
    Gogh.Galileo.Vector = constants.GALILEO_DISPLACEMENT_C
    succeed, (x, y, z) = Gogh.Galileo.GetVectorAtPoint_py(200, 0, 0)

    # 曲げ起点 (100, 0) と変形後の点を結ぶ線分が X 軸となす角度を計算
    bending_point = np.array((100, 0))
    bended_point = np.array((200 + 1000 * x.Real, 1000 * z.Real))
    dx, dz = bended_point - bending_point
    degree = np.arctan2(-dz, dx)

    return degree * 360 / (2*np.pi)  # 単位: 度


if __name__ == '__main__':

    # 数値最適化問題の初期化 (最適化手法を決定します)
    opt = OptunaOptimizer(
        sampler_class=BoTorchSampler,
        sampler_kwargs=dict(
            n_startup_trials=3,
        )
    )

    # FEMOpt オブジェクトの初期化 (最適化問題とFemtetとの接続を行います)
    femopt = FEMOpt(opt=opt)

    # 設計変数を最適化問題に追加 (femprj ファイルに登録されている変数を指定してください)
    femopt.add_parameter("rot", 90, lower_bound=80, upper_bound=100)

    # 目的関数を最適化問題に追加
    # 曲げ角度の目標は 90° です。
    femopt.add_objective(bending, name='曲げ角度（度）', direction=90)

    # 最適化を実行
    femopt.set_random_seed(42)
    femopt.optimize(n_trials=10)
