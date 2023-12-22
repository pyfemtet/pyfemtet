# -*- coding: utf-8 -*-
"""
対応する解析モデル：pas_ex1_parametric.femprj
円柱状の障害物を有する円筒管内を流れる空気についての簡易流体解析です。
設計パラメータは以下の通りです。
r：障害物の半径
h：障害物の長さ
p：空気の速度ポテンシャル
流量を特定の値にするために最適な r, h, p を求めます。
"""
import os
from pyfemtet.opt import FemtetInterface, OptimizerOptuna



# os.chdir(os.path.dirname(__file__))


# 目的関数（流量）の定義
def flow(Femtet):
    """pas_ex1_parametric.femprj の結果から流量を取得します。

    目的関数または拘束関数内で constants を用いる場合、
    その関数内に import 文を記述する必要があります。
    """
    from win32com.client import constants
    Gogh = Femtet.Gogh
    Gogh.Pascal.Vector = constants.PASCAL_VELOCITY_C
    _, ret = Gogh.SimpleIntegralVectorAtFace_py([2], [0], constants.PART_VEC_Y_PART_C)
    flow = ret.Real
    return flow


if __name__ == '__main__':

    # Femtet との接続を行います。
    fem = FemtetInterface(os.path.join(os.path.dirname(__file__), 'pas_ex1_parametric.femprj'), connect_method='auto')

    # 最適化用オブジェクトに Femtet インターフェースを関連付けます。
    femopt = OptimizerOptuna(fem)

    # 最適化の設定を行います。
    femopt.add_parameter('r', lower_bound=1, upper_bound=99)
    femopt.add_parameter('h', lower_bound=1, upper_bound=290)
    femopt.add_parameter('p', lower_bound=0.01, upper_bound=1)
    femopt.add_objective(flow, direction=0.05, name='流量(m3/s)')
    femopt.add_objective(flow, direction=0.05, name='流量(m3/s)')

    # 最適化の実行
    # femopt.set_random_seed(42)
    opt = femopt.main(n_trials=30, n_parallel=3, method='TPE', use_lhs_init=False)
