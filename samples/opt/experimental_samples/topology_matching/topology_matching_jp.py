r"""Topology Matching を用いた最適化

最適化では設計パラメータを変化させて
モデル形状を更新します。

このとき、モデルの作り方によっては
CAD 内部のトポロジー番号が変わってしまい
境界条件やメッシュサイズの割り当てが
意図しない状態になる問題が知られています。

Femtet 及び PyFemtet では 論文[1] の技術を応用した
Topology Matching 機能を実験的に実装しました。

本サンプルでは、従来の方法では境界条件が壊れてしまう問題に対し
Topology Matching を用いて最適化を行うデモを行います。


# 制限

本機能は、ボディ数が 1 つのみのモデルに対応しています。


# 前提条件

1. Femtet 2025.0.2 以降が必要です。

2. Topology Matching の利用には追加のモジュールが必要です。
下記コマンドでモジュールをインストールしてください。
(MIT ライセンスで公開されているライブラリ `brepmatching` がインストールされます)

    py -m pip install -U pyfemtet[matching]

    または

    py -m pip install -U brepmatching

[1]
Benjamin Jones, James Noeckel, Milin Kodnongbua, Ilya Baran, and Adriana Schulz. 2023.
B-rep Matching for Collaborating Across CAD Systems.
ACM Trans. Graph. 42, 4, Article 104 (August 2023), 13 pages.
https://doi.org/10.1145/3592125

"""
import os
from time import sleep
import numpy as np
from optuna.samplers import BruteForceSampler
from win32com.client import Dispatch, constants
from pyfemtet.opt import FEMOpt
from pyfemtet.opt.optimizer import OptunaOptimizer

# Topology matching を行いながら Femtet で
# 最適化を行う為の機能のインポート
from pyfemtet.opt.interface.beta import FemtetWithTopologyMatching


here = os.path.dirname(__file__)


def calc_angle(Femtet):
    """Femtet の解析結果から金型プレートの傾きを計算する関数"""

    cx = Femtet.GetVariableValue('cx')
    cy = Femtet.GetVariableValue('cy')
    cl2 = Femtet.GetVariableValue('cl2')

    Femtet = Dispatch('FemtetMacro.Femtet')
    Femtet.OpenCurrentResult(True)
    Femtet.Gogh.Activate()

    point1 = Dispatch('FemtetMacro.GaudiPoint')
    point1.X = cx
    point1.Y = cy
    point1.Z = cl2
    point2 = Dispatch('FemtetMacro.GaudiPoint')
    point2.X = 0.
    point2.Y = 0.
    point2.Z = cl2

    Femtet.Gogh.Galileo.Vector = constants.GALILEO_DISPLACEMENT_C
    Femtet.Gogh.Galileo.Part = constants.PART_VEC_C
    sleep(0.1)
    succeeded, ret = Femtet.Gogh.Galileo.MultiGetVectorAtPoint_py((point1, point2,))

    (cmplx_x1, cmplx_y1, cmplx_z1), (cmplx_x2, cmplx_y2, cmplx_z2) = ret
    x1 = cmplx_x1.Real
    y1 = cmplx_y1.Real
    z1 = cmplx_z1.Real
    x2 = cmplx_x2.Real
    y2 = cmplx_y2.Real
    z2 = cmplx_z2.Real

    dz = (point2.Z + z2) - (point1.Z + z1)
    dxy = np.sqrt(((point2.X + x2) - (point1.X + x1)) ** 2 + ((point2.Y + y2) - (point1.Y + y1)) ** 2)

    angle = dz / dxy

    return angle * 1000000


def main():
    # Topology matching を行いながら Femtet で
    # 最適化を行う為の機能の初期化
    fem = FemtetWithTopologyMatching(
        femprj_path=os.path.join(here, 'topology_matching.femprj'),
        model_name='model quarter',
    )

    # 以下、最適化問題のセットアップ
    opt = OptunaOptimizer(
        sampler_class=BruteForceSampler
    )

    femopt = FEMOpt(fem=fem, opt=opt,)

    femopt.add_parameter('w', 80, 70, 90, step=10)
    femopt.add_parameter('d', 65, 60, 70, step=5)

    femopt.add_objective('angle', calc_angle)

    femopt.set_random_seed(42)

    femopt.optimize(
        confirm_before_exit=False,
    )


if __name__ == '__main__':
    main()
