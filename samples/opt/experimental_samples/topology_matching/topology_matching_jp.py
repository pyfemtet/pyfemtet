"""Topology Matching を用いた最適化

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


3. 事前準備として、下記の手順を実行してください。
- SOLIDWORKS のインストール
- C:\temp フォルダを作成する
- 以下のファイルを同じフォルダに配置
    - topology_matching.py (このファイル)
    - cad_ex01_SW_fillet.SLDPRT
    - cad_ex01_SW_fillet.femprj


Solidworks 連携機能の詳細については下記ページもご覧ください。
https://pyfemtet.readthedocs.io/ja/stable/examples/Sldworks_ex01/Sldworks_ex01.html


[1]
Benjamin Jones, James Noeckel, Milin Kodnongbua, Ilya Baran, and Adriana Schulz. 2023.
B-rep Matching for Collaborating Across CAD Systems.
ACM Trans. Graph. 42, 4, Article 104 (August 2023), 13 pages.
https://doi.org/10.1145/3592125

"""
import os
from dotenv import load_dotenv
here = os.path.dirname(__file__)
load_dotenv(dotenv_path=os.path.join(here, ".env"))

import os
from win32com.client import constants
from pyfemtet.opt import FEMOpt
from pyfemtet.opt.interface import FemtetWithSolidworksInterface
from pyfemtet.opt.interface.beta import FemtetWithSolidworksInterfaceWithTopologyMatching


here, me = os.path.split(__file__)
os.chdir(here)


def von_mises(Femtet):
    """モデルの最大フォン・ミーゼス応力を取得します。"""
    # ミーゼス応力計算
    Gogh = Femtet.Gogh
    Gogh.Galileo.Potential = constants.GALILEO_VON_MISES_C
    succeed, (x, y, z), mises = Gogh.Galileo.GetMAXPotentialPoint_py(constants.CMPX_REAL_C)
    return mises


def mass(Femtet):
    """モデルの質量を取得します。"""
    return Femtet.Gogh.Galileo.GetMass('H_beam')


def C_minus_B(Femtet, opt):
    """C 寸法と B 寸法の差を計算します。"""
    A, B, C = opt.get_parameter('values')
    return C - B


def main():
    # SW-Femtet 連携オブジェクトの初期化
    # この処理により、Python プロセスは Femtet に接続を試みます。
    fem = FemtetWithSolidworksInterfaceWithTopologyMatching(
        femprj_path='cad_ex01_SW_fillet.femprj',
        sldprt_path='cad_ex01_SW_fillet.SLDPRT',
        open_result_with_gui=False,
    )

    # FEMOpt オブジェクトの初期化 (最適化問題とFemtetとの接続を行います)
    femopt = FEMOpt(fem=fem)

    # 設計変数を最適化問題に追加
    femopt.add_parameter('A', 10, lower_bound=1, upper_bound=59)
    femopt.add_parameter('B', 10, lower_bound=1, upper_bound=40)
    femopt.add_parameter('C', 20, lower_bound=5, upper_bound=59)

    # 拘束関数を最適化問題に追加
    femopt.add_constraint(fun=C_minus_B, name='C>B', lower_bound=1, args=(femopt.opt,))

    # 目的関数を最適化問題に追加
    femopt.add_objective(fun=von_mises, name='von Mises (Pa)')
    femopt.add_objective(fun=mass, name='mass (kg)')

    # 最適化を実行
    femopt.set_random_seed(42)
    femopt.optimize(n_trials=20)


if __name__ == '__main__':
    main()
