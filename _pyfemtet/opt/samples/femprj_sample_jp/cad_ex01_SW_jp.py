"""外部 CAD (SOLIDWORKS) 連携

Femtet の応力解析ソルバ、および
Dassault Systemes 社製 CAD ソフト SOLIDWORKS を用いて
軽量かつ高剛性な H 型鋼の設計を行います。

事前準備として、下記の手順を実行してください。
- SOLIDWORKS のインストール
- C:\temp フォルダを作成する
    - Note: SOLIDWORKS が .x_t ファイルをこのフォルダに保存します。
- 以下のファイルを同じフォルダに配置
    - cad_ex01_SW_jp.py (このファイル)
    - cad_ex01_SW.SLDPRT
    - cad_ex01_SW_jp.femprj
"""

import os

from win32com.client import constants

from pyfemtet.opt import FEMOpt
from pyfemtet.opt.interface import FemtetWithSolidworksInterface
from pyfemtet.core import ModelError


here, me = os.path.split(__file__)
os.chdir(here)


def von_mises(Femtet):
    """モデルの最大フォン・ミーゼス応力を取得します。

    Note:
        目的関数または制約関数は、
        第一引数としてFemtetを受け取り、
        戻り値としてfloat型を返す必要があります。

    Warning:
        CAD 連携機能では、意図しない位置に境界条件が設定される可能性があります。

        この例では、境界条件が意図したとおりに割り当てられている場合、
        最大変位は常に負になります。最大変位が正の場合、境界条件の割り当てが
        失敗したとみなし、ModelError を送出します。

        最適化中に ModelError、MeshError、または SolveError が発生した場合、
        最適化プロセスは試行を失敗とみなし、次のトライアルにスキップします。
    """

    # 簡易的な境界条件の正しさチェック
    dx, dy, dz = Femtet.Gogh.Galileo.GetMaxDisplacement_py()
    if dz >= 0:
        raise ModelError('境界条件の設定が間違っています。')

    # ミーゼス応力計算
    Gogh = Femtet.Gogh
    Gogh.Galileo.Potential = constants.GALILEO_VON_MISES_C
    succeed, (x, y, z), mises = Gogh.Galileo.GetMAXPotentialPoint_py(constants.CMPX_REAL_C)

    return mises


def mass(Femtet):
    """モデルの質量を取得します。"""
    return Femtet.Gogh.Galileo.GetMass('H_beam')


def C_minus_B(Femtet, opt):
    """C 寸法と B 寸法の差を計算します。

    別の例では、次のスニペットを使用して設計変数にアクセスします。

        A = Femtet.GetVariableValue('A')
    
    ただし、CAD 連携機能を使用する場合、設計変数が .femprj ファイルに
    設定されていないため、この方法は機能しません。

    CAD 連携機能を使用する場合、以下の方法で設計変数にアクセスすることができます。

        # add_parameter() で追加したパラメータの変数名をキーとする辞書を得る方法
        params: dict = opt.get_parameter()
        A = params['A']

    又は

        # add_parameter() で追加した順のパラメータの値の配列を得る方法
        values: np.ndarray = opt.get_parameter('values')
        A, B, C = values

    目的関数と拘束関数は、最初の引数の後に任意の変数を取ることができます。
    FEMOpt のメンバ変数 opt には get_parameter() というメソッドがあります。
    このメソッドによって add_parameter() で追加された設計変数を取得できます。
    opt を第 2 引数として取ることにより、目的関数または拘束関数内で
    get_parameter() を実行して設計変数を取得できます。
    """
    A, B, C = opt.get_parameter('values')
    return C - B


if __name__ == '__main__':

    # NX-Femtet 連携オブジェクトの初期化
    # この処理により、Python プロセスは Femtet に接続を試みます。
    fem = FemtetWithSolidworksInterface(
        sldprt_path='cad_ex01_SW.SLDPRT',
        open_result_with_gui=False,
    )

    # FEMOpt オブジェクトの初期化 (最適化問題とFemtetとの接続を行います)
    femopt = FEMOpt(fem=fem)

    # 設計変数を最適化問題に追加 (femprj ファイルに登録されている変数を指定してください)
    femopt.add_parameter('A', 10, lower_bound=1, upper_bound=59)
    femopt.add_parameter('B', 10, lower_bound=1, upper_bound=40)
    femopt.add_parameter('C', 20, lower_bound=5, upper_bound=59)

    # 拘束関数を最適化問題に追加
    femopt.add_constraint(C_minus_B, 'C>B', lower_bound=1, args=femopt.opt)

    # 目的関数を最適化問題に追加
    femopt.add_objective(von_mises, name='von Mises (Pa)')
    femopt.add_objective(mass, name='mass (kg)')

    # 最適化を実行
    femopt.set_random_seed(42)
    femopt.optimize(n_trials=20)
