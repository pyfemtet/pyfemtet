"""外部 CAD ソフト NX との連携

外部 CAD ソフト NX で作成したパラメトリックモデルを
Femtet にインポートして最適化を行います。

事前準備として、以下の作業を行ってください。
    - NX_ex01.prt と NX_ex01.femprj を同じフォルダに配置する
    - C:\temp フォルダを作成する
        ※ 最適化の実行中に NX から .x_t ファイルがエクスポートされます。
"""

import os
from pyfemtet.opt import FEMOpt, OptunaOptimizer
from pyfemtet.opt.interface import FemtetWithNXInterface


here, me = os.path.split(__file__)
os.chdir(here)


def disp(Femtet):
    """解析結果から Z 方向最大変位を取得します。
    目的関数の第一引数は Femtet インスタンスである必要があります。
    """
    _, _, ret = Femtet.Gogh.Galileo.GetMaxDisplacement_py()
    return ret


def volume(Femtet):
    """解析結果からモデル体積を取得します。
    目的関数の第一引数は Femtet インスタンスである必要があります。
    """
    _, ret = Femtet.Gogh.CalcVolume_py([0])
    return ret


def C_minus_B(_, opt):
    """C 寸法と B 寸法の差を計算します。

    拘束関数の第一引数は Femtet インスタンスである必要がありますが、
    この例では使用していません。

    ほかの例では設計変数にアクセスする際以下のスニペットを用いますが、
    CAD 連携を行う場合、Femtet に設計変数が設定されていないためです。
        A = Femtet.GetVariableValue('A')

    pyfemtet.opt では目的関数・拘束関数の第二引数以降に任意の変数を設定できます。
    この例では、第二引数に FEMOpt のメンバー変数 opt を取り、
    そのメソッド get_parameter() を用いて設計変数を取得しています。

    """
    A, B, C = opt.get_parameter('values')
    return C - B


if __name__ == '__main__':

    # NX-Femtet 連携オブジェクトを用意
    # ここで起動している Femtet と紐づけがされます。
    fem = FemtetWithNXInterface(
        prt_path='NX_ex01.prt',
        femprj_path='NX_ex01.femprj',
    )

    opt = OptunaOptimizer(
        sampler_kwargs=dict(
            n_startup_trials=5,
        )
    )

    # 最適化処理を行うオブジェクトに連携オブジェクトを紐づけ
    femopt = FEMOpt(fem=fem, opt=opt)

    # 設計変数の設定
    femopt.add_parameter('A', 10, lower_bound=1, upper_bound=59)
    femopt.add_parameter('B', 10, lower_bound=1, upper_bound=40)
    femopt.add_parameter('C', 20, lower_bound=5, upper_bound=59)

    # 拘束関数の設定
    femopt.add_constraint(C_minus_B, 'C>B', lower_bound=1, args=femopt.opt)

    # 目的関数の設定
    femopt.add_objective(disp, name='変位', direction=0)
    femopt.add_objective(volume, name='体積', direction='minimize')

    # 最適化の実行
    femopt.set_random_seed(42)
    femopt.main(n_trials=20)
    femopt.terminate_all()
