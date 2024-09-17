"""拘束付き最適化を実装するサンプル。

このセクションでは、拘束の種類と、拘束を必要とするモデルで
最適化を実行する手順について説明します。

"""

from optuna_integration import BoTorchSampler
from pyfemtet.opt import FEMOpt, OptunaOptimizer


def mises_stress(Femtet):
    """フォンミーゼス応力を目的関数として計算します。

    この関数は、最適化の実行中に FEMOpt オブジェクトによって
    自動的に呼び出されます。

    引数:
        Femtet: PyFemtet を使用して目的関数または拘束関数を
            定義する場合、最初の引数は Femtet インスタンスを
            取る必要があります。

    戻り値:
        float: 目的または拘束関数は単一の float を返すよう定義してください。
    """
    return Femtet.Gogh.Galileo.GetMaxStress_py()[2]


def radius_diff(Femtet, opt):
    """パイプの外側の半径と内側の半径の差を計算します。

    この拘束は、最適化の実行中にパイプの内側の半径が
    外側の半径を超えないようにするために呼び出されます。

    注意:
        OptunaOptimizer の BoTorchSampler を使用していて、
        strict な拘束を使用する場合、パラメータを提案するため
        に繰り返し計算が必要になるため、Femtet へのアクセスが
        非常に遅くなる可能性があることに注意してください。
        この関数の例のように、Femtet にアクセスするのではなく、
        Optimizer オブジェクトを介してパラメータを取得して計算
        を実行することをお勧めします。

        非推奨::

            p = Femtet.GetVariableValue('p')

        代わりに::

            params = opt.get_parameter()
            p = params['p']

    引数:
        Femtet: PyFemtet を使用して目的関数または拘束関数を
            定義する場合、最初の引数は Femtet インスタンスを
            取る必要があります。
        opt: このオブジェクトを使用すると、Femtet を経由せず
            に外側の半径と内側の半径の値を取得できます。
    """
    params = opt.get_parameter()
    internal_r = params['internal_r']
    external_r = params['external_r']
    return external_r - internal_r


if __name__ == '__main__':
    # 最適化手法のセットアップ
    opt = OptunaOptimizer(
        sampler_class=BoTorchSampler,
        sampler_kwargs=dict(
            n_startup_trials=3,  # 最初の 3 回はランダムサンプリングを行います。
        )
    )
    femopt = FEMOpt(opt=opt)

    # 変数の追加
    femopt.add_parameter("external_r", 10, lower_bound=0.1, upper_bound=10)
    femopt.add_parameter("internal_r", 5, lower_bound=0.1, upper_bound=10)

    # 最適化の実行中に外側の半径を超えないように strict 拘束を追加します。
    femopt.add_constraint(
        radius_diff,  # 拘束関数 (ここでは 外半径 - 内半径).
        name='管厚さ',  # 拘束関数にはプログラム上の名前とは別に自由な名前を付与できます.
        lower_bound=1,  # 拘束関数の下限 (ここでは管の厚みを最低 1 とする).
        args=(femopt.opt,)  # 拘束関数に渡される、Femtet 以外の追加の引数.
    )

    # 目的関数の追加
    femopt.add_objective(mises_stress, name='ミーゼス応力')

    # 最適化の実行
    femopt.set_random_seed(42)
    femopt.optimize(n_trials=10)
