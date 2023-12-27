"""鐘の応力調和解析

gal_ex11_parametric.femprj に対し応力調和解析を行い、
第一共振振動数を特定の値にする鐘の寸法を探索します。
"""

from pyfemtet.opt import OptimizerOptuna


def fundamental_resonance(Femtet):
    """Femmtet の解析結果から共振周波数を取得します。

    Femtet : マクロを使用するためのインスタンスです。詳しくは "Femtet マクロヘルプ / CFemtet クラス" をご覧ください。
        目的関数は第一引数に Femtet インスタンスを取る必要があります。

    freq : 計算された共振周波数（複素数）です。Real プロパティにより実数に変換します。
        目的関数は単一の float を返す必要があります。

    """
    Femtet.Gogh.Galileo.Mode = 0
    freq = Femtet.Gogh.Galileo.GetFreq()  # CComplex
    return freq.Real  # Hz


def volume(Femtet):
    """Femmtet の解析結果からモデル体積を取得します。

    Femtet : マクロを使用するためのインスタンスです。詳しくは "Femtet マクロヘルプ / CFemtet クラス" をご覧ください。
        目的関数は第一引数に Femtet インスタンスを取る必要があります。

    v : 計算されたモデル体積です。
        目的関数は単一の float を返す必要があります。

    """
    v = Femtet.Gogh.CalcVolume_py([0])[1]  # m3
    return v


def thickness(Femtet):
    """変数の組み合わせから鐘の厚みを計算します。

    Femtet : マクロを使用するためのインスタンスです。詳しくは "Femtet マクロヘルプ / CFemtet クラス" をご覧ください。
        拘束関数は第一引数に Femtet インスタンスを取る必要があります。

    t : 計算された鐘モデルの厚みです。
        拘束関数は単一の float を返す必要があります。

    """
    external_r = Femtet.GetVariableValue('external_r')
    internal_r = Femtet.GetVariableValue('internal_r')
    t = external_r - internal_r
    return t
    

if __name__ == '__main__':

    femopt = OptimizerOptuna()

    femopt.add_parameter('internal_r', initial_value=1.1, lower_bound=0.2, upper_bound=3)
    femopt.add_parameter('external_r', initial_value=1.5, lower_bound=0.1, upper_bound=2.9)

    femopt.add_objective(fundamental_resonance, '共振周波数(Hz)', 150)
    femopt.add_objective(volume, '体積(m3)', 1000)
    
    femopt.add_constraint(thickness, '厚さ(m)', lower_bound=0.3)
    
    femopt.set_random_seed(42)
    femopt.main(n_parallel=3, n_trials=50)
