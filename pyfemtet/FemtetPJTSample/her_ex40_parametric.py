from time import sleep

import numpy as np
from scipy.signal import find_peaks
from tqdm import tqdm
from optuna.integration.botorch import BoTorchSampler

from pyfemtet.opt import OptunaOptimizer, FEMOpt



class SParameterCalculator:
    """S パラメータ・共振周波数計算用クラスです。"""
    
    def __init__(self):
        self.freq = []
        self.S = []
        self.interpolated_function = None
        self.resonance_frequency = None
        self.minimum_S = None

    def get_result_from_Femtet(self, Femtet):
        """Femtet の解析結果から周波数と S パラメータの関係を取得します。"""

        # 前準備
        Femtet.OpenCurrentResult(True)
        Gogh = Femtet.Gogh
        
        # 各モードに対して周波数と S(1,1) を取得する
        mode = 0
        freq_list = []
        dB_S_list = []
        for mode in tqdm(range(Gogh.Hertz.nMode), '結果取得中'):
            # Femtet 結果画面のモード設定
            Gogh.Hertz.Mode = mode
            sleep(0.01)
            # 周波数の取得
            freq = Gogh.Hertz.GetFreq().Real
            # S パラメータの取得
            comp_S = Gogh.Hertz.GetSMatrix(0, 0)
            norm = np.linalg.norm((comp_S.Real, comp_S.Imag))
            dB_S = 20 * np.log10(norm)
            # 結果の取得
            freq_list.append(freq)
            dB_S_list.append(dB_S)
        self.freq = freq_list
        self.S = dB_S_list

    def calc_resonance_frequency(self):
        """S に対して最初のピークを与える freq を計算します。"""
        x = -np.array(self.S)
        peaks, _ = find_peaks(x, height=None, threshold=None, distance=None, prominence=0.5, width=None, wlen=None, rel_height=0.5, plateau_size=None)
        from pyfemtet.core import SolveError
        if len(peaks) == 0:
            raise SolveError('ピークが検出されませんでした。')
        self.resonance_frequency = self.freq[peaks[0]]
        self.minimum_S = self.S[peaks[0]]

    def get_resonance_frequency(self, Femtet):
        """共振周波数を計算します。

        Femtet : マクロを使用するためのインスタンスです。詳しくは "Femtet マクロヘルプ / CFemtet クラス" をご覧ください。
            目的関数は第一引数に Femtet インスタンスを取る必要があります。

        f : 計算された共振周波数です。
            目的関数は単一の float を返す必要があります。

        """
        self.get_result_from_Femtet(Femtet)
        self.calc_resonance_frequency()
        f = self.resonance_frequency * 1e-9
        return f  # GHz
        

def antenna_is_smaller_than_substrate(Femtet):
    """アンテナのサイズと基板のサイズの関係を計算します。

    Femtet : マクロを使用するためのインスタンスです。詳しくは "Femtet マクロヘルプ / CFemtet クラス" をご覧ください。
        拘束関数は第一引数に Femtet インスタンスを取る必要があります。

    Sx/2 - ant_r : 基板サイズとアンテナサイズの差です。
        拘束関数は単一の float を返す必要があります。

    """
    ant_r = Femtet.GetVariableValue('ant_r')
    Sx = Femtet.GetVariableValue('sx')
    return Sx/2 - ant_r


def port_is_inside_antenna(Femtet):
    """給電ポートの位置とアンテナのサイズの関係を計算します。

    Femtet : マクロを使用するためのインスタンスです。詳しくは "Femtet マクロヘルプ / CFemtet クラス" をご覧ください。
        拘束関数は第一引数に Femtet インスタンスを取る必要があります。

    ant_r - xf : アンテナ辺縁と給電ポートの位置の差です。
        拘束関数は単一の float を返す必要があります。

    """
    ant_r = Femtet.GetVariableValue('ant_r')
    xf = Femtet.GetVariableValue('xf')
    return ant_r - xf


if __name__ == '__main__':
    # S パラメータ計算用クラス
    s = SParameterCalculator()

    # 最適化手法を定義するオブジェクトを用意
    opt = OptunaOptimizer(
        sampler_class=BoTorchSampler,
        sampler_kwargs=dict(
            n_startup_trials=5,
        )
    )
    
    # 最適化処理を行うオブジェクトを用意
    femopt = FEMOpt(
        opt=opt,
    )  # ここで起動している Femtet が紐づけされます
    
    # 設計変数の設定
    femopt.add_parameter('ant_r', 10, 5, 20, memo='円形アンテナの半径')
    femopt.add_parameter('sx', 50, 40, 60, memo='基板のサイズ')
    femopt.add_parameter('xf', 5, 1, 20, memo='給電ポートの偏心量')

    # 拘束の設定
    femopt.add_constraint(antenna_is_smaller_than_substrate, 'アンテナサイズ', lower_bound=1)
    femopt.add_constraint(port_is_inside_antenna, 'ポート位置', lower_bound=1)

    # 目的の設定
    femopt.add_objective(s.get_resonance_frequency, '第一共振周波数（GHz）', direction=3.0)

    femopt.set_random_seed(42)
    femopt.main(n_trials=20)
    femopt.terminate_all()
