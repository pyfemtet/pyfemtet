"""単目的最適化: 円形パッチアンテナの共振周波数

Femtet の電磁波解析ソルバを利用して、円形パッチアンテナの
電磁波調和解析を行い、共振特性を目標の値にする設計を行います。

対応プロジェクト: her_ex40_parametric_jp.femprj
"""
from time import sleep

import numpy as np
from scipy.signal import find_peaks
from tqdm import tqdm
from optuna.integration.botorch import BoTorchSampler

from pyfemtet.core import SolveError
from pyfemtet.opt import OptunaOptimizer, FEMOpt


class SParameterCalculator:
    """Sパラメータ計算用クラス"""
    
    def __init__(self):
        self.freq = []
        self.S = []
        self.interpolated_function = None
        self.resonance_frequency = None
        self.minimum_S = None

    def _get_freq_and_S_parameter(self, Femtet):
        """周波数とSパラメータの関係を取得します。"""

        Gogh = Femtet.Gogh
        
        freq_list = []
        dB_S_list = []
        for mode in tqdm(range(Gogh.Hertz.nMode), '周波数と S(1, 1) の関係を取得'):
            # 周波数モード設定
            Gogh.Hertz.Mode = mode
            sleep(0.01)

            # 周波数を取得
            freq = Gogh.Hertz.GetFreq().Real

            # S(1, 1) を取得
            comp_S = Gogh.Hertz.GetSMatrix(0, 0)
            norm = np.linalg.norm((comp_S.Real, comp_S.Imag))
            dB_S = 20 * np.log10(norm)

            # 結果を保存
            freq_list.append(freq)
            dB_S_list.append(dB_S)

        self.freq = freq_list
        self.S = dB_S_list

    def _calc_resonance_frequency(self):
        """Sパラメータの第一ピークを与える周波数を取得します。"""
        peaks, _ = find_peaks(-np.array(self.S), height=None, threshold=None, distance=None, prominence=0.5, width=None, wlen=None, rel_height=0.5, plateau_size=None)
        if len(peaks) == 0:
            raise SolveError('S(1,1) のピークを取得できませんでした。')
        self.resonance_frequency = self.freq[peaks[0]]
        self.minimum_S = self.S[peaks[0]]

    def get_resonance_frequency(self, Femtet):
        """パッチアンテナの共振周波数を計算します。

        Note:
            目的関数または制約関数は、
            第一引数としてFemtetを受け取り、
            戻り値としてfloat型を返す必要があります。

        Params:
            Femtet: Femtet をマクロで操作するためのインスタンスです。詳細な情報については、「Femtet マクロヘルプ」をご覧ください。
        
        Returns:
            float: パッチアンテナの共振周波数。
        """
        self._get_freq_and_S_parameter(Femtet)
        self._calc_resonance_frequency()
        return self.resonance_frequency  # 単位: Hz
        

def antenna_is_smaller_than_substrate(Femtet, opt):
    """アンテナの大きさと基板の大きさの関係を計算します。

    この関数は、変数の更新によってモデル形状が破綻しないように
    変数の組み合わせを拘束するために使われます。

    Params:
        Femtet: Femtet をマクロで操作するためのインスタンスです。詳細な情報については、「Femtet マクロヘルプ」をご覧ください。
    
    Returns:
        float: 基板エッジとアンテナエッジの間隙。1 mm 以上が必要です。
    """
    params = opt.get_parameter()
    r = params['antenna_radius']
    w = params['substrate_w']
    return w / 2 - r  # 単位: mm


def port_is_inside_antenna(Femtet, opt):
    """給電ポートの位置とアンテナの大きさの関係を計算します。"""
    params = opt.get_parameter()
    r = params['antenna_radius']
    x = params['port_x']
    return r - x  # 単位: mm。1 mm 以上が必要です。


if __name__ == '__main__':
    # 周波数特性を計算するためのオブジェクトを初期化
    s = SParameterCalculator()

    # 数値最適化問題の初期化 (最適化手法を決定します)
    opt = OptunaOptimizer(
        sampler_class=BoTorchSampler,
        sampler_kwargs=dict(
            n_startup_trials=10,
        )
    )
    
    # FEMOpt オブジェクトの初期化 (最適化問題とFemtetとの接続を行います)
    femopt = FEMOpt(opt=opt)
    
    # 設計変数を最適化問題に追加 (femprj ファイルに登録されている変数を指定してください)
    femopt.add_parameter('antenna_radius', 10, 5, 20)
    femopt.add_parameter('substrate_w', 50, 40, 60)
    femopt.add_parameter('port_x', 5, 1, 20)

    # 拘束関数を最適化問題に追加
    femopt.add_constraint(antenna_is_smaller_than_substrate, 'アンテナと基板エッジの間隙', lower_bound=1, args=(opt,))
    femopt.add_constraint(port_is_inside_antenna, 'アンテナエッジと給電ポートの間隙', lower_bound=1, args=(opt,))

    # 目的関数を最適化問題に追加
    # 共振周波数の目標は 3.0 GHz です。
    femopt.add_objective(s.get_resonance_frequency, '第一共振周波数(Hz)', direction=3.0 * 1e9)

    femopt.set_random_seed(42)
    femopt.optimize(n_trials=15)
