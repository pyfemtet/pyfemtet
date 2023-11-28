from time import sleep

import numpy as np
from scipy.signal import find_peaks
from tqdm import tqdm

from pythoncom import com_error

from PyFemtet.opt import FemtetOptuna

from win32com.client import Dispatch
from win32com.client import constants

# S パラメータ・共振周波数計算用クラス
class SParameterCalculator:
    
    def __init__(self):
        self.freq = []
        self.S = []
        self.interpolated_function = None
        self.resonance_frequency = None
        self.minimum_S = None

    def get_result_from_Femtet(self, Femtet):
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
            sleep(0.1) # mode 変更が反映されるまでの猶予時間
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
        # S に対して最初のピークを与える freq を計算
        x = -np.array(self.S)
        peaks, _ = find_peaks(x, height=None, threshold=None, distance=None, prominence=0.5, width=None, wlen=None, rel_height=0.5, plateau_size=None)
        from PyFemtet.opt.core import SolveError
        if len(peaks)==0:
            raise SolveError('ピークが検出されませんでした。')
        self.resonance_frequency = self.freq[peaks[0]]
        self.minimum_S = self.S[peaks[0]]

    def get_resonance_frequency(self, Femtet):
        self.get_result_from_Femtet(Femtet)
        self.calc_resonance_frequency()
        return self.resonance_frequency * 1e-9
        

# サイズ拘束用関数
def anntena_is_smaller_than_substrate(Femtet):
    ant_r = Femtet.GetVariableValue('ant_r')
    Sx = Femtet.GetVariableValue('sx')
    return Sx/2 - ant_r # 1 以上になるようにする

def port_is_inside_anntena(Femtet):
    ant_r = Femtet.GetVariableValue('ant_r')
    xf = Femtet.GetVariableValue('xf')
    return ant_r - xf # 1 以上になるようにする
    
def size(Femtet):
    Sx = Femtet.GetVariableValue('sx')
    return Sx**2
    
if __name__=='__main__':
    # S パラメータ計算用クラス
    s = SParameterCalculator()
    
    # 最適化連携クラス
    FEMOpt = FemtetOptuna() # ここで起動している Femtet が紐づけされます
    
    # 最適化の設定
    FEMOpt.add_parameter('ant_r', 10, 5, 100, memo='円形アンテナの半径')
    FEMOpt.add_parameter('sx', 50, 5, 100, memo='基板のサイズ')
    FEMOpt.add_parameter('xf', 5, 1, 100, memo='給電ポートの偏心量')
    FEMOpt.add_constraint(anntena_is_smaller_than_substrate, lower_bound=1)
    FEMOpt.add_constraint(port_is_inside_anntena, lower_bound=1)
    FEMOpt.add_objective(s.get_resonance_frequency, '第一共振周波数（GHz）', direction=3.5)
    FEMOpt.add_objective(size, '基板サイズ（mm2）')
    
    FEMOpt.set_process_monitor()
    
    FEMOpt.main()
    
    import matplotlib.pyplot as plt
    plt.show()
    
