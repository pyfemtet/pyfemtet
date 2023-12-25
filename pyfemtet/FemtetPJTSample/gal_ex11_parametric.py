from pyfemtet.opt import OptimizerOptuna

from win32com.client import constants


def fundamental_resonance(Femtet):
    """mode[0] の共振周波数の取得"""
    Femtet.Gogh.Galileo.Mode = 0
    freq = Femtet.Gogh.Galileo.GetFreq()  # CComplex
    return freq.Real  # Hz


def volume(Femtet):
    return Femtet.Gogh.CalcVolume_py([0])[1]  # m3


def thickness(Femtet):
    external_r = Femtet.GetVariableValue('external_r')
    internal_r = Femtet.GetVariableValue('internal_r')
    # Femtet.Gogh
    return external_r - internal_r
    

if __name__ == '__main__':

    femopt = OptimizerOptuna()

    femopt.add_parameter('internal_r', initial_value=1.1, lower_bound=0.2, upper_bound=3)
    femopt.add_parameter('external_r', initial_value=1.5, lower_bound=0.1, upper_bound=2.9)

    femopt.add_objective(fundamental_resonance, '共振周波数(Hz)', 150)
    femopt.add_objective(volume, '体積(m3)', 1000)
    
    femopt.add_constraint(thickness, '厚さ(m)', lower_bound=0.3)
    
    femopt.set_random_seed(42)
    femopt.main(n_parallel=3, n_trials=50)
