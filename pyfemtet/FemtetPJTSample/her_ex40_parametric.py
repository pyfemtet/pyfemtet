"""Single-objective optimization: Resonant frequency of a circular patch antenna

Using Femtet’s electromagnetic wave analysis solver,
we explain an example of setting the resonant frequency
of a circular patch antenna to a specific value.
"""
from time import sleep

import numpy as np
from scipy.signal import find_peaks
from tqdm import tqdm
from optuna.integration.botorch import BoTorchSampler

from pyfemtet.opt import OptunaOptimizer, FEMOpt


class SParameterCalculator:
    """This class is for calculating S-parameters and resonance frequencies."""
    
    def __init__(self):
        self.freq = []
        self.S = []
        self.interpolated_function = None
        self.resonance_frequency = None
        self.minimum_S = None

    def get_result_from_Femtet(self, Femtet):
        """Obtain the relationship between frequency and S-parameter from the Femtet analysis results."""

        # Preparation
        Femtet.OpenCurrentResult(True)
        Gogh = Femtet.Gogh
        
        # Obtain the frequency and S(1,1) for each mode
        mode = 0
        freq_list = []
        dB_S_list = []
        for mode in tqdm(range(Gogh.Hertz.nMode), 'Obtaining frequency and S-parameter'):
            # Femtet result screen mode settings
            Gogh.Hertz.Mode = mode
            sleep(0.01)
            # Get frequency
            freq = Gogh.Hertz.GetFreq().Real
            # Get S-parameters
            comp_S = Gogh.Hertz.GetSMatrix(0, 0)
            norm = np.linalg.norm((comp_S.Real, comp_S.Imag))
            dB_S = 20 * np.log10(norm)
            # Get results
            freq_list.append(freq)
            dB_S_list.append(dB_S)
        self.freq = freq_list
        self.S = dB_S_list

    def calc_resonance_frequency(self):
        """Compute the frequency that gives the first peak for S-parameter."""
        x = -np.array(self.S)
        peaks, _ = find_peaks(x, height=None, threshold=None, distance=None, prominence=0.5, width=None, wlen=None, rel_height=0.5, plateau_size=None)
        from pyfemtet.core import SolveError
        if len(peaks) == 0:
            raise SolveError('No peaks detected.')
        self.resonance_frequency = self.freq[peaks[0]]
        self.minimum_S = self.S[peaks[0]]

    def get_resonance_frequency(self, Femtet):
        """Calculate the resonant frequency.

        Note:
            The objective or constraint function
            must take a Femtet as its first argument
            and must return a single float.

        Params:
            Femtet: An instance for using Femtet macros. For more information, see "Femtet Macro Help / CFemtet Class".
        
        Returns:
            float: A resonance frequency.
        """
        self.get_result_from_Femtet(Femtet)
        self.calc_resonance_frequency()
        f = self.resonance_frequency * 1e-9
        return f  # GHz
        

def antenna_is_smaller_than_substrate(Femtet):
    """Calculate the relationship between antenna size and board size.

    This function is used to constrain the model
    from breaking down while changing parameters.

    Params:
        Femtet: An instance for using Femtet macros.
    
    Returns:
        float: Difference between the board size and antenna size. Must be equal to or grater than 1 mm.
    """
    ant_r = Femtet.GetVariableValue('ant_r')
    Sx = Femtet.GetVariableValue('sx')
    return Sx/2 - ant_r


def port_is_inside_antenna(Femtet):
    """Calculate the relationship between the feed port location and antenna size.

    This function is used to constrain the model
    from breaking down while changing parameters.

    Params:
        Femtet: An instance for using Femtet macros.
    
    Returns:
        float: Difference between the antenna edge and the position of the feed port. Must be equal to or grater than 1 mm.
    """
    ant_r = Femtet.GetVariableValue('ant_r')
    xf = Femtet.GetVariableValue('xf')
    return ant_r - xf


if __name__ == '__main__':
    # Define the object for calculating S-parameters and resonance frequencies.
    s = SParameterCalculator()

    # Define mathematical optimization object.
    opt = OptunaOptimizer(
        sampler_class=BoTorchSampler,
        sampler_kwargs=dict(
            n_startup_trials=10,
        )
    )
    
    # Define FEMOpt object (This process integrates mathematical optimization and FEM.).
    femopt = FEMOpt(opt=opt)
    
    # Add design variables (Use variable names set in Femtet) to the optimization problem.
    femopt.add_parameter('ant_r', 10, 5, 20)
    femopt.add_parameter('sx', 50, 40, 60)
    femopt.add_parameter('xf', 5, 1, 20)

    # Add constraint to the optimization problem.
    femopt.add_constraint(antenna_is_smaller_than_substrate, 'board_antenna_clearance', lower_bound=1)
    femopt.add_constraint(port_is_inside_antenna, 'antenna_port_clearance', lower_bound=1)

    # Add objective to the optimization problem.
    # The target frequency is 3 GHz.
    femopt.add_objective(s.get_resonance_frequency, 'First_resonant_frequency(GHz)', direction=3.0)

    femopt.set_random_seed(42)
    femopt.optimize(n_trials=20)
    femopt.terminate_all()
