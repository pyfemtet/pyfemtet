"""Single-objective optimization: Resonant frequency of a circular patch antenna

Using Femtet’s electromagnetic wave analysis solver,
we explain an example of setting the resonant frequency
of a circular patch antenna to a specific value.

Corresponding project: her_ex40_parametric.femprj
"""
from time import sleep

import numpy as np
from scipy.signal import find_peaks
from tqdm import tqdm
from optuna.integration.botorch import BoTorchSampler

from pyfemtet.core import SolveError
from pyfemtet.opt import OptunaOptimizer, FEMOpt


class SParameterCalculator:
    """Calculating S-parameters and resonance frequencies."""
    
    def __init__(self):
        self.freq = []
        self.S = []
        self.interpolated_function = None
        self.resonance_frequency = None
        self.minimum_S = None

    def _get_freq_and_S_parameter(self, Femtet):
        """Obtain the relationship between frequency and S-parameter"""

        Gogh = Femtet.Gogh
        
        freq_list = []
        dB_S_list = []
        for mode in tqdm(range(Gogh.Hertz.nMode), 'Obtain frequency and S-parameter.'):
            # mode setting
            Gogh.Hertz.Mode = mode
            sleep(0.01)

            # Obtain frequency
            freq = Gogh.Hertz.GetFreq().Real

            # Obtain S(1, 1)
            comp_S = Gogh.Hertz.GetSMatrix(0, 0)
            norm = np.linalg.norm((comp_S.Real, comp_S.Imag))
            dB_S = 20 * np.log10(norm)

            # Save them
            freq_list.append(freq)
            dB_S_list.append(dB_S)

        self.freq = freq_list
        self.S = dB_S_list

    def _calc_resonance_frequency(self):
        """Compute the frequency that gives the first peak for S-parameter."""
        peaks, _ = find_peaks(-np.array(self.S), height=None, threshold=None, distance=None, prominence=0.5, width=None, wlen=None, rel_height=0.5, plateau_size=None)
        if len(peaks) == 0:
            raise SolveError('No S(1,1) peaks detected.')
        self.resonance_frequency = self.freq[peaks[0]]
        self.minimum_S = self.S[peaks[0]]

    def get_resonance_frequency(self, Femtet):
        """Calculate the resonant frequency.

        Note:
            The objective or constraint function should take Femtet
            as its first argument and return a float as the output.

        Params:
            Femtet: This is an instance for manipulating Femtet with macros. For detailed information, please refer to "Femtet Macro Help".
        
        Returns:
            float: A resonance frequency of the antenna.
        """
        self._get_freq_and_S_parameter(Femtet)
        self._calc_resonance_frequency()
        return self.resonance_frequency  # unit: Hz
        

def antenna_is_smaller_than_substrate(Femtet, opt):
    """Calculate the relationship between antenna size and board size.

    This function is used to constrain the model
    from breaking down while changing parameters.

    Returns:
        float: Difference between the substrate size and antenna size. Must be equal to or grater than 1 mm.
    """
    params = opt.get_parameter()
    r = params['antenna_radius']
    w = params['substrate_w']
    return w / 2 - r  # unit: mm


def port_is_inside_antenna(Femtet, opt):
    """Calculate the relationship between the feed port location and antenna size."""
    params = opt.get_parameter()
    r = params['antenna_radius']
    x = params['port_x']
    return r - x  # unit: mm. Must be equal to or grater than 1 mm.


if __name__ == '__main__':
    # Initialize the object for calculating frequency characteristics.
    s = SParameterCalculator()

    # Initialize the numerical optimization problem.
    # (determine the optimization method)
    opt = OptunaOptimizer(
        sampler_class=BoTorchSampler,
        sampler_kwargs=dict(
            n_startup_trials=10,
        )
    )
    
    # Initialize the FEMOpt object.
    # (establish connection between the optimization problem and Femtet)
    femopt = FEMOpt(opt=opt)
    
    # Add design variables to the optimization problem.
    # (Specify the variables registered in the femprj file.)
    femopt.add_parameter('antenna_radius', 10, 5, 20)
    femopt.add_parameter('substrate_w', 50, 40, 60)
    femopt.add_parameter('port_x', 5, 1, 20)

    # Add the constraint function to the optimization problem.
    femopt.add_constraint(antenna_is_smaller_than_substrate, 'antenna and substrate clearance', lower_bound=1, args=(opt,))
    femopt.add_constraint(port_is_inside_antenna, 'antenna and port clearance', lower_bound=1, args=(opt,))

    # Add the objective function to the optimization problem.
    # The target frequency is 3.0 GHz.
    femopt.add_objective(s.get_resonance_frequency, 'first resonant frequency(Hz)', direction=3.0 * 1e9)

    femopt.set_random_seed(42)
    femopt.optimize(n_trials=15)
