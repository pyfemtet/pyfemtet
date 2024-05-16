"""Parallel computing / Multi-objective optimization: heating element on board

Perform thermal conduction analysis on wat_ex14_parametric.femprj
and search for board dimensions and chip placement dimensions that
minimize the board dimensions while minimizing the temperature rise
due to chip heat generation.
"""
from pyfemtet.opt import FEMOpt


def max_temperature(Femtet, body_name):
    """Get the maximum temperature of the chip.

    Note:
        The objective or constraint function
        must take a Femtet as its first argument
        and must return a single float.

    Params:
        Femtet: An instance for using Femtet macros. For more information, see "Femtet Macro Help / CFemtet Class".

    Returns:
        float: Max-temperature.
    """
    Gogh = Femtet.Gogh

    temp, _, _ = Gogh.Watt.GetTemp_py(body_name)

    return temp  # degree


def substrate_size(Femtet):
    """Calculate the substrate size.

    Params:
        Femtet: An instance for using Femtet macros. For more information, see "Femtet Macro Help / CFemtet Class".
    
    Returns:
        float: The area occupied by the board in the XY plane.
    """
    subs_w = Femtet.GetVariableValue('substrate_w')
    subs_d = Femtet.GetVariableValue('substrate_d')

    return subs_w * subs_d  # mm2


if __name__ == '__main__':

    # Define FEMOpt object (This process integrates mathematical optimization and FEM.).
    femopt = FEMOpt()

    # Add design variables (Use variable names set in Femtet) to the optimization problem.
    femopt.add_parameter("substrate_w", 40, lower_bound=22, upper_bound=40)
    femopt.add_parameter("substrate_d", 60, lower_bound=33, upper_bound=60)

    # Add objective to the optimization problem.
    femopt.add_objective(max_temperature, name='main chip temp.', args=('MAINCHIP',))
    femopt.add_objective(max_temperature, name='sub chip temp.', args=('SUBCHIP',))
    femopt.add_objective(substrate_size, name='substrate size')

    # Run optimization.
    femopt.set_random_seed(42)
    # femopt.optimize(n_trials=20)
    femopt.optimize(n_trials=20, n_parallel=3)  # Change only this line.
    femopt.terminate_all()
