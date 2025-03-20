"""Multi-objective optimization: Air cooling of IC on a printed circuit board (forced convection).

Using Femtet's simple fluid-thermal analysis solver,
design a solution that reduces the chip temperature
while minimizing the substrate size.

Related project: paswat_ex1_parametric.femprj
"""
from pyfemtet.opt import FEMOpt


def chip_temp(Femtet, chip_name):
    """Obtain the maximum temperature of the chip.

    Note:
        The objective or constraint function should take Femtet
        as its first argument and return a float as the output.

    Params:
        Femtet: An instance for manipulating Femtet with macros. For detailed information, please refer to "Femtet Macro Help".
        chip_name (str): The body attribute name defined in femprj. Valid values are 'MAINCHIP' or 'SUBCHIP'.

    Returns:
        float: The maximum temperature of the body with the specified body attribute name.
    """
    Gogh = Femtet.Gogh

    max_temperature, min_temperature, mean_temperature = Gogh.Watt.GetTemp(chip_name)

    return max_temperature  # unit: degree


def substrate_size(Femtet):
    """Calculate the occupied area on the XY plane of the substrate."""
    substrate_w = Femtet.GetVariableValue('substrate_w')
    substrate_d = Femtet.GetVariableValue('substrate_d')
    return substrate_w * substrate_d  # unit: mm2


if __name__ == '__main__':

    # Initialize the FEMOpt object.
    # (establish connection between the optimization problem and Femtet)
    femopt = FEMOpt()

    # Add design variables to the optimization problem.
    # (Specify the variables registered in the femprj file.)
    femopt.add_parameter("substrate_w", 40, lower_bound=22, upper_bound=60)
    femopt.add_parameter("substrate_d", 60, lower_bound=34, upper_bound=60)
    femopt.add_parameter("rot", 0, lower_bound=0, upper_bound=180)

    # Add the objective function to the optimization problem.
    # The target bending angle is 90 degrees.
    femopt.add_objective(chip_temp, name='max temp. of<br>MAINCHIP (degree)', direction='minimize', args=('MAINCHIP',))
    femopt.add_objective(chip_temp, name='max temp. of<br>SUBCHIP (degree)', direction='minimize', args=('SUBCHIP',))
    femopt.add_objective(substrate_size, name='substrate size (mm2)', direction='minimize')

    # Run optimization.
    femopt.set_random_seed(42)
    femopt.optimize(n_trials=15)
