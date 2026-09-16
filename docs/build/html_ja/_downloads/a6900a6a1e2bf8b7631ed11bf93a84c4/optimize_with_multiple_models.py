import os
from time import sleep
from pyfemtet.opt import FEMOpt, FemtetInterface
from pyfemtet.opt.interface import NoFEM

here = os.path.dirname(__file__)


def get_natural_frequencys(Femtet):
    out = []
    Gogh = Femtet.Gogh
    Gogh.Activate()
    for mode in range(Gogh.Galileo.nMode):
        Gogh.Galileo.Mode = mode
        sleep(0.1)
        f = Gogh.Galileo.GetFreq().Real
        print(f"{mode=}, {f=}")
        out.append(f)
    return out


def difference_between_nearest_natural_frequency(Femtet, runtime_frequency):
    distances = [abs(runtime_frequency - f) for f in get_natural_frequencys(Femtet)]
    return min(distances)


def calc_cooling_area_radius_bounds(opt):
    params = opt.get_variables()
    internal_radius = params.get("internal_radius")
    return internal_radius + 1, 12


def main():
    # Model information
    femprj_path = os.path.join(here, "cylinder-shaft-cooling.femprj")
    model_name_1 = "cooling - 2d"
    model_name_2 = "frequency - 3d"

    # Define and setup model 1: Thermal-Fluid Analysis
    fem1 = FemtetInterface(
        femprj_path=femprj_path,
        model_name=model_name_1,
    )

    # Using Parametric Output Result Setting
    # as an objective
    fem1.use_parametric_output_as_objective(
        1, "minimize"  # Minimize Max Temperature
    )

    # Initialize FEMOpt with model 1
    femopt = FEMOpt(fem=fem1)

    # Define model 2: Resonance Analysis
    fem2 = FemtetInterface(
        femprj_path=femprj_path,
        model_name=model_name_2,
    )

    # Add it to femopt.
    # The return value "ctx2" is the optimization problem settings
    # related in the appended FEM.
    ctx2 = femopt.add_fem(fem2)

    # Then we add objective to the ctx instead of femopt.
    # By this, the Femtet object related in fem2 is passed
    # as `Femtet` argument of `difference_between_nearest_natural_frequency`.
    ctx2.add_objective(
        name="difference_between_nearest_natural_frequency",
        fun=difference_between_nearest_natural_frequency,
        direction='maximize',
        kwargs=dict(runtime_frequency=1400),
    )

    # Setup design parameter.
    femopt.add_parameter(
        name='internal_radius',
        initial_value=5,
        lower_bound=2,
        upper_bound=10,
    )
    femopt.add_parameter(
        name='cooling_area_radius',
        initial_value=10,
        properties=dict(
            dynamic_bounds_fun=calc_cooling_area_radius_bounds
        )
    )

    # Run optimization
    femopt.optimize(
        n_trials=50,
        confirm_before_exit=False,
    )


if __name__ == '__main__':
    main()
