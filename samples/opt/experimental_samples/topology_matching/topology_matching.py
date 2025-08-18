r"""Optimization Using Topology Matching

In optimization, design parameters are varied to update the model shape.

At this time, depending on how the model is created,
the topology IDs inside the CAD may change,
which is a known issue that can cause unintended states
in the assignment of boundary conditions and mesh sizes.

In Femtet and PyFemtet, we have experimentally implemented
a Topology Matching feature applying the technique described in paper [1].

This sample demonstrates optimization using Topology Matching
to address the problem where boundary conditions are broken
with conventional methods.

# Limitations
This feature supports models with only one body.

# Prerequisites
1. Femtet version 2025.0.2 or later is required.

2. Additional modules are needed to use Topology Matching.
Please install the module using the following command.
(The library brepmatching published under the MIT license will be installed.)

    py -m pip install -U pyfemtet[matching]

        or

    py -m pip install -U brepmatching

[1]
Benjamin Jones, James Noeckel, Milin Kodnongbua, Ilya Baran, and Adriana Schulz. 2023.
B-rep Matching for Collaborating Across CAD Systems.
ACM Trans. Graph. 42, 4, Article 104 (August 2023), 13 pages.
https://doi.org/10.1145/3592125

"""
import os
from time import sleep
import numpy as np
from optuna.samplers import BruteForceSampler
from win32com.client import Dispatch, constants
from pyfemtet.opt import FEMOpt
from pyfemtet.opt.optimizer import OptunaOptimizer

# Import the functionality to perform optimization in Femtet
# while doing topology matching
from pyfemtet.opt.interface.beta import FemtetWithTopologyMatching


here = os.path.dirname(__file__)


def calc_angle(Femtet):
    """Function to calculate the tilt of the mold plate
    from Femtet's analysis results"""

    cx = Femtet.GetVariableValue('cx')
    cy = Femtet.GetVariableValue('cy')
    cl2 = Femtet.GetVariableValue('cl2')

    Femtet = Dispatch('FemtetMacro.Femtet')
    Femtet.OpenCurrentResult(True)
    Femtet.Gogh.Activate()

    point1 = Dispatch('FemtetMacro.GaudiPoint')
    point1.X = cx
    point1.Y = cy
    point1.Z = cl2
    point2 = Dispatch('FemtetMacro.GaudiPoint')
    point2.X = 0.
    point2.Y = 0.
    point2.Z = cl2

    Femtet.Gogh.Galileo.Vector = constants.GALILEO_DISPLACEMENT_C
    Femtet.Gogh.Galileo.Part = constants.PART_VEC_C
    sleep(0.1)
    succeeded, ret = Femtet.Gogh.Galileo.MultiGetVectorAtPoint_py((point1, point2,))

    (cmplx_x1, cmplx_y1, cmplx_z1), (cmplx_x2, cmplx_y2, cmplx_z2) = ret
    x1 = cmplx_x1.Real
    y1 = cmplx_y1.Real
    z1 = cmplx_z1.Real
    x2 = cmplx_x2.Real
    y2 = cmplx_y2.Real
    z2 = cmplx_z2.Real

    dz = (point2.Z + z2) - (point1.Z + z1)
    dxy = np.sqrt(((point2.X + x2) - (point1.X + x1)) ** 2 + ((point2.Y + y2) - (point1.Y + y1)) ** 2)

    angle = dz / dxy

    return angle * 1000000


def main():
    # Initialize the functionality to perform optimization in Femtet
    # while doing topology matching
    fem = FemtetWithTopologyMatching(
        femprj_path=os.path.join(here, 'topology_matching.femprj'),
        model_name='model quarter',
    )

    # Set up the optimization problem below
    opt = OptunaOptimizer(
        sampler_class=BruteForceSampler
    )

    femopt = FEMOpt(fem=fem, opt=opt,)

    femopt.add_parameter('w', 80, 70, 90, step=10)
    femopt.add_parameter('d', 65, 60, 70, step=5)

    femopt.add_objective('angle', calc_angle)

    femopt.set_random_seed(42)

    femopt.optimize(
        confirm_before_exit=False,
    )


if __name__ == '__main__':
    main()
