from contextlib import closing

from pyfemtet.opt.interface.solidworks_interface import SolidworksInterface
from pyfemtet.opt.interface import FemtetWithSolidworksInterface
from pyfemtet.opt.optimizer import OptunaOptimizer
from pyfemtet.opt.femopt import FEMOpt

from tests import get


def test_solidworks_interface_update():

    fem = SolidworksInterface(
        sldprt_path=get(__file__, 'test_solidworks_interface.sldprt'),
        close_solidworks_on_terminate=True,
    )

    with closing(fem):
        fem._setup_before_parallel()
        fem._setup_after_parallel()
        fem.update_parameter({'x': 20})
        fem.update_model()


def test_femtet_with_solidworks_interface():
    fem = FemtetWithSolidworksInterface(
        femprj_path=get(__file__, 'test_femtet_with_cad_interface.femprj'),
        sldprt_path=get(__file__, 'test_solidworks_interface.sldprt'),
        close_solidworks_on_terminate=True,
    )

    with closing(fem):
        fem._setup_before_parallel()
        fem._setup_after_parallel()
        fem.update_parameter({'x': 20})
        fem.update_model()


def test_parallel_opt__femtet_with_solidworks():

    fem = FemtetWithSolidworksInterface(
        femprj_path=get(__file__, 'test_femtet_with_cad_interface.femprj'),
        sldprt_path=get(__file__, 'test_solidworks_interface.sldprt'),
        close_solidworks_on_terminate=True,
    )
    fem.use_parametric_output_as_objective(1, 'minimize')

    opt = OptunaOptimizer()
    opt.fem = fem
    opt.add_parameter(name='x', initial_value=1, lower_bound=1, upper_bound=2,)
    opt.add_parameter(name='y', initial_value=1, lower_bound=1, upper_bound=2,)
    opt.add_parameter(name='z', initial_value=1, lower_bound=1, upper_bound=2,)

    opt.n_trials = 9

    femopt = FEMOpt()
    femopt.opt = opt
    femopt.optimize(n_parallel=1)


if __name__ == '__main__':
    # test_solidworks_interface_update()
    # test_femtet_with_solidworks_interface()
    test_parallel_opt__femtet_with_solidworks()
