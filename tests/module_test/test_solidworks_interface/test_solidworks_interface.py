import os
import sys
import subprocess

import pytest

from pyfemtet.opt.interface._solidworks_interface import SolidworksInterface
from pyfemtet.opt.interface import FemtetWithSolidworksInterface
from pyfemtet.opt.optimizer import OptunaOptimizer
from pyfemtet.opt.femopt import FEMOpt

from tests import get
from tests.utils.closing import closing


def _run(fun_name):
    here, filename = os.path.split(__file__)
    module_name = filename.removesuffix('.py')

    subprocess.run(
        f'{sys.executable} '
        f'-c '
        f'"'
        f'import os;'
        f'import sys;'
        f'sys.path.append(os.getcwd());'
        f'import {module_name} as tst;'
        f'tst.{fun_name}()'
        f'"',
        cwd=os.path.abspath(here),
        shell=True,
    ).check_returncode()


# subprocess 経由で呼ばないと windows fatal exception が起こる
def _impl_solidworks_interface_update():

    fem = SolidworksInterface(
        sldprt_path=get(__file__, 'test_solidworks_interface.sldprt'),
        close_solidworks_on_terminate=True,
    )

    with closing(fem):
        fem._setup_before_parallel()
        fem._setup_after_parallel()
        fem.update_parameter({'x': 20})
        fem.update_model()


@pytest.mark.cad
def test_solidworks_interface_update():
    _run(_impl_solidworks_interface_update.__name__)


def _impl_femtet_with_solidworks_interface():
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


@pytest.mark.femtet
@pytest.mark.cad
def test_femtet_with_solidworks_interface():
    _run(_impl_femtet_with_solidworks_interface.__name__)


def _impl_parallel_femtet_with_solidworks():
    fem = FemtetWithSolidworksInterface(
        femprj_path=get(__file__, 'test_femtet_with_cad_interface.femprj'),
        sldprt_path=get(__file__, 'test_solidworks_interface.sldprt'),
        close_solidworks_on_terminate=True,
    )
    fem.use_parametric_output_as_objective(1, 'minimize')

    opt = OptunaOptimizer()
    opt.fem = fem
    opt.add_parameter(name='x', initial_value=10, lower_bound=5, upper_bound=20,)
    opt.add_parameter(name='y', initial_value=10, lower_bound=5, upper_bound=20,)
    opt.add_parameter(name='z', initial_value=10, lower_bound=5, upper_bound=20,)

    opt.n_trials = 9

    femopt = FEMOpt()
    femopt.opt = opt
    femopt.optimize(n_parallel=3, confirm_before_exit=False)


@pytest.mark.femtet
@pytest.mark.cad
def test_parallel_femtet_with_solidworks():
    _run(_impl_parallel_femtet_with_solidworks.__name__)


if __name__ == '__main__':
    test_solidworks_interface_update()
    test_femtet_with_solidworks_interface()
    test_parallel_femtet_with_solidworks()
