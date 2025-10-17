import os
import sys
import subprocess

import pytest

from pyfemtet.opt.optimizer import AbstractOptimizer
from pyfemtet.opt.interface._solidworks_interface import SolidworksInterface
from pyfemtet.opt.interface import FemtetWithSolidworksInterface
from pyfemtet.opt.optimizer import OptunaOptimizer
from pyfemtet.opt.femopt import FEMOpt
from pyfemtet.opt.problem.variable_manager import *

from tests import get
from tests.utils.closing import closing

here = os.path.dirname(__file__)


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
        x = Variable()
        x.name = 'x'
        x.value = 20
        x.properties = {'unit': 'mm'}

        fem._setup_before_parallel()
        fem._setup_after_parallel(AbstractOptimizer())
        fem.update_parameter({'x': x})
        fem.update_model()

        # input('Enter to quit...')


@pytest.mark.cad
@pytest.mark.skip('pytest with Solidworks is unstable')
def test_solidworks_interface_update():
    _run(_impl_solidworks_interface_update.__name__)


def _impl_femtet_with_solidworks_interface():
    fem = FemtetWithSolidworksInterface(
        femprj_path=get(__file__, 'test_femtet_with_cad_interface.femprj'),
        sldprt_path=get(__file__, 'test_solidworks_interface.sldprt'),
        close_solidworks_on_terminate=True,
    )

    with closing(fem):
        x = Variable()
        x.name = 'x'
        x.value = 20

        fem._setup_before_parallel()
        fem._setup_after_parallel()
        fem.update_parameter(dict(x=x))
        fem.update_model()


@pytest.mark.femtet
@pytest.mark.cad
@pytest.mark.skip('pytest with Solidworks is unstable')
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
    opt.add_parameter(name='x', initial_value=10, lower_bound=5, upper_bound=20, )
    opt.add_parameter(name='y', initial_value=10, lower_bound=5, upper_bound=20, )
    opt.add_parameter(name='z', initial_value=10, lower_bound=5, upper_bound=20, )

    opt.n_trials = 9

    femopt = FEMOpt()
    femopt.opt = opt
    femopt.optimize(n_parallel=3, confirm_before_exit=False)


@pytest.mark.femtet
@pytest.mark.cad
@pytest.mark.skip('pytest with Solidworks is unstable')
def test_parallel_femtet_with_solidworks():
    _run(_impl_parallel_femtet_with_solidworks.__name__)


def _impl_sldasm():
    from pyfemtet.opt.problem.variable_manager import Variable
    from pyfemtet.opt.problem.problem import TrialInput

    fem = SolidworksInterface(
        sldprt_path=os.path.join(here, "sldasm", "Assem2.SLDASM"),
    )

    fem.connect_sw()
    fem._setup_after_parallel(AbstractOptimizer())

    array_gap = Variable()
    array_gap.name = 'array_gap'
    array_gap.value = 10  # from 1

    common_variable = Variable()
    common_variable.name = 'common_variable'
    common_variable.value = 2  # from 1

    gap = Variable()
    gap.name = 'gap'
    gap.value = 0.25  # from 0.5

    base_size = Variable()
    base_size.name = 'base_size'
    base_size.value = 15  # from 10

    base_thickness = Variable()
    base_thickness.name = 'base_thickness'
    base_thickness.value = 0.5  # from 2

    cylinder_diameter = Variable()
    cylinder_diameter.name = 'cylinder_diameter'
    cylinder_diameter.value = 3  # from 0.5

    fem.update_parameter(
        x=TrialInput(
            array_gap=array_gap,
            common_variable=common_variable,
            gap=gap,
            base_size=base_size,
            base_thickness=base_thickness,
            cylinder_diameter=cylinder_diameter,
        )
    )
    fem.update_model()

    from pyfemtet.opt.interface._solidworks_interface.solidworks_interface import SolidworksVariableManager
    mgr = SolidworksVariableManager()

    out = mgr.get_equations_recourse(fem.swModel)
    print(out)

    fem.close()

    reference = ['"array_gap" = 10', '"D3@ﾛｰｶﾙ直線ﾊﾟﾀｰﾝ1"= "array_gap" + "D2@ｽｹｯﾁ2@base-1.Part@Assem1-2.Assembly"', '"common_variable" = 2', '"test"= "D3@ﾛｰｶﾙ直線ﾊﾟﾀｰﾝ1"', '"trtr.3"= "D2@ｽｹｯﾁ2@base-1.Part@Assem1-2.Assembly"', '"gap" = 0.25', '"D5@ｽｹｯﾁ2@base-1.Part"= "D1@ｽｹｯﾁ1@cylinder-1.Part" + "gap"', '"ret"= "D5@ｽｹｯﾁ2@base-1.Part"', '"D1@ｽｹｯﾁ1@cylinder-1.Part" = 0.5', '"cylinder_diameter" = 3', '"common_variable" = 2', '"D1@ｽｹｯﾁ1"= "cylinder_diameter"', '"base_size" = 15', '"D1@ｽｹｯﾁ2"="base_size"', '"D2@ｽｹｯﾁ2"="base_size"', '"D3@ｽｹｯﾁ2"="base_size" / 2', '"D4@ｽｹｯﾁ2"="base_size" / 2', '"base_thickness" = 0.5', '"D1@ﾎﾞｽ - 押し出し1"="base_thickness"', '"common_variable" = 2', '"gap" = 0.25', '"D5@ｽｹｯﾁ2@base-1.Part"= "D1@ｽｹｯﾁ1@cylinder-1.Part" + "gap"', '"ret"= "D5@ｽｹｯﾁ2@base-1.Part"', '"D1@ｽｹｯﾁ1@cylinder-1.Part" = 0.5', '"cylinder_diameter" = 3', '"common_variable" = 2', '"D1@ｽｹｯﾁ1"= "cylinder_diameter"', '"base_size" = 15', '"D1@ｽｹｯﾁ2"="base_size"', '"D2@ｽｹｯﾁ2"="base_size"', '"D3@ｽｹｯﾁ2"="base_size" / 2', '"D4@ｽｹｯﾁ2"="base_size" / 2', '"base_thickness" = 0.5', '"D1@ﾎﾞｽ - 押し出し1"="base_thickness"', '"common_variable" = 2', '"gap" = 0.25', '"D5@ｽｹｯﾁ2@base-1.Part"= "D1@ｽｹｯﾁ1@cylinder-1.Part" + "gap"', '"ret"= "D5@ｽｹｯﾁ2@base-1.Part"', '"D1@ｽｹｯﾁ1@cylinder-1.Part" = 0.5', '"cylinder_diameter" = 3', '"common_variable" = 2', '"D1@ｽｹｯﾁ1"= "cylinder_diameter"', '"base_size" = 15', '"D1@ｽｹｯﾁ2"="base_size"', '"D2@ｽｹｯﾁ2"="base_size"', '"D3@ｽｹｯﾁ2"="base_size" / 2', '"D4@ｽｹｯﾁ2"="base_size" / 2', '"base_thickness" = 0.5', '"D1@ﾎﾞｽ - 押し出し1"="base_thickness"', '"common_variable" = 2']

    assert sorted(out) == sorted(reference)


def test_sldasm():
    _run(_impl_sldasm.__name__)


if __name__ == '__main__':
    _impl_solidworks_interface_update()
    _impl_femtet_with_solidworks_interface()
    _impl_parallel_femtet_with_solidworks()
    _impl_sldasm()
