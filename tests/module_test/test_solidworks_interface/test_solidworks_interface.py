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
        close_solidworks_on_terminate=True,
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


@pytest.mark.femtet
@pytest.mark.cad
@pytest.mark.skip('pytest with Solidworks is unstable')
def test_sldasm():
    _run(_impl_sldasm.__name__)


@pytest.mark.femtet
@pytest.mark.cad
@pytest.mark.skip('pytest with Solidworks is unstable')
def test_multiple_solidworks_projects():
    from contextlib import nullcontext

    fem1 = FemtetWithSolidworksInterface(
        femprj_path=get(__file__, 'test_femtet_with_cad_interface.femprj'),
        sldprt_path=get(__file__, 'test_solidworks_interface.sldprt'),
        close_solidworks_on_terminate=True,
    )
    fem1.use_parametric_output_as_objective(1, 'minimize')

    fem2 = FemtetWithSolidworksInterface(
        sldprt_path=os.path.join(here, "sldasm", "Assem2.SLDASM"),
        femprj_path=os.path.join(here, "sldasm", "test_sldasm.femprj"),
        close_solidworks_on_terminate=True,
    )
    fem2.update = lambda: None  # モック

    with closing(fem1), closing(fem2):
        fem1.connect_sw()
        fem2.connect_sw()

        opt = AbstractOptimizer()
        _ctx1 = opt.add_fem(fem1)
        _ctx2 = opt.add_fem(fem2)

        _ctx1.add_parameter(name='x', initial_value=10, lower_bound=5, upper_bound=20, )
        _ctx1.add_parameter(name='y', initial_value=10, lower_bound=5, upper_bound=20, )
        _ctx1.add_parameter(name='z', initial_value=10, lower_bound=5, upper_bound=20, )
        _ctx2.add_parameter(name='array_gap', initial_value=10, lower_bound=1, upper_bound=20, )
        _ctx2.add_parameter(name='common_variable', initial_value=2, lower_bound=1, upper_bound=20, )
        _ctx2.add_parameter(name='gap', initial_value=0.25, lower_bound=0.1, upper_bound=20, )
        _ctx2.add_parameter(name='base_size', initial_value=15, lower_bound=0.1, upper_bound=20, )
        _ctx2.add_parameter(name='base_thickness', initial_value=0.5, lower_bound=0.1, upper_bound=20, )
        _ctx2.add_parameter(name='cylinder_diameter', initial_value=3, lower_bound=0.1, upper_bound=20, )

        opt.fem_manager.all_fems_as_a_fem._setup_before_parallel()
        opt._finalize()
        opt.fem_manager.all_fems_as_a_fem._setup_after_parallel(opt)

        with nullcontext():
            x = NumericParameter()
            x.name = 'x'
            x.value = 10
            x.lower_bound = 5
            x.upper_bound = 20
            x.pass_to_fem = True
            x.step = None

            y = NumericParameter()
            y.name = 'y'
            y.value = 10
            y.lower_bound = 5
            y.upper_bound = 20
            y.pass_to_fem = True
            y.step = None

            z = NumericParameter()
            z.name = 'z'
            z.value = 10
            z.lower_bound = 5
            z.upper_bound = 20
            z.pass_to_fem = True
            z.step = None

            array_gap = NumericParameter()
            array_gap.name = 'array_gap'
            array_gap.value = 10
            array_gap.lower_bound = 1
            array_gap.upper_bound = 20
            array_gap.pass_to_fem = True
            array_gap.step = None

            common_variable = NumericParameter()
            common_variable.name = 'common_variable'
            common_variable.value = 2
            common_variable.lower_bound = 1
            common_variable.upper_bound = 20
            common_variable.pass_to_fem = True
            common_variable.step = None

            gap = NumericParameter()
            gap.name = 'gap'
            gap.value = 0.25
            gap.lower_bound = 0.1
            gap.upper_bound = 20
            gap.pass_to_fem = True
            gap.step = None

            base_size = NumericParameter()
            base_size.name = 'base_size'
            base_size.value = 15
            base_size.lower_bound = 0.1
            base_size.upper_bound = 20
            base_size.pass_to_fem = True
            base_size.step = None

            base_thickness = NumericParameter()
            base_thickness.name = 'base_thickness'
            base_thickness.value = 0.5
            base_thickness.lower_bound = 0.1
            base_thickness.upper_bound = 20
            base_thickness.pass_to_fem = True
            base_thickness.step = None

            cylinder_diameter = NumericParameter()
            cylinder_diameter.name = "cylinder_diameter"
            cylinder_diameter.value = 3
            cylinder_diameter.lower_bound = 0.1
            cylinder_diameter.upper_bound = 20
            cylinder_diameter.pass_to_fem = True
            cylinder_diameter.step = None

        ss = opt._get_solve_set()
        ss.solve(
            dict(
                x=x,
                y=y,
                z=z,
                array_gap=array_gap,
                common_variable=common_variable,
                gap=gap,
                base_size=base_size,
                base_thickness=base_thickness,
                cylinder_diameter=cylinder_diameter,
            )
        )


if __name__ == '__main__':
    # _impl_solidworks_interface_update()
    # _impl_femtet_with_solidworks_interface()
    # _impl_parallel_femtet_with_solidworks()
    # _impl_sldasm()
    test_multiple_solidworks_projects()
