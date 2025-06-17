from pyfemtet.opt.interface._surrogate_model_interface.botorch_interface import PoFBoTorchInterface
from pyfemtet.opt.problem.variable_manager import *

from tests import get


def test_pof_botorch_load():
    PoFBoTorchInterface(
        history_path=get(__file__, 'test_history.reccsv')
    )


def debug_pof_botorch_pof():
    PoFBoTorchInterface._debug = True
    fem = PoFBoTorchInterface(
        history_path=get(__file__, 'test_history_2.reccsv')
    )
    print(fem.calc_pof())
    fem._debug = False

    x = Variable()
    x.name = 'x'
    x.value = 50

    y = Variable()
    y.name = 'y'
    y.value = 0

    za = Variable()
    za.name = 'zA'
    za.value = 1

    zb = Variable()
    zb.name = 'zB'
    zb.value = 0

    zc = Variable()
    zc.name = 'zC'
    zc.value = 0

    fem.update_parameter(dict(x=x, y=y, zA=za, zB=zb, zC=zc))
    print(fem.calc_pof())


def debug_pof_botorch_pof_2():
    PoFBoTorchInterface._debug = True
    fem = PoFBoTorchInterface(
        history_path=get(__file__, 'test_history_4.reccsv')
    )
    print(fem.calc_pof())


def test_pof_botorch_pof():
    # PoFBoTorchInterface._debug = True
    fem = PoFBoTorchInterface(
        history_path=get(__file__, 'test_history_3.reccsv')
    )

    x = Variable()
    x.name = 'x'
    x.value = 50

    y = Variable()
    y.name = 'y'
    y.value = 0

    z = Variable()
    z.name = 'z'
    z.value = 'A'

    fem.update_parameter(dict(x=x, y=y, z=z))
    print(fem.calc_pof())
    # assert abs(fem.calc_pof() - 0.18713788252574215) < 0.05

    x = Variable()
    x.name = 'x'
    x.value = 75

    y = Variable()
    y.name = 'y'
    y.value = 1

    z = Variable()
    z.name = 'z'
    z.value = 'B'

    fem.update_parameter(dict(x=x, y=y, z=z))
    print(fem.calc_pof())
    # assert abs(fem.calc_pof() - 0.3386818792532815) < 0.05

    x = Variable()
    x.name = 'x'
    x.value = 75

    y = Variable()
    y.name = 'y'
    y.value = 0.5

    z = Variable()
    z.name = 'z'
    z.value = 'C'

    fem.update_parameter(dict(x=x, y=y, z=z))
    print(fem.calc_pof())
    # assert abs(fem.calc_pof() - 0.9940278367170938) < 0.05


if __name__ == '__main__':
    # test_pof_botorch_load()
    test_pof_botorch_pof()
    # debug_pof_botorch_pof()
    # debug_pof_botorch_pof_2()
