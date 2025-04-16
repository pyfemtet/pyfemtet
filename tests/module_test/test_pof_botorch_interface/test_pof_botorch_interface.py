from pyfemtet.opt.interface._surrogate_model_interface.botorch_interface import PoFBoTorchInterface

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
    fem.update_parameter(dict(x=50, y=0, zA=1, zB=0, zC=0))
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
    fem.update_parameter(dict(x=50, y=0, z='A'))
    print(fem.calc_pof())
    # assert abs(fem.calc_pof() - 0.18713788252574215) < 0.05
    fem.update_parameter(dict(x=75, y=1, z='B'))
    print(fem.calc_pof())
    # assert abs(fem.calc_pof() - 0.3386818792532815) < 0.05
    fem.update_parameter(dict(x=75, y=0.5, z='C'))
    print(fem.calc_pof())
    # assert abs(fem.calc_pof() - 0.9940278367170938) < 0.05


if __name__ == '__main__':
    # test_pof_botorch_load()
    # test_pof_botorch_pof()
    # debug_pof_botorch_pof()
    debug_pof_botorch_pof_2()
