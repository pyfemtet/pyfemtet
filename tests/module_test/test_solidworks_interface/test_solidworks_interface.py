from contextlib import closing

from pyfemtet.opt.interface.solidworks_interface import SolidworksInterface
from pyfemtet.opt.interface import FemtetWithSolidworksInterface

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


if __name__ == '__main__':
    # test_solidworks_interface_update()
    test_femtet_with_solidworks_interface()
