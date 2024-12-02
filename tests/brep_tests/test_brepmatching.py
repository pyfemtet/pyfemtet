import os
from femtetutils import util
from win32com.client import Dispatch
from pyfemtet.brep import ModelUpdater

import logging
logger = logging.getLogger('brepmatching')
logger.setLevel(logging.DEBUG)


def test_brepmatching():
    util.auto_execute_femtet()
    Femtet = Dispatch('FemtetMacro.Femtet')
    Femtet.LoadProject(
        os.path.join(
            os.path.dirname(__file__),
            'test_brepmatching.femprj'
        ),
        True
    )

    mu = ModelUpdater(Femtet)

    def update_model():
        Femtet.UpdateVariable('r', 4)
        Femtet.Gaudi.ReExecute()
        Femtet.Redraw()

    mu.update_model_with_prediction(
        Femtet,
        rebuild_fun=update_model,
    )

    mu.quit()


if __name__ == '__main__':
    test_brepmatching()
