import os
import subprocess
from time import time, sleep
from dotenv import load_dotenv
from win32com.client import Dispatch
from pyfemtet.opt.problem.problem import TrialInput, Variable

# topology_matching 関係機能を import する前に
# テスト用環境変数の設定が必要
here = os.path.dirname(__file__)
load_dotenv(dotenv_path=os.path.join(here, ".env"))

from pyfemtet.beta.topology_matching import reexecute_model_with_topology_matching
from pyfemtet.opt.interface.beta import FemtetWithTopologyMatching


class FemtetForTest:
    def __enter__(self):
        self.popen = subprocess.Popen(
            os.path.join(os.environ.get("FEMTET_ROOT_DIR"), "Femtet_d.exe")
        )
        self.Femtet = Dispatch("FemtetMacro.Femtet")
        start = time()
        while self.Femtet.hWnd <= 0:
            sleep(1)
            if time() - start > 30:
                self.popen.kill()
                raise TimeoutError("Failed to launch Femtet.")
        return self.Femtet

    def __exit__(self, exc_type, exc_val, exc_tb):
        succeeded = self.Femtet.Exit(True)
        if not succeeded:
            self.popen.kill()


def update_model(Femtet):
    if not Femtet.UpdateVariable("x", 4):
        Femtet.ShowLastError()
    if not Femtet.UpdateVariable("y", 4):
        Femtet.ShowLastError()
    if not Femtet.UpdateVariable("z", 4):
        Femtet.ShowLastError()
    Femtet.Gaudi.ReExecute()
    Femtet.Redraw()


def test_brepmatching():
    femprj_path = __file__.removesuffix("py") + "femprj"
    with FemtetForTest() as Femtet:

        sleep(1)

        print(f"{Femtet.hWnd=}")
        Femtet.LoadProject(femprj_path, True)

        sleep(1)

        reexecute_model_with_topology_matching(
            Femtet=Femtet, rebuild_fun=update_model, args=(Femtet,)
        )

        sleep(1)


def test_interface_topology_matching():
    femprj_path = __file__.removesuffix("py") + "femprj"
    with FemtetForTest() as Femtet:

        sleep(1)

        print(f"{Femtet.hWnd=}")
        fem = FemtetWithTopologyMatching(
            femprj_path=femprj_path,
            connect_method='existing',
        )

        x = Variable()
        x.name = 'x'
        x.value = 4
        y = Variable()
        y.name = 'y'
        y.value = 4
        z = Variable()
        z.name = 'z'
        z.value = 4
        trial_input = TrialInput(
            x=x, y=y, z=z
        )

        sleep(1)

        fem.update_parameter(trial_input)
        fem.update()

        sleep(1)

if __name__ == '__main__':
    test_interface_topology_matching()
