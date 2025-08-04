import os
import subprocess
from time import time, sleep
from dotenv import load_dotenv
from win32com.client import Dispatch

here = os.path.dirname(__file__)
load_dotenv(dotenv_path=os.path.join(here, ".env"))

from pyfemtet.topology_matching import reexecute_model_with_topology_matching


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
        print(f"{Femtet.hWnd=}")
        Femtet.LoadProject(femprj_path, True)

        reexecute_model_with_topology_matching(
            Femtet=Femtet,
            rebuild_fun=update_model,
            args=(Femtet,)
        )
