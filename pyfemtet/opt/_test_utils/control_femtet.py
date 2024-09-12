from time import sleep
from subprocess import run
from multiprocessing import Process
from tqdm import tqdm
from win32com.client import Dispatch
from femtetutils import util

import logging


logger = logging.getLogger("test")
logger.setLevel(logging.DEBUG)


def _open_femprj(femprj_path):
    Femtet = Dispatch("FemtetMacro.Femtet")
    for _ in tqdm(range(5), "Wait to complete Dispatch before opening femprj..."):
        sleep(1)
    Femtet.LoadProject(femprj_path, True)


def launch_femtet(femprj_path):
    # launch Femtet externally
    util.execute_femtet()
    pid = util.get_last_executed_femtet_process_id()
    for _ in tqdm(range(8), "Wait to complete Femtet launching."):
        sleep(1)

    # open femprj in a different process
    # to release Femtet for sample program
    if femprj_path:
        p = Process(
            target=_open_femprj,
            args=(femprj_path,),
        )
        p.start()
        p.join()


def taskkill_femtet():
    for _ in tqdm(range(3), "Wait before taskkill Femtet"):
        sleep(1)
    run(["taskkill", "/f", "/im", "Femtet.exe"])
    for _ in tqdm(range(3), "Wait after taskkill Femtet"):
        sleep(1)
