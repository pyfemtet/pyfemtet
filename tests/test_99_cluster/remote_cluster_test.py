import os
import json
from time import sleep
from subprocess import Popen

import numpy as np

from pyfemtet.opt import FEMOpt, NoFEM


here, me = os.path.split(__file__)
os.chdir(here)


def x():
    return np.random.rand()


def y():
    return np.random.rand()


if __name__ == '__main__':

    # get remote info
    json_path = os.path.join(here, 'remote.json')
    with open(json_path, 'r') as f:
        json_data = json.load(f)
    hostname = json_data['hostname']
    port = json_data['port']

    # launch scheduler
    Popen(
        f'powershell -file launch_scheduler.ps1 -port {port}',
        shell=True
    )
    sleep(10)

    # femopt setup
    fem = NoFEM()
    femopt = FEMOpt(fem=fem, scheduler_address=f'tcp://{hostname}:{port}')
    femopt.add_parameter('a', 0, -1, 1)
    femopt.add_objective(x)
    femopt.add_objective(y)
    femopt.optimize(n_parallel=3, n_trials=30)
    femopt.terminate_all()
