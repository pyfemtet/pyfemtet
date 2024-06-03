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


def test_remote_cluster():
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
    femopt.optimize(n_parallel=3, n_trials=90)
    femopt.terminate_all()


def test_local_cluster():

    hostname = 'localhost'
    port = 60000

    # launch scheduler
    Popen(
        f'powershell -file launch_scheduler.ps1 -port {port}',
        shell=True
    )

    # launch worker
    Popen(
        f'powershell -file launch_workers.ps1 -hostname {hostname} -port {port} -nworkers -1',
        shell=True
    )

    sleep(20)

    # femopt setup
    fem = NoFEM()
    femopt = FEMOpt(fem=fem, scheduler_address=f'tcp://{hostname}:{port}')
    femopt.add_parameter('a', 0, -1, 1)
    femopt.add_objective(x)
    femopt.add_objective(y)
    femopt.optimize(n_parallel=3, n_trials=30)
    femopt.terminate_all()

if __name__ == '__main__':
    test_local_cluster()
