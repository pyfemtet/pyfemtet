import os
import json
from time import sleep
from subprocess import Popen

import numpy as np
from win32com.client import constants

from pyfemtet.opt import FEMOpt, NoFEM, FemtetInterface


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


def max_disp(Femtet):
    return Femtet.Gogh.Galileo.GetMaxDisplacement_py()[1]


def volume(Femtet, opt):
    d, h, _ = opt.get_parameter('values')
    w = Femtet.GetVariableValue('w')
    return d * h * w


def mises(Femtet):
    Gogh = Femtet.Gogh
    Gogh.Galileo.Tensor = constants.GALILEO_STRESS_C
    _, tensor = Gogh.Galileo.GetTensorAtNode_py(Gogh.Data.MeshElementArray(0).MeshNodeArray(0).Index, 0)
    _, value = Gogh.Galileo.TransEquivalentValue_py(constants.GALILEO_STRESS_C, tensor, 0)
    return value


def bottom_surface(_, opt):
    d, h, w = opt.get_parameter('values')
    return d * w


def test_local_cluster_with_femtet():

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
    # fem setup
    femprj_path = os.path.join(
        here,
        '..',
        'test_4_constants',
        'test_simple_femtet_with_constants.femprj'
    )
    fem = FemtetInterface(
        femprj_path, connect_method='new',
        save_pdt=True
    )

    # femopt setup
    femopt = FEMOpt(fem=fem, scheduler_address=f'tcp://{hostname}:{port}')

    # problem setup
    femopt.opt.seed = 42
    femopt.add_parameter('d', 5, 1, 10)
    femopt.add_parameter('h', 5, 1, 10)
    femopt.add_parameter('w', 5, 1, 10)
    femopt.add_objective(max_disp, '最大変位(m)')
    femopt.add_objective(volume, '体積(mm3)', args=femopt.opt)
    femopt.add_objective(mises, 'mises 応力()')
    femopt.add_constraint(bottom_surface, '底面積<=30', upper_bound=30, args=femopt.opt)
    femopt.optimize(n_trials=10, n_parallel=3, wait_setup=True)
    femopt.terminate_all()



def test_remote_cluster_with_femtet():
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

    # fem setup
    femprj_path = os.path.join(
        here,
        '..',
        'test_4_constants',
        'test_simple_femtet_with_constants.femprj'
    )
    fem = FemtetInterface(
        femprj_path, connect_method='new',
        save_pdt = True
    )

    # femopt setup
    femopt = FEMOpt(fem=fem, scheduler_address=f'tcp://{hostname}:{port}')

    # problem setup
    femopt.opt.seed = 42
    femopt.add_parameter('d', 5, 1, 10)
    femopt.add_parameter('h', 5, 1, 10)
    femopt.add_parameter('w', 5, 1, 10)
    femopt.add_objective(max_disp, '最大変位(m)')
    femopt.add_objective(volume, '体積(mm3)', args=femopt.opt)
    femopt.add_objective(mises, 'mises 応力()')
    femopt.add_constraint(bottom_surface, '底面積<=30', upper_bound=30, args=femopt.opt)
    femopt.optimize(n_trials=10, n_parallel=3, wait_setup=True)
    femopt.terminate_all()


if __name__ == '__main__':
    test_local_cluster_with_femtet()
