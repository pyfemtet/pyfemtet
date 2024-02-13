import os
import sys
from glob import glob
from time import sleep
from multiprocessing import Process
from subprocess import run

from tqdm import tqdm
from femtetutils import util
from win32com.client import Dispatch

import pyfemtet
from pyfemtet.dispatch_extensions import _get_hwnds


LIBRARY_ROOT = os.path.dirname(pyfemtet.__file__)
SAMPLE_DIR = os.path.join(LIBRARY_ROOT, 'FemtetPJTSample')


def open_femprj(femprj_path):
    Femtet = Dispatch('FemtetMacro.Femtet')
    for _ in tqdm(range(5), 'wait for dispatch Femtet'):
        sleep(1)
    Femtet.LoadProject(femprj_path, True)


def main(py_script_path):
    femprj_path = py_script_path.replace('.py', '.femprj')

    if not os.path.isfile(femprj_path):
        return None

    # launch Femtet
    util.execute_femtet()
    pid = util.get_last_executed_femtet_process_id()
    for _ in tqdm(range(5), 'wait for launch Femtet'):
        sleep(1)

    # open femprj
    p = Process(
        target=open_femprj,
        args=(femprj_path,),
    )
    p.start()
    p.join()

    # run script
    run([sys.executable, py_script_path], check=True)

    # shutdown Femtet
    util.close_femtet(_get_hwnds(pid)[0], 1, True)


def test_gau_ex08_parametric():
    py_script_path = os.path.join(SAMPLE_DIR, 'gau_ex08_parametric.py')
    main(py_script_path)


def test_her_ex40_parametric():
    py_script_path = os.path.join(SAMPLE_DIR, 'her_ex40_parametric.py')
    main(py_script_path)


def test_wat_ex14_parametric():
    py_script_path = os.path.join(SAMPLE_DIR, 'wat_ex14_parametric.py')
    main(py_script_path)


def test_nx_ex01():
    py_script_path = os.path.join(SAMPLE_DIR, 'NX_ex01', 'NX_ex01.py')
    main(py_script_path)


def test_sldworks_ex01():
    py_script_path = os.path.join(SAMPLE_DIR, 'Sldworks_ex01', 'Sldworks_ex01.py')
    main(py_script_path)


if __name__ == '__main__':
    for path in glob(os.path.join(SAMPLE_DIR, '*.py'), recursive=True):
        print(path)
        main(path)
