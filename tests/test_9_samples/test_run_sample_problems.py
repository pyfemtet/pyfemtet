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

from pyfemtet import _test_util


LIBRARY_ROOT = os.path.dirname(pyfemtet.__file__)
SAMPLE_DIR = os.path.join(LIBRARY_ROOT, 'opt', 'femprj_sample')

RECORD_MODE = False


def main(py_script_path):
    femprj_path = py_script_path.replace('.py', '.femprj')

    # launch and open Femtet externally
    _test_util.launch_femtet(femprj_path)

    try:

        # run script
        run([sys.executable, py_script_path], check=True)

        # search generated csv
        generated_csv_path = _test_util.find_latest_csv()
        if RECORD_MODE:
            # copy to same directory
            _test_util.record_result(
                src=generated_csv_path,
                py_path=py_script_path,
                suffix='_test_result'
            )
        else:
            # search recorded csv
            recorded_csv = _test_util.py_to_reccsv(
                py_path=py_script_path,
                suffix='_test_result'
            )
            # raise assertion error if needed
            _test_util.is_equal_result(
                generated_csv_path,
                recorded_csv
            )

        # shutdown Femtet
        _test_util.taskkill_femtet()

    except Exception as e:
        # shutdown Femtet anyway
        _test_util.taskkill_femtet()
        raise e


def test_sample_gau_ex08_parametric():
    py_script_path = os.path.join(SAMPLE_DIR, 'gau_ex08_parametric.py')
    main(py_script_path)


def test_sample_her_ex40_parametric():
    py_script_path = os.path.join(SAMPLE_DIR, 'her_ex40_parametric.py')
    main(py_script_path)


def test_sample_wat_ex14_parametric():
    py_script_path = os.path.join(SAMPLE_DIR, 'wat_ex14_parametric.py')
    main(py_script_path)


def test_sample_paswat_ex1_parametric():
    py_script_path = os.path.join(SAMPLE_DIR, 'paswat_ex1_parametric.py')
    main(py_script_path)


def test_sample_gal_ex58_parametric():
    py_script_path = os.path.join(SAMPLE_DIR, 'gal_ex58_parametric.py')
    main(py_script_path)


def test_cad_sample_nx_ex01():
    py_script_path = os.path.join(SAMPLE_DIR, 'cad_ex01_NX.py')
    main(py_script_path)


def test_cad_sample_sldworks_ex01():
    py_script_path = os.path.join(SAMPLE_DIR, 'cad_ex01_SW.py')
    main(py_script_path)


if __name__ == '__main__':
    # for path in glob(os.path.join(SAMPLE_DIR, '*.py'), recursive=True):
    #     print(path)
    #     if not 'cad' in path:
    #         main(path)
    test_sample_gau_ex08_parametric()
