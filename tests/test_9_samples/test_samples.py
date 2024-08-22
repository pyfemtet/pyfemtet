import os
import sys
import subprocess
import datetime
from glob import glob

import pyfemtet

from pyfemtet import _test_util
from pyfemtet.message import encoding
# encoding = 'utf-8'


here = os.path.dirname(__file__)
test_script = os.path.join(here, 'test_of_test.py')

LIBRARY_ROOT = os.path.dirname(pyfemtet.__file__)
SAMPLE_DIR = os.path.join(LIBRARY_ROOT, 'opt', 'femprj_sample')

RECORD_MODE = False

os.chdir(here)


def run(py_script_path, log_path):
    with open(log_path, 'w', encoding=encoding) as f:
        # run sample script
        process = subprocess.Popen(
            [sys.executable, py_script_path],
            stdin=subprocess.PIPE,
            stdout=f,
            stderr=f,
            cwd=os.path.dirname(py_script_path),
            encoding=encoding,
        )

        # press enter to skip input() in sample script
        process.stdin.write('\n')
        process.stdin.flush()
        process.communicate()


def postprocess(py_script_path, log_path):
    created_csv_path = _test_util.find_latest_csv(dir_path=os.path.dirname(py_script_path))
    reccsv_path = _test_util.py_to_reccsv(py_script_path, suffix='_test_result')

    if RECORD_MODE:
        if os.path.exists(reccsv_path): os.remove(reccsv_path)
        os.rename(created_csv_path, reccsv_path)
    else:
        if py_script_path != test_script:
            _test_util.is_equal_result(reccsv_path, created_csv_path, log_path)


def check_traceback(log_path):
    # get log
    with open(log_path, 'r', encoding=encoding, newline='\n') as f:
        lines = f.readlines()

    # remove after 'Finished. Press Enter to quit...'
    out1 = []
    for line in lines:
        if 'Finished. Press Enter to quit...' in line:
            break
        out1.append(line)

    # remove ignored traceback
    out2 = []
    skip_next = False
    for line in out1:
        if skip_next:
            skip_next = False
            continue
        skip_next = 'Exception ignored in: ' in line
        out2.append(line)

    # check traceback
    log = '\n'.join(out2)
    assert 'Traceback' not in log, '実行したスクリプトの出力に無視されていない例外があります。'


def main(femprj_path=None):
    # launch and open Femtet
    _test_util.launch_femtet(femprj_path)

    # get py_script
    if femprj_path is None:
        py_script_path = test_script
    else:
        py_script_path = femprj_path.replace('.femprj', '.py')

    # get log file
    basename = os.path.basename(py_script_path)
    log_path = os.path.join(here, datetime.datetime.now().strftime(f'%y%m%d_%H：%M_{basename}.log'))

    # run
    run(py_script_path, log_path)

    # terminate Femtet
    _test_util.taskkill_femtet()

    # check error
    check_traceback(log_path)

    # check result
    postprocess(py_script_path, log_path)


def test_sample_gau_ex08_parametric():
    femprj_path = os.path.join(SAMPLE_DIR, 'gau_ex08_parametric.femprj')
    main(femprj_path)


def test_sample_her_ex40_parametric():
    femprj_path = os.path.join(SAMPLE_DIR, 'her_ex40_parametric.femprj')
    main(femprj_path)


def test_sample_wat_ex14_parametric():
    femprj_path = os.path.join(SAMPLE_DIR, 'wat_ex14_parametric.femprj')
    main(femprj_path)


def test_sample_paswat_ex1_parametric():
    femprj_path = os.path.join(SAMPLE_DIR, 'paswat_ex1_parametric.femprj')
    main(femprj_path)


def test_sample_gal_ex58_parametric():
    femprj_path = os.path.join(SAMPLE_DIR, 'gal_ex58_parametric.femprj')
    main(femprj_path)


def test_cad_sample_nx_ex01():
    femprj_path = os.path.join(SAMPLE_DIR, 'cad_ex01_NX.femprj')
    main(femprj_path)


def test_cad_sample_sldworks_ex01():
    femprj_path = os.path.join(SAMPLE_DIR, 'cad_ex01_SW.femprj')
    main(femprj_path)


def test_sample_parametric():
    femprj_path = os.path.join(SAMPLE_DIR, 'ParametricIF.femprj')
    main(femprj_path)


def test_test():
    main()


if __name__ == '__main__':

    RECORD_MODE = False

    # main()

    test_sample_gau_ex08_parametric()
    test_sample_her_ex40_parametric()
    test_sample_wat_ex14_parametric()
    test_sample_paswat_ex1_parametric()
    test_sample_gal_ex58_parametric()
    test_cad_sample_nx_ex01()
    test_cad_sample_sldworks_ex01()
    test_sample_parametric()
