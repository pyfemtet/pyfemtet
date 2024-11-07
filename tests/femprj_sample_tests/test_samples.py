import os
import sys
import subprocess
import datetime

import pyfemtet

# noinspection PyProtectedMember
from pyfemtet.opt._test_utils import record_history, control_femtet
from pyfemtet._message import encoding
log_encoding = 'utf-8'

here = os.path.dirname(__file__)
test_script = os.path.join(here, 'test_of_test.py')

LIBRARY_ROOT = os.path.dirname(pyfemtet.__file__)
SAMPLE_DIR = os.path.join(LIBRARY_ROOT, 'opt', 'samples', 'femprj_sample')

RECORD_MODE = False

os.chdir(here)


def run(py_script_path, log_path):
    with open(log_path, 'w', encoding=log_encoding) as f:
        # run sample script
        process = subprocess.Popen(
            [sys.executable, py_script_path],
            stdin=subprocess.PIPE,
            stdout=f,
            stderr=f,
            cwd=os.path.dirname(py_script_path),
            encoding=log_encoding,
        )

        # press enter to skip input() in sample script
        process.stdin.write('\n')
        process.stdin.flush()
        process.communicate()

        return process.returncode


def postprocess(py_script_path, log_path):
    created_csv_path = record_history.find_latest_csv(dir_path=os.path.dirname(py_script_path))
    reccsv_path = record_history.py_to_reccsv(py_script_path, suffix='_test_result')

    if RECORD_MODE:
        if os.path.exists(reccsv_path): os.remove(reccsv_path)
        os.rename(created_csv_path, reccsv_path)
    else:
        if py_script_path != test_script:
            record_history.is_equal_result(reccsv_path, created_csv_path, log_path)


def check_traceback(log_path):
    # get log
    try:
        with open(log_path, 'r', encoding=log_encoding, newline='\n') as f:
            lines = f.readlines()
    except UnicodeDecodeError:
        with open(log_path, 'r', encoding=encoding, newline='\n') as f:
            lines = f.readlines()

    for line in lines:
        assert "unexpected exception raised on worker" not in line, '実行したスクリプトの出力に無視されていない例外があります。'


def main(femprj_path=None):
    # launch and open Femtet
    control_femtet.launch_femtet(femprj_path)

    # get py_script
    if femprj_path is None:
        py_script_path = test_script
    else:
        py_script_path = femprj_path.replace('.femprj', '.py')

    # create log file (default name with "failed")
    basename = os.path.basename(py_script_path)
    log_path = os.path.join(here, datetime.datetime.now().strftime(f'%y%m%d_%H：%M_failed_{basename}.log'))

    # run
    print("============================================")
    print('Launch subprocess / Running sample script...')
    print("============================================")
    return_code = run(py_script_path, log_path)

    # terminate Femtet
    control_femtet.taskkill_femtet()

    # check error 1 (before optimize())
    if return_code != 0:
        raise Exception('スクリプトの実行中、optimize() の実施前にエラーが発生しました。')

    # check error 2 (inside optimize())
    check_traceback(log_path)

    # check result
    postprocess(py_script_path, log_path)

    # If here, test ran successfully (without any unexpected error or curious optimization result.)
    # rename log file: "failed" to "succeed"
    new_log_path = log_path.replace('_failed_', '_succeed_')
    os.rename(log_path, new_log_path)


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


def test_sample_constraint():
    femprj_path = os.path.join(SAMPLE_DIR, 'constrained_pipe.femprj')
    main(femprj_path)


if __name__ == '__main__':

    def test_test():
        main()


    def try_test(f):
        try:
            f()
        except Exception as e:
            # from traceback import print_exception
            from time import sleep
            print()
            print(f.__name__, "test failed")
            print("==========")
            sleep(1)
            # print_exception(e)  # 次の print() が先に来る  # どこで失敗したか見る必要があるのはテストをデバッグするときだけ
            print(e)
            print()


    # RECORD_MODE = True
    # try_test(test_sample_constraint)

    # RECORD_MODE = False
    # try_test(test_sample_constraint)

    # try_test(test_test)
    # try_test(test_sample_gau_ex08_parametric)
    # try_test(test_sample_her_ex40_parametric)
    # try_test(test_sample_wat_ex14_parametric)
    # try_test(test_sample_paswat_ex1_parametric)
    try_test(test_sample_gal_ex58_parametric)
    # try_test(test_cad_sample_nx_ex01)
    # try_test(test_cad_sample_sldworks_ex01)
    # try_test(test_sample_parametric)
    # ry_test(test_sample_constraint)
