import os
import sys
import subprocess

import pyfemtet

from pyfemtet import _test_util


LIBRARY_ROOT = os.path.dirname(pyfemtet.__file__)
SAMPLE_DIR = os.path.join(LIBRARY_ROOT, 'opt', 'femprj_sample')

RECORD_MODE = True

os.chdir(LIBRARY_ROOT)


def main(py_script_path):
    femprj_path = py_script_path.replace('.py', '.femprj')

    # launch and open Femtet externally
    _test_util.launch_femtet(femprj_path)

    try:

        # run script
        # run([sys.executable, py_script_path], check=True)
        process = subprocess.Popen(
            [sys.executable, py_script_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # press enter to skip input() in sample script
        process.stdin.write(b'\n')
        process.stdin.flush()

        # wait for the subprocess
        out, err = process.communicate()

        try:
            out = out.decode('cp932')
        except UnicodeDecodeError:
            out = out.decode('utf-8')
        try:
            err = err.decode('cp932')
        except UnicodeDecodeError:
            err = err.decode('utf-8')

        # check stdout and stderr
        print('===== stdout =====')
        print(out)
        print()
        print('===== stderr =====')
        print(err)
        print()

        # if include Exception, raise Exception
        # [x]test: continue if succeed
        # [x]test: continue if logger (all logger were stderr)
        # [x]test: stop if raise error in script
        # [x]test: stop if raise error in module (raise RuntimeError in terminate_all())
        if 'Traceback (most recent call last):' in err:
            raise Exception(f'Unexpected Error has occurred in {py_script_path}')

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
                recorded_csv,
                os.path.join(
                    os.path.dirname(__file__),
                    os.path.basename(py_script_path).replace('.py', '.txt')
                )
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


def test_sample_parametric():
    py_script_path = os.path.join(SAMPLE_DIR, 'ParametricIF.py')
    main(py_script_path)


if __name__ == '__main__':
    # for path in glob(os.path.join(SAMPLE_DIR, '*.py'), recursive=True):
    #     print(path)
    #     if not 'cad' in path:
    #         main(path)
    test_sample_parametric()
