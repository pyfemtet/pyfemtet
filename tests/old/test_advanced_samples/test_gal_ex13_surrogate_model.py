import os
import pyfemtet
from femtetutils import util
from win32com.client import Dispatch
from glob import glob
from contextlib import nullcontext
from pyfemtet.opt._test_utils.record_history import is_equal_result

import pytest

project_root = os.path.abspath(os.path.join(os.path.dirname(pyfemtet.__file__), '..'))
sample_root = os.path.join(project_root, 'pyfemtet/opt/advanced_samples/surrogate_model')

target_files = dict(
    femprj=os.path.join(sample_root, 'gal_ex13_parametric.femprj'),
    py_training=os.path.join(sample_root, 'gal_ex13_create_training_data.py'),
    py_optimize=os.path.join(sample_root, 'gal_ex13_optimize_with_surrogate.py'),
)

# 存在確認
for target_file_path in target_files.values():
    assert os.path.exists(target_file_path)


class LaunchFemtet:

    def __enter__(self):
        util.auto_execute_femtet(wait_second=15)


    def __exit__(self, exc_type, exc_val, exc_tb):
        pid = util.get_last_executed_femtet_process_id()

        if pid == 0:
            return

        Femtet = Dispatch('FemtetMacro.Femtet')
        Femtet.Exit(True)


# @pytest.mark.fem
def test_make_training_data():

    # Femtet を起動して学習スクリプトを実行
    with LaunchFemtet():

        # training pre-request
        for path in glob(os.path.join(sample_root, 'training_data.*')):
            os.remove(path)

        # training
        py_path = target_files['py_training']
        with open(py_path, 'r') as f:
            script = f.read()

        # script の設定を変える
        script = script.replace(
            'femopt.optimize(',
            'femopt.optimize(n_trials=30, confirm_before_exit=False'
        )

        # run
        exec(script, {'__file__': py_path, '__name__': '__main__'})

        # get result
        pass


@pytest.mark.no_fem
def test_optimize_with_surrogate_model():

    # サンプルを起動
    with nullcontext():

        # remove existing optimized data
        for path in glob(os.path.join(sample_root, 'optimized_result_target_*')):
            os.remove(path)

        # optimization script
        py_path = target_files['py_optimize']
        with open(py_path, 'r') as f:
            script = f.read()

        # replace training data
        script = script.replace(
            'training_data.csv',
            os.path.join(os.path.dirname(__file__), 'training_data.reccsv').replace('\\', os.path.altsep)
        )

        # run
        exec(script, {'__file__': py_path, '__name__': '__main__'})

        # get result
        result_path_1 = os.path.join(
            sample_root, 'optimized_result_target_1000.csv'
        )
        result_path_2 = os.path.join(
            sample_root, 'optimized_result_target_2000.csv'
        )

        # check result
        reference_path_1 = os.path.join(os.path.dirname(__file__), 'optimized_result_target_1000.reccsv')
        reference_path_2 = os.path.join(os.path.dirname(__file__), 'optimized_result_target_2000.reccsv')

        is_equal_result(result_path_1, reference_path_1)
        is_equal_result(result_path_2, reference_path_2)
        print('surrogate passed!')


if __name__ == '__main__':
    test_make_training_data()
    test_optimize_with_surrogate_model()
