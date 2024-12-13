import os
from pathlib import Path
from optuna.samplers import QMCSampler
from pyfemtet.opt import FEMOpt, OptunaOptimizer
from pyfemtet.opt.interface._excel_interface import ExcelInterface
import pyfemtet.opt._test_utils.record_history as rh

import pytest


@pytest.mark.excel
def test_excel_interface():
    here = os.path.dirname(__file__)
    xlsm_path = f'{here}\\io_and_solve.xlsm'
    ref_csv_path = f'{here}\\ref.csv'
    csv_path = ref_csv_path if RECORD_MODE else f'{here}\\dif.csv'
    femprj_path = os.path.join(os.path.dirname(xlsm_path), 'sample.femprj')  # xlsm と同じフォルダに配置する

    if RECORD_MODE:
        if os.path.exists(ref_csv_path):
            os.remove(ref_csv_path)
        if os.path.exists(ref_csv_path.replace('.csv', '.db')):
            os.remove(ref_csv_path.replace('.csv', '.db'))
    else:
        if os.path.exists(csv_path):
            os.remove(csv_path)
        if os.path.exists(csv_path.replace('.csv', '.db')):
            os.remove(csv_path.replace('.csv', '.db'))

    fem = ExcelInterface(
        input_xlsm_path=xlsm_path,
        input_sheet_name='input',
        output_sheet_name='output',
        procedure_name='FemtetMacro.FemtetMain',
        procedure_args=(os.path.basename(femprj_path).removesuffix('.femprj'),),  # 拡張子は入れない（xlsm の実装に合わせる）
        procedure_timeout=60,
        with_call_femtet=False,
        connect_method='new',
        setup_procedure_name='launch_femtet',
        teardown_procedure_name='terminate_femtet',
        related_file_paths=[Path(femprj_path)],
        visible=True,
        interactive=True,
    )


    opt = OptunaOptimizer(
        sampler_class=QMCSampler,
    )

    femopt = FEMOpt(
        fem=fem,
        opt=opt,
        history_path=csv_path,
    )

    femopt.set_random_seed(42)

    df = femopt.optimize(
        n_trials=30,
        confirm_before_exit=False,
        n_parallel=4,
    )

    csv_path = femopt.history_path

    if RECORD_MODE:
        rh.remove_femprj_metadata_from_csv(
            csv_path=ref_csv_path,
        )

    else:
        # csv 取得
        dif_values = rh._get_simplified_df_values(csv_path, exclude_columns=['原点変位'])
        index = dif_values[:, 0].argsort(axis=0)
        dif_values = dif_values[index]
        index = dif_values[:, 1].argsort(axis=0)
        dif_values = dif_values[index]
        index = dif_values[:, 2].argsort(axis=0)
        dif_values = dif_values[index]

        # ref csv 取得
        ref_values = rh._get_simplified_df_values(ref_csv_path, exclude_columns=['原点変位'])
        index = ref_values[:, 0].argsort(axis=0)
        ref_values = ref_values[index]
        index = ref_values[:, 1].argsort(axis=0)
        ref_values = ref_values[index]
        index = ref_values[:, 2].argsort(axis=0)
        ref_values = ref_values[index]

        # 比較
        threshold = 0.1
        import numpy as np
        if (np.abs(dif_values - ref_values) / ref_values).mean() > threshold:
            assert False, f'ref との平均差異が {int(threshold * 100)}% 超です。'
        else:
            print('ExcelInterface, PASSED!')


RECORD_MODE = False


if __name__ == '__main__':
    # RECORD_MODE = True
    test_excel_interface()
