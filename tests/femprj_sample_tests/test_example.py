"""
下記のサンプルスクリプトの構造を前提に
サンプルスクリプトを改変してテストを行う


<import 文>

<目的関数の定義>

if __name__ == '__main__':
=> def main():  にする

femopt.optimize(...
=> femopt.optimize(history_path=history_path, confirm_before_exit=False, ...  にする

"""

import os
import sys
import datetime
import importlib
import subprocess
import shutil
from pyfemtet._i18n import ENCODING
import numpy as np
import pandas as pd
from femtetutils import util
from win32com.client import Dispatch

from tests.utils.history_processor import remove_additional_data

import pytest


here = os.path.dirname(__file__)

project_root = rf'{here}\..\..'
sample_root = rf'{project_root}\samples\opt\femprj_samples'
sample_root_jp = rf'{project_root}\samples\opt\femprj_samples_jp'

results = here + '/results'


class SampleTest:

    def __init__(
            self,
            sample_py_path: str,
            sample_femprj_path: str or None = None,
            record_mode: bool = False,
            threshold: float = 0.05,
            related_file_paths: list[str] = None,
    ):
        self.py_path: str = os.path.abspath(sample_py_path)
        self.py_path_jp: str = os.path.join(sample_root_jp, os.path.basename(self.py_path.replace('.py', '_jp.py')))
        self.femprj_path: str = os.path.abspath(sample_femprj_path) if sample_femprj_path is not None else self.py_path.replace('.py', '.femprj')
        self.record_mode: bool = record_mode
        self.threshold = threshold
        self._now = datetime.datetime.now().strftime(f'%y%m%d_%H：%M')

        if related_file_paths is None:
            self.related_file_paths = []
        else:
            self.related_file_paths = related_file_paths

        assert os.path.exists(self.py_path), self.py_path
        assert os.path.exists(self.femprj_path), self.femprj_path
        if not os.path.isfile(self.py_path_jp):
            print(f'jp file of {os.path.basename(self.py_path)} not found.')

    def ref_path(self, jp):
        """
        Notes:
            here
             +- results
                 +- <basename1>_ref.csv
                 +- <datetime>_<basename1>_dif.csv (current)
                 +- <datetime>_<basename2>_dif.csv
                 ...

        Returns:
            path of <basename1>_ref.csv

        """
        if jp:
            target = self.py_path_jp
        else:
            target = self.py_path
        base = os.path.basename(target)
        return results + '/' + base + '_ref.csv'

    def dif_path(self, jp):
        if jp:
            target = self.py_path_jp
        else:
            target = self.py_path
        base = os.path.basename(target)
        return results + '/' + self._now + '_' + base + '_dif.csv'

    def load_script(self, jp=False) -> str:
        if jp:
            target = self.py_path_jp
        else:
            target = self.py_path

        with open(target, 'r', encoding='utf-8') as f:
            script = f.read()
        return script

    def modify_script(self, script: str, jp) -> str:

        history_path_path = self.ref_path(jp) if self.record_mode else self.dif_path(jp)

        _buff = script

        # femopt.optimize(...
        # => femopt.optimize(history_path, confirm_before_exit=False, ...  にする
        _buff = _buff.replace(
            f'femopt.optimize(',
            f'femopt.optimize(confirm_before_exit=False, history_path=r"{history_path_path}", '
        )

        _buff = _buff.replace(
            "if __name__ == '__main__':",
            "def main():",
        )

        return _buff

    def save_script(self, script: str):

        # create main script
        path = here + '/generated_sample_script.py'
        with open(path, 'w', encoding='utf-8') as f:
            f.write(script)

        # copy related paths
        for path in self.related_file_paths:
            filename = os.path.basename(path)
            dst_path = os.path.join(here, filename)
            if os.path.exists(dst_path):
                os.remove(dst_path)
            shutil.copy(path, dst_path)

        return path

    def run(self, jp=False):

        # Femtet 起動
        util.auto_execute_femtet()
        Femtet = Dispatch('FemtetMacro.Femtet')

        # オープン
        Femtet.LoadProject(
            self.femprj_path,
            True  # bForce
        )

        # カレントディレクトリの変更（一応）
        _buff = os.getcwd()
        os.chdir(here)

        try:
            # スクリプトの編集実行
            script = self.load_script(jp)
            mod_script = self.modify_script(script, jp)
        
            # {GEN_PY_NAME}.py を保存
            self.save_script(mod_script)
        
            # 前の history があれば削除
            if self.record_mode and os.path.exists(self.ref_path(jp)):
                os.remove(self.ref_path(jp))
                db_path = self.ref_path(jp).replace('.csv', '.db')
                if os.path.exists(db_path):
                    os.remove(db_path)
            elif (not self.record_mode) and os.path.exists(self.dif_path(jp)):
                os.remove(self.dif_path(jp))
                db_path = self.dif_path(jp).replace('.csv', '.db')
                if os.path.exists(db_path):
                    os.remove(db_path)

            # {GEN_PY_NAME}.py を実行
            # noinspection PyUnresolvedReferences
            import generated_sample_script
            importlib.reload(generated_sample_script)
            generated_sample_script.main()
        
        finally:
            # カレントディレクトリを戻す（一応）
            os.chdir(_buff)
        
            # 起動した Femtet を終了
            Femtet.Exit(True)  # force

        # record_mode なら、extra_data を削除する
        if self.record_mode:
            remove_additional_data(self.ref_path(jp))

        # そうでなければ、ref と比較する
        else:
            # csv 取得
            dif_values = _get_simplified_df_values(self.dif_path(jp))

            # ref csv 取得
            ref_values = _get_simplified_df_values(self.ref_path(jp))

            # 比較
            rate = (np.abs(dif_values - ref_values) / ref_values).mean()
            if rate > self.threshold:
                assert False, f'ref との平均差異が {int(rate*100)} であり、 {int(self.threshold*100)}% 超です。'
            else:
                print(f'{os.path.basename(self.py_path)}, PASSED!')


def _get_simplified_df_values(csv_path):
    try:
        with open(csv_path, 'r', encoding=ENCODING) as f:
            meta_header = f.readline()
    except UnicodeDecodeError:
        with open(csv_path, 'r', encoding='utf-8') as f:
            meta_header = f.readline()

    meta_header = 'removed' + meta_header.split('}"')[-1]
    meta_header = meta_header.split(',')

    try:
        df = pd.read_csv(csv_path, encoding=ENCODING, header=2)
    except Exception as e:
        try:
            df = pd.read_csv(csv_path, encoding='utf-8', header=2)
        except Exception as e2:
            raise e2 from e

    prm_names = []
    for meta_data, col in zip(meta_header, df.columns):
        if meta_data in ('prm.num.value', 'prm.cat.value'):
            prm_names.append(col)

    obj_names = []
    for meta_data, col in zip(meta_header, df.columns):
        if meta_data == 'obj':
            obj_names.append(col)

    pdf = pd.DataFrame()

    for col in prm_names:
        pdf[col] = df[col]

    for col in obj_names:
        pdf[col] = df[col]

    pdf.dropna(axis=1, inplace=True)

    return pdf.values.astype(float)


@pytest.mark.skip(reason='No reproductive.')
def test_constrained_pipe(record_mode=False):
    sample_test = SampleTest(
        rf'{sample_root}\constrained_pipe.py',
        record_mode=record_mode,
        threshold=0.5,
    )
    sample_test.run()
    sample_test.run(jp=True)


@pytest.mark.sample
@pytest.mark.femtet
def test_sample_gau_ex08_parametric(record_mode=False):
    sample_test = SampleTest(
        rf'{sample_root}\gau_ex08_parametric.py',
        record_mode=record_mode,
        threshold=0.2,
    )
    sample_test.run()
    sample_test.run(jp=True)


@pytest.mark.sample
@pytest.mark.femtet
def test_sample_her_ex40_parametric(record_mode=False):
    sample_test = SampleTest(
        rf'{sample_root}\her_ex40_parametric.py',
        record_mode=record_mode,
        threshold=0.1,
    )
    sample_test.run()
    sample_test.run(jp=True)


@pytest.mark.sample
@pytest.mark.femtet
def test_sample_wat_ex14_parametric(record_mode=False):
    sample_test = SampleTest(
        rf'{sample_root}\wat_ex14_parametric.py',
        record_mode=record_mode,
    )
    sample_test.run()
    sample_test.run(jp=True)


@pytest.mark.sample
@pytest.mark.femtet
def test_sample_paswat_ex1_parametric(record_mode=False):
    sample_test = SampleTest(
        rf'{sample_root}\paswat_ex1_parametric.py',
        record_mode=record_mode,
    )
    sample_test.run()
    sample_test.run(jp=True)


@pytest.mark.sample
@pytest.mark.femtet
def test_sample_gal_ex58_parametric(record_mode=False):
    sample_test = SampleTest(
        rf'{sample_root}\gal_ex58_parametric.py',
        record_mode=record_mode,
    )
    sample_test.run()
    sample_test.run(jp=True)


@pytest.mark.sample
@pytest.mark.femtet
def test_sample_parametric_if(record_mode=False):
    sample_test = SampleTest(
        rf'{sample_root}\ParametricIF.py',
        record_mode=record_mode,
    )
    sample_test.run()
    sample_test.run(jp=True)


def impl_cad_sample_sldworks_ex01(record_mode=False):
    sample_test = SampleTest(
        rf'{sample_root}\cad_ex01_SW.py',
        record_mode=record_mode,
        related_file_paths=[rf'{sample_root}\cad_ex01_SW.SLDPRT'],
    )
    sample_test.run()
    sample_test.run(jp=True)


# pytest 特有の windows fatal exception 対策
def _run(fun_name):
    here_, filename = os.path.split(__file__)
    module_name = filename.removesuffix('.py')

    subprocess.run(
        f'{sys.executable} '
        f'-c '
        f'"'
        f'import os;'
        f'import sys;'
        f'sys.path.append(os.getcwd());'
        f'import {module_name} as tst;'
        f'tst.{fun_name}()'
        f'"',
        cwd=os.path.abspath(here_),
        shell=True,
    ).check_returncode()


@pytest.mark.sample
@pytest.mark.femtet
@pytest.mark.cad
@pytest.mark.skip('pytest with Solidworks is unstable')
def test_cad_sample_sldworks_ex01():
    _run(impl_cad_sample_sldworks_ex01.__name__)


@pytest.mark.sample
@pytest.mark.femtet
@pytest.mark.cad
def test_cad_sample_nx_ex01(record_mode=False):
    sample_test = SampleTest(
        rf'{sample_root}\cad_ex01_NX.py',
        record_mode=record_mode,
        related_file_paths=[rf'{sample_root}\cad_ex01_NX.prt'],
    )
    sample_test.run()
    sample_test.run(jp=True)


if __name__ == '__main__':
    # test_constrained_pipe(record_mode=True)
    # test_sample_gau_ex08_parametric(record_mode=True)
    # test_sample_her_ex40_parametric(record_mode=True)
    # test_sample_wat_ex14_parametric(record_mode=True)
    # test_sample_paswat_ex1_parametric(record_mode=True)
    # test_sample_gal_ex58_parametric(record_mode=True)
    # test_sample_parametric_if(record_mode=True)
    test_cad_sample_sldworks_ex01()
    # test_cad_sample_nx_ex01(record_mode=True)

    test_sample_gau_ex08_parametric(record_mode=False)
