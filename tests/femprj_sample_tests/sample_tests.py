"""
下記のサンプルスクリプトの構造を前提に
サンプルスクリプトを改変してテストを行う


<import 文>

<目的関数の定義>

if __name__ == '__main__':
=> def main():  にする

    ...

    femopt = FEMOpt(...  # fem は指定していない前提
    => femopt.FEMOpt(history_path, にする

    ...


    femopt.optimize(...
    => femopt.optimize(confirm_before_exit=False, ...  にする

"""

import os
import datetime
import pyfemtet
import numpy as np
import pandas as pd
from femtetutils import util
from win32com.client import Dispatch


pyfemtet_root = os.path.dirname(pyfemtet.__file__)
sample_root = rf'{pyfemtet_root}\opt\samples\femprj_sample'

# here = os.path.dirname(__file__)
here = f'{pyfemtet_root}\\..\\tests\\femprj_sample_tests'
results = here + '/results'


class SampleTest:

    def __init__(
            self,
            sample_py_path: str,
            sample_femprj_path: str or None = None,
            record_mode: bool = False,
            threshold: float = 0.05,
    ):
        self.py_path: str = os.path.abspath(sample_py_path)
        self.femprj_path: str = os.path.abspath(sample_femprj_path) if sample_femprj_path is not None else self.py_path.replace('.py', '.femprj')
        self.record_mode: bool = record_mode
        self.threshold = threshold
        self._now = datetime.datetime.now().strftime(f'%y%m%d_%H：%M')

        assert os.path.exists(self.py_path)
        assert os.path.exists(self.femprj_path)

    @property
    def ref_path(self):
        """
        Notes:
            here
             +- result
                 +- <basename1>_ref.csv
                 +- <datetime>_<basename1>_dif.csv (current)
                 +- <datetime>_<basename2>_dif.csv
                 ...

        Returns:
            path of <basename1>_ref.csv

        """
        base = os.path.basename(self.py_path)
        return results + '/' + base + '_ref.csv'

    @property
    def dif_path(self):
        base = os.path.basename(self.py_path)
        return results + '/' + self._now + '_' + base + '_dif.csv'

    def load_script(self) -> str:
        with open(self.py_path, 'r', encoding='utf-8') as f:
            script = f.read()
        return script

    def modify_script(self, script: str) -> str:

        # femopt = FEMOpt(...
        # => femopt.FEMOpt(history_path=...,  にする
        history_path_path = self.ref_path if self.record_mode else self.dif_path

        _buff = script.replace(
            f'femopt = FEMOpt(',
            f'femopt = FEMOpt(history_path=r"{history_path_path}", '
        )

        # femopt.optimize(...
        # => femopt.optimize(confirm_before_exit=False, ...  にする
        _buff = _buff.replace(
            f'femopt.optimize(',
            f'femopt.optimize(confirm_before_exit=False, '
        )

        _buff = _buff.replace(
            "if __name__ == '__main__':",
            "def main():",
        )

        return _buff

    def save_script(self, script: str):
        path = here + '/generated_sample_script.py'
        with open(path, 'w', encoding='utf-8') as f:
            f.write(script)
        return path

    def run(self):

        # Femtet 起動
        util.auto_execute_femtet()
        Femtet = Dispatch('FemtetMacro.Femtet')

        # オープン
        Femtet.LoadProject(
            self.femprj_path,
            True  # bForce
        )

        try:
            # カレントディレクトリの変更（一応）
            _buff = os.getcwd()
            os.chdir(here)
        
            # スクリプトの編集実行
            script = self.load_script()
            mod_script = self.modify_script(script)
        
            # {GEN_PY_NAME}.py を保存
            self.save_script(mod_script)
        
            # history csv を削除
            if self.record_mode and os.path.exists(self.ref_path):
                os.remove(self.ref_path)
            elif (not self.record_mode) and os.path.exists(self.dif_path):
                os.remove(self.dif_path)
        
            # {GEN_PY_NAME}.py を実行
            import generated_sample_script
            generated_sample_script.main()
        
        finally:
            # カレントディレクトリを戻す（一応）
            os.chdir(_buff)
        
            # 起動した Femtet を終了
            Femtet.Exit(force := True)

        # record_mode なら、何もしない
        # そうでなければ、ref と比較する
        if not self.record_mode:
            # csv 取得
            dif_values = get_simplified_df_values(self.dif_path)

            # ref csv 取得
            ref_values = get_simplified_df_values(self.ref_path)

            # 比較
            if (np.abs(dif_values - ref_values) / ref_values).mean() > self.threshold:
                assert False, f'ref との平均差異が {int(self.threshold*100)}% 超です。'
            else:
                print(f'{os.path.basename(self.py_path)}, PASSED!')


def get_simplified_df_values(csv_path):
    with open(csv_path, 'r', encoding='cp932') as f:
        meta_header = f.readline()
    meta_header = 'removed' + meta_header.split('}"')[-1]
    meta_header = meta_header.split(',')

    df = pd.read_csv(csv_path, encoding='cp932', header=2)

    prm_names = []
    for meta_data, col in zip(meta_header, df.columns):
        if meta_data == 'prm':
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

    return pdf.values.astype(float)


if __name__ == '__main__':

    sample_test = SampleTest(
        rf'{sample_root}\constrained_pipe.py',
        record_mode=False,
    )
    sample_test.run()