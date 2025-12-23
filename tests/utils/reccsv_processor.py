import os
import numpy as np


class RECCSV:

    def __init__(self, base_path, record: bool):
        self.base_path = base_path
        self.record = record

        self.csv_path = base_path + '.csv'
        self.db_path = base_path + '.db'
        self.reccsv_path = base_path + '.reccsv'
        if self.record:
            if os.path.isfile(self.reccsv_path):
                os.remove(self.reccsv_path)
        if os.path.isfile(self.csv_path):
            os.remove(self.csv_path)
        if os.path.isfile(self.db_path):
            os.remove(self.db_path)

    def check(
            self,
            check_columns_float,
            check_columns_str,
            dif_df,
            rtol=None,
    ):
        if self.record:
            os.rename(self.csv_path, self.reccsv_path)
        else:
            from pyfemtet.opt.history import History
            reference = History()
            reference.load_csv(self.reccsv_path, with_finalize=True)
            ref_df = reference.get_df()
            for col in check_columns_float:
                ref = [f'{v: .4e}' for v in ref_df[col].values]
                dif = [f'{v: .4e}' for v in dif_df[col].values]
                if rtol is None:
                    is_ok = [r == d for r, d in zip(ref, dif)]
                else:
                    is_ok = [np.isclose(float(r), float(d), atol=0., rtol=rtol)
                             for r, d in zip(ref, dif)]
                assert all(is_ok), f'{col} error.'
            for col in check_columns_str:
                ref = ref_df[col].values
                dif = dif_df[col].values
                is_ok = [r == d for r, d in zip(ref, dif)]
                assert all(is_ok), f'{col} error.'
            print("RECCSV check passed.")
