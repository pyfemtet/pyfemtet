import datetime
import pandas as pd
from win32com.client import constants


class OptimizationState:
    """Dask worker 間で共有すべき state"""
    def __init__(self):
        self._state = 'undefined'

    def get_state(self):
        return self._state

    def set_state(self, value):
        print(f'---{value}---')
        self._state = value


class History:
    """Dask worker 間で共有すべき history"""
    path = ''
    df = None

    def __init__(self):
        self.path = datetime.datetime.now().strftime('%Y%m%d_%H%M%S.csv')

    def init(self, prm_names, obj_names):
        columns = []
        columns.extend(prm_names)
        columns.extend(obj_names)
        self.df = pd.DataFrame(columns=columns)

    def record(self, x, obj_values, cns_values):
        row = []
        row.extend(x)
        row.extend(obj_values)
        row.extend(cns_values)
        self.df.loc[len(self.df)] = row
        try:
            self.df.to_csv(self.path, encoding='shift-jis')
        except PermissionError:
            pass


class Scapegoat:
    """Helper class for parallelize Femtet."""
    # constants を含む関数を並列化するために
    # メイン処理で一時的に constants への参照を
    # このオブジェクトにして、後で restore する
    pass


def restore_constants(fun):
    """Helper function for parallelize Femtet."""
    for varname in fun.__globals__:
        if isinstance(fun.__globals__[varname], Scapegoat):
            fun.__globals__[varname] = constants
