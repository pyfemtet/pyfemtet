import datetime
import pandas as pd


class OptimizationState:
    """Dask worker 間で共有すべき state"""
    def __init__(self):
        self._state = 'undefined'

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, value):
        print(f'---{value}---')
        self._state = value


class History:
    """Dask worker 間で共有すべき history"""

    def __init__(self):
        self.path = datetime.datetime.now().strftime('%Y%m%d_%H%M%S.csv')
        self.df = None

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
        self.df.to_csv(self.path)
