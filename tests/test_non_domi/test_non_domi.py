import os
import pandas as pd
from pyfemtet.opt import History
from pyfemtet.opt._femopt_core import Objective
import pyfemtet.opt._femopt_core

from contextlib import nullcontext
pyfemtet.opt._femopt_core.Lock = nullcontext


class HistoryDebug(History):

    def __init__(self):
        # hypervolume 計算メソッド
        self._hv_reference = 'dynamic-pareto'

        # 引数の処理
        self.path = os.path.join(os.path.dirname(__file__), 'test_non_domi.csv')
        self.prm_names = ['x1']
        self.obj_names = ['y1', 'y2']
        self.cns_names = []
        self.extra_data = dict()
        self.meta_columns = None
        self.__scheduler_address = None
        self._HistoryDebug__scheduler_address = None

        # 最適化実行中かどうか
        self.is_processing = False

    def get_df(self, valid_only=False) -> pd.DataFrame:
        if valid_only:
            return self.filter_valid(self._df)
        else:
            return self._df

    def set_df(self, df: pd.DataFrame):
        self._df = df


def test_non_domi():

    history = HistoryDebug()

    self = history
    columns, meta_columns = self.create_df_columns()
    df = pd.DataFrame()
    for c in columns:
        df[c] = None
    self.meta_columns = meta_columns
    self.set_df(df)

    print(history)

    history.get_df()

    y1 = [0., 0., 0., 1., 1., 1.,]
    y2 = [1., 2., 3., 4., 5., 6.,]

    for y1_, y2_ in zip(y1, y2):

        parameters = pd.DataFrame(dict(
            name=['x1'],
            value=[0],
            lower_bound=[0.],
            upper_bound=[1.],
        ))
        objectives = dict(
            y1=Objective(name='y1', direction='minimize', fun=lambda *args, y1__=y1_, **kwargs: y1__, args=(), kwargs={}, ),
            y2=Objective(name='y2', direction=4, fun=lambda *args, y1__=y1_, **kwargs: y1__, args=(), kwargs={}, ),
        )
        constraints = dict()

        obj_values = [y1_, y2_]
        cns_values = []
        message = ''
        postprocess_func = None
        postprocess_args = None

        history.record(
            parameters,
            objectives,
            constraints,
            obj_values,
            cns_values,
            message,
            postprocess_func,
            postprocess_args,
        )

    print(history.get_df()[['y1', 'y2', 'feasible', 'non_domi']])


if __name__ == '__main__':
    test_non_domi()
