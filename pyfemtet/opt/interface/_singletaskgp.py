from typing import Optional, List

import numpy as np
import pandas as pd
from scipy.stats.distributions import norm

from pyfemtet.core import SolveError
from pyfemtet.logger import get_module_logger
from pyfemtet.opt._femopt_core import History, Objective
from pyfemtet.opt.interface._base import FEMInterface
from pyfemtet.opt.prediction.single_task_gp import SingleTaskGPModel
from pyfemtet.opt.optimizer._base import AbstractOptimizer

logger = get_module_logger('opt.interface', __name__)


class SurrogateModelInterface(FEMInterface):
    pass


class PoFBoTorchInterface(SurrogateModelInterface):

    def __init__(
            self,
            history_path: str = None,
            history: History = None,
    ):

        self.history: History
        self.model_f: SingleTaskGPModel
        self.model: SingleTaskGPModel
        self.prm: dict[str, float] = dict()
        self.obj: dict[str, float] = dict()
        self.df_prm: pd.DataFrame
        self.df_obj: pd.DataFrame

        # history_path が与えられた場合、history をコンストラクトする
        if history_path is not None:
            history = History(history_path=history_path)

        # history が与えられるかコンストラクトされている場合
        if history is not None:
            # 学習データを準備する
            df_prm = history.get_df()[history.prm_names]
            df_obj = history.get_df()[history.obj_names]

            # obj の名前を作る
            for obj_name in history.obj_names:
                self.obj[obj_name] = np.nan

            # prm の名前を作る
            for prm_name in history.prm_names:
                self.prm[prm_name] = np.nan

            self.history = history

        # history から作らない場合、引数チェック
        else:
            # assert len(train_x) == len(train_y)
            raise NotImplementedError

        self.df_prm = df_prm
        self.df_obj = df_obj

        FEMInterface.__init__(
            self,
            history=history,  # コンストラクト済み history を渡せば並列計算時も何もしなくてよい
        )

    def filter_feasible(self, x: np.ndarray, y: np.ndarray, return_feasibility=False):
        feasible_idx = np.where(~np.isnan(y.sum(axis=1)))
        if return_feasibility:
            # calculated or not
            y = np.zeros_like(y)
            y[feasible_idx] = 1.
            # satisfy weak feasibility or not
            infeasible_idx = np.where(~self.history.get_df()['feasible'].values)
            y[infeasible_idx] = .0
            return y.reshape((-1, 1))
        else:
            return x[feasible_idx], y[feasible_idx]

    def train(self):
        # df そのまま用いて training する
        x, y = self.filter_feasible(self.df_prm.values, self.df_obj.values)
        assert len(x) > 0, 'No feasible results in training data.'
        self.model.fit(x, y)

    def train_f(self):
        # df そのまま用いて training する
        y = self.filter_feasible(self.df_prm.values, self.df_obj.values, return_feasibility=True)
        if y.min() == 1:
            self.model_f.predict = lambda *args, **kwargs: (1., 0.001)
        self.model_f.fit(self.df_prm.values, y)

    def _setup_after_parallel(self, *args, **kwargs):

        opt: AbstractOptimizer = kwargs['opt']

        obj: Objective
        for obj_name, obj in opt.objectives.items():
            obj.fun = lambda: self.obj[obj_name]

        # model training
        self.model = SingleTaskGPModel()
        self.model.set_bounds_from_history(self.history.get_df(), self.history)
        self.train()

        # model_f training
        self.model_f = SingleTaskGPModel(is_noise_free=False)
        self.model_f.set_bounds_from_history(self.history.get_df(), self.history)
        self.train_f()

    def update_parameter(self, parameters: pd.DataFrame, with_warning=False) -> Optional[List[str]]:
        for i, row in parameters.iterrows():
            name, value = row['name'], row['value']
            self.prm[name] = value

    def update(self, parameters: pd.DataFrame) -> None:
        # self.prm 更新
        self.update_parameter(parameters)

        # history.prm_name 順に並べ替え
        x = np.array([self.prm[k] for k in self.history.prm_names])

        # feasibility の計算
        mean_f, std_f = self.model_f.predict(np.array([x]))
        pof = 1. - norm.cdf(0.5, loc=mean_f, scale=std_f)
        if pof < 0.5:
            raise SolveError('PoF < 0.5')

        # 実際の計算(mean は history.obj_names 順)
        mean, _ = self.model.predict(np.array([x]))

        # 目的関数の更新
        self.obj = {obj_name: value for obj_name, value in zip(self.history.obj_names, mean)}
