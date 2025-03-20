from typing import Optional, List, Any
from abc import ABC

import numpy as np
import pandas as pd

from pyfemtet.logger import get_module_logger
from pyfemtet.opt._femopt_core import History, Objective
from pyfemtet.opt.interface._base import FEMInterface
from pyfemtet.opt.optimizer._base import AbstractOptimizer

logger = get_module_logger('opt.interface', __name__)


class SurrogateModelInterfaceBase(FEMInterface, ABC):
    def __init__(
            self,
            history_path: str = None,
            train_history: History = None,
    ):

        self.train_history: History
        self.model: Any
        self.prm: dict[str, float] = dict()
        self.obj: dict[str, float] = dict()
        self.df_prm: pd.DataFrame
        self.df_obj: pd.DataFrame

        # history_path が与えられた場合、train_history をコンストラクトする
        if history_path is not None:
            train_history = History(history_path=history_path)

        # train_history が与えられるかコンストラクトされている場合
        if train_history is not None:
            # 学習データを準備する
            df_prm = train_history.get_df()[train_history.prm_names]
            df_obj = train_history.get_df()[train_history.obj_names]

            # obj の名前を作る
            for obj_name in train_history.obj_names:
                self.obj[obj_name] = np.nan

            # prm の名前を作る
            for prm_name in train_history.prm_names:
                self.prm[prm_name] = np.nan

            self.train_history = train_history

        # history から作らない場合、引数チェック
        else:
            # assert len(train_x) == len(train_y)
            raise NotImplementedError

        self.df_prm = df_prm
        self.df_obj = df_obj

        FEMInterface.__init__(
            self,
            train_history=train_history,  # コンストラクト済み train_history を渡せば並列計算時も何もしなくてよい
        )

    def filter_feasible(self, x: np.ndarray, y: np.ndarray, return_feasibility=False):
        feasible_idx = np.where(~np.isnan(y.sum(axis=1)))
        if return_feasibility:
            # calculated or not
            feas = np.zeros((len(y), 1), dtype=float)
            feas[feasible_idx] = 1.
            return x, feas
        else:
            return x[feasible_idx], y[feasible_idx]

    def _setup_after_parallel(self, *args, **kwargs):
        opt: AbstractOptimizer = kwargs['opt']
        obj: Objective

        # add_objective された目的のうち、
        # training data に含まれる名前で
        # あるものは fun を上書き
        for obj_name, obj in opt.objectives.items():
            if obj_name in self.train_history.obj_names:
                obj.fun = lambda obj_name_=obj_name: self.obj[obj_name_]

    def update_parameter(self, parameters: pd.DataFrame, with_warning=False) -> Optional[List[str]]:
        for i, row in parameters.iterrows():
            name, value = row['name'], row['value']
            self.prm[name] = value
