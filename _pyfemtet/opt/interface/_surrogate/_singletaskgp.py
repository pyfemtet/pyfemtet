import numpy as np
import pandas as pd
from scipy.stats.distributions import norm

from pyfemtet.core import SolveError
from pyfemtet.logger import get_module_logger
from pyfemtet.opt.interface._surrogate._base import SurrogateModelInterfaceBase
from pyfemtet.opt.prediction.single_task_gp import SingleTaskGPModel

from pyfemtet._message.messages import Message as Msg


logger = get_module_logger('opt.interface', __name__)


class PoFBoTorchInterface(SurrogateModelInterfaceBase):
    model_f: SingleTaskGPModel
    model: SingleTaskGPModel
    threshold = 0.5

    def train(self):
        # df そのまま用いて training する
        x, y = self.filter_feasible(self.df_prm.values, self.df_obj.values)
        assert len(x) > 0, 'No feasible results in training data.'
        self.model.fit(x, y)

    def train_f(self):
        # df そのまま用いて training する
        x, y = self.filter_feasible(self.df_prm.values, self.df_obj.values, return_feasibility=True)
        if y.min() == 1:  # feasible values only
            self.model_f.predict = lambda *args, **kwargs: (1., 0.001)  # mean, std
        self.model_f.fit(x, y)

    def _setup_after_parallel(self, *args, **kwargs):

        # update objectives
        SurrogateModelInterfaceBase._setup_after_parallel(
            self, *args, **kwargs
        )

        # model training
        self.model = SingleTaskGPModel()
        self.model.set_bounds_from_history(self.train_history)
        self.train()

        # model_f training
        self.model_f = SingleTaskGPModel(is_noise_free=False)
        self.model_f.set_bounds_from_history(self.train_history)
        self.train_f()

    def update(self, parameters: pd.DataFrame) -> None:
        # self.prm 更新
        SurrogateModelInterfaceBase.update_parameter(
            self, parameters
        )

        # train_history.prm_name 順に並べ替え
        x = np.array([self.prm[k] for k in self.train_history.prm_names])

        # feasibility の計算
        mean_f, std_f = self.model_f.predict(np.array([x]))
        pof = 1. - norm.cdf(self.threshold, loc=mean_f, scale=std_f)
        if pof < self.threshold:
            raise SolveError(Msg.INFO_POF_IS_LESS_THAN_THRESHOLD)

        # 実際の計算(現時点で mean は train_history.obj_names 順)
        _mean, _std = self.model.predict(np.array([x]))
        mean = _mean[0]

        # 目的関数の更新
        self.obj = {obj_name: value for obj_name, value in zip(self.train_history.obj_names, mean)}
