import numpy as np
import pandas as pd

from pyfemtet.logger import get_module_logger
from pyfemtet.opt.interface._surrogate._base import SurrogateModelInterfaceBase
from pyfemtet.opt.prediction.polynomial_expansion import PolynomialExpansionModel


logger = get_module_logger('opt.interface', __name__)


class PolynomialChaosInterface(SurrogateModelInterfaceBase):

    model: PolynomialExpansionModel  # surrogate_func_expansion

    def train(self):
        x, y = self.filter_feasible(self.df_prm.values, self.df_obj.values)
        assert len(x) > 0, 'No feasible results in training data.'
        self.model.fit(x, y)

    def _setup_after_parallel(self, *args, **kwargs):

        # # update objectives
        # SurrogateModelInterfaceBase._setup_after_parallel(
        #     self, *args, **kwargs
        # )

        # train model
        self.model = PolynomialExpansionModel()
        self.model.set_bounds_from_history(self.history)
        self.train()

    def update(self, parameters: pd.DataFrame) -> None:
        # self.prm 更新
        SurrogateModelInterfaceBase.update_parameter(
            self, parameters
        )

        # history.prm_name 順に並べ替え
        x = np.array([self.prm[k] for k in self.history.prm_names])

        # prediction
        dist_mean, _ = self.model.predict(x)

        # 目的関数の更新
        self.obj = {obj_name: value for obj_name, value in zip(self.history.obj_names, dist_mean)}


if __name__ == '__main__':
    import os
    from pyfemtet.opt._femopt_core import History

    os.chdir(os.path.dirname(__file__))

    history = History(history_path='sample.csv')
    fem = PolynomialChaosInterface(history=history)

    import pandas as pd
    df = history.get_df()
    parameters = pd.DataFrame(
        dict(
            name=history.prm_names,
            value=[1. for _ in history.prm_names]
        )
    )

    fem._setup_after_parallel()

    fem.update(
        parameters
    )
