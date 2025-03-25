import numpy as np
from scipy.stats.distributions import norm

from gpytorch.priors.torch_priors import GammaPrior

from pyfemtet.opt.history import *
from pyfemtet.opt.prediction.model import PyFemtetModel, SingleTaskGPModel
from pyfemtet.opt.exceptions import *  # should import after dask importing

from pyfemtet.opt.interface.surrogate_model.surrogate_interface import AbstractSurrogateModelInterfaceBase

from pyfemtet.logger import get_module_logger

logger = get_module_logger('opt.interface', False)

debug = False


class BoTorchInterface(AbstractSurrogateModelInterfaceBase):

    def __init__(self, history_path: str = None, train_history: History = None):
        AbstractSurrogateModelInterfaceBase.__init__(self, history_path, train_history)

        self.model = SingleTaskGPModel()
        self.pyfemtet_model = PyFemtetModel()

        # get main only
        df = self.train_history.get_df(MAIN_FILTER)

        # filter succeeded only
        df = df[df['state'] == TrialState.succeeded]

        # training
        self.pyfemtet_model.update_model(self.model)
        self.pyfemtet_model.fit(
            history=self.train_history,
            df=df,
            observation_noise='no',
        )

    def update(self) -> None:
        # update current objective values
        x = np.array([self.current_prm_values.values()])

        y, _ = self.pyfemtet_model.predict(x)
        y = y[0]

        for obj_name, obj_value in zip(self.train_history.obj_names, y):
            self.current_obj_values.update({obj_name: obj_value})


class PoFBoTorchInterface(BoTorchInterface, AbstractSurrogateModelInterfaceBase):

    def __init__(self, history_path: str):
        AbstractSurrogateModelInterfaceBase.__init__(self, history_path, None)

        self.model = SingleTaskGPModel()
        self.pyfemtet_model = PyFemtetModel()
        self.model_c = SingleTaskGPModel()
        self.pyfemtet_model_c = PyFemtetModel()
        self.train_history_c = History()
        self.train_history_c.load_csv(history_path, with_finalize=True)
        self.pof_threshold = 0.5
        self.cdf_threshold = 0.25  # float or 'sample_mean'

        # use feasibility as a single objective
        self.train_history_c.obj_names = ['feasibility']

        # get main only
        df = self.train_history.get_df(MAIN_FILTER)
        df_c = self.train_history_c.get_df(MAIN_FILTER)

        # filter succeeded only for main
        df = df[df['state'] == TrialState.succeeded]

        # convert type bool to float
        df_c = df_c.astype({'feasibility': float})

        # training main
        self.pyfemtet_model.update_model(self.model)
        self.pyfemtet_model.fit(
            history=self.train_history,
            df=df,
            observation_noise='no',
        )

        # training model_c
        self.pyfemtet_model_c.update_model(self.model_c)
        self.pyfemtet_model_c.fit(
            history=self.train_history_c,
            df=df_c,
            observation_noise=None,
            covar_module_settings=dict(
                name='matern_kernel_with_gamma_prior',
                nu=2.5,
                lengthscale_prior=GammaPrior(1.0, 9.0),  # default: 3, 6
                outputscale_prior=GammaPrior(1.0, 0.15),  # default: 2, 0.15
            )
        )

        # set auto cdf_threshold
        if self.cdf_threshold == 'sample_mean':
            self.cdf_threshold = df_c['feasibility'].mean()

        if debug:
            self._debug_df_c = df_c

    def calc_pof(self):
        if debug:
            import plotly.graph_objects as go

            x1 = np.linspace(-1, 1, 21)
            x2 = np.linspace(-1, 1, 21)
            xx1, xx2 = np.meshgrid(x1, x2)
            x_plot = np.array([xx1.ravel(), xx2.ravel()]).T
            y_mean, y_std = self.pyfemtet_model_c.predict(x_plot)
            # cdf_threshold = self.cdf_threshold
            # cdf_threshold = 0.5
            cdf_threshold = 0.25  # 不明なところは pof が 1 近くにすればあとは ACQF がうまいことやってくれる
            # cdf_threshold = self.cdf_threshold * 0.5
            pof = 1 - norm.cdf(cdf_threshold, y_mean, y_std)

            fig = go.Figure()
            fig.add_trace(
                go.Contour(
                    x0=x1[0],
                    y0=x2[0],
                    dx=np.diff(x1)[0],
                    dy=np.diff(x2)[0],
                    z=pof.reshape(xx1.shape),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=self._debug_df_c['x1'],
                    y=self._debug_df_c['x2'],
                    mode='markers',
                    marker=dict(
                        color=self._debug_df_c['feasibility'],
                        colorscale='greens',
                    ),
                )
            )
            fig.show()

            fig = go.Figure()
            fig.add_trace(
                go.Surface(
                    x=xx1,
                    y=xx2,
                    z=y_mean.reshape(xx1.shape),
                )
            )
            fig.add_trace(
                go.Surface(
                    x=xx1,
                    y=xx2,
                    z=(y_mean + y_std).reshape(xx1.shape),
                    opacity=0.3,
                )
            )
            fig.add_trace(
                go.Surface(
                    x=xx1,
                    y=xx2,
                    z=(y_mean - y_std).reshape(xx1.shape),
                    opacity=0.3,
                )
            )
            fig.add_trace(
                go.Scatter3d(
                    x=self._debug_df_c['x1'],
                    y=self._debug_df_c['x2'],
                    z=self._debug_df_c['feasibility'],
                )
            )
            fig.show()

            return

        x = np.array([self.current_prm_values.values()])

        f_mean, f_std = self.pyfemtet_model_c.predict(x)
        f_mean, f_std = f_mean[0][0], f_std[0][0]

        if isinstance(self.cdf_threshold, float):
            cdf_threshold = self.cdf_threshold
        else:
            raise NotImplementedError(
                f'self.cdf_threshold must be float, '
                f'passed {self.cdf_threshold}'
            )

        pof = 1 - norm.cdf(cdf_threshold, f_mean, f_std)

        if pof < self.pof_threshold:
            logger.info(f'サロゲートモデルによって、入力変数 {self.current_prm_values} '
                        f'に対して実モデルが成立する確率が {pof:.2f} だと推定されました。'
                        f'この値が PoF 閾値 {self.pof_threshold} を下回っているので、'
                        f'拘束違反エラーとして扱います。')
            raise HiddenConstraintViolation(f'PoF < {self.pof_threshold}')

    def update(self) -> None:

        # BoTorchInterface.update() の前に PoF を計算する
        self.calc_pof()

        BoTorchInterface.update(self)


def debug_1():

    import os

    fem = PoFBoTorchInterface(
        history_path=os.path.join(
            os.path.dirname(__file__),
            'debug-pof-botorch.reccsv'
        )
    )
    fem.update_parameter(dict(x1=70, x2='A'))
    fem.update()
    print(fem.current_obj_values)


if __name__ == '__main__':
    debug_1()
