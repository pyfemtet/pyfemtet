from typing import Sequence

import numpy as np
from scipy.stats.distributions import norm

# from gpytorch.priors.torch_priors import GammaPrior

from pyfemtet._i18n import _
from pyfemtet.opt.history import *
from pyfemtet.opt.exceptions import *

from pyfemtet.opt.prediction._model import PyFemtetModel, SingleTaskGPModel

from pyfemtet.opt.interface._surrogate_model_interface.base_surrogate_interface import AbstractSurrogateModelInterfaceBase

from pyfemtet.logger import get_module_logger

logger = get_module_logger('opt.interface', False)


__all__ = [
    'BoTorchInterface',
    'PoFBoTorchInterface',
]


class BoTorchInterface(AbstractSurrogateModelInterfaceBase):

    current_obj_std_values: dict[str, float]

    def __init__(
            self,
            history_path: str = None,
            train_history: History = None,
            _output_directions: (
                    Sequence[str | float]
                    | dict[str, str | float]
                    | dict[int, str | float]
            ) = None
    ):
        AbstractSurrogateModelInterfaceBase.__init__(
            self,
            history_path,
            train_history,
            _output_directions
        )

        self.model = SingleTaskGPModel()
        self.pyfemtet_model = PyFemtetModel()
        self.current_obj_std_values = {}

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
        x = np.array([[variable.value for variable in self.current_prm_values.values()]])

        y, s = self.pyfemtet_model.predict(x)
        y = y[0]
        s = s[0]

        for obj_name, obj_value, obj_std in zip(self.train_history.obj_names, y, s):
            self.current_obj_values.update({obj_name: obj_value})
            self.current_obj_std_values.update({obj_name: obj_std})


class PoFBoTorchInterface(BoTorchInterface, AbstractSurrogateModelInterfaceBase):

    _debug: bool = False

    def __init__(
            self,
            history_path: str,
            train_history: History = None,
            observation_noise: float | str | None = None,
            feasibility_noise: float | str | None = None,
            feasibility_cdf_threshold: float | str = 0.5,  # or 'sample_mean'
            _output_directions: (
                    Sequence[str | float]
                    | dict[str, str | float]
                    | dict[int, str | float]
            ) = None
    ):
        AbstractSurrogateModelInterfaceBase.__init__(
            self,
            history_path,
            train_history,
            _output_directions
        )

        self.model = SingleTaskGPModel()
        self.pyfemtet_model = PyFemtetModel()
        self.model_c = SingleTaskGPModel()
        self.pyfemtet_model_c = PyFemtetModel()
        self.train_history_c = History()
        self.train_history_c.load_csv(history_path, with_finalize=True)
        self.pof_threshold = 0.5
        self.feasibility_cdf_threshold = feasibility_cdf_threshold
        self.current_obj_std_values = {}

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
            observation_noise=observation_noise,
        )

        # training model_c
        self.pyfemtet_model_c.update_model(self.model_c)
        self.pyfemtet_model_c.fit(
            history=self.train_history_c,
            df=df_c,
            # observation_noise=None,
            # observation_noise='no',
            # observation_noise=0.001,
            observation_noise=feasibility_noise,
            # covar_module_settings=dict(
            #     name='matern_kernel_with_gamma_prior',
            #     nu=2.5,
            #     lengthscale_prior=GammaPrior(1.0, 9.0),  # default: 3, 6
            #     outputscale_prior=GammaPrior(1.0, 0.15),  # default: 2, 0.15
            # )
            covar_module_settings=dict(
                name='get_covar_module_with_dim_scaled_prior_extension',
                loc_coef=0.01,
                scale_coef=0.01,
            )
        )

        # set auto feasibility_cdf_threshold
        if self.feasibility_cdf_threshold == 'sample_mean':
            self.feasibility_cdf_threshold = df_c['feasibility'].mean()

        if self._debug:
            self._debug_df_c = df_c

    def calc_pof(self):

        if self._debug:
            import plotly.graph_objects as go

            df = self._debug_df_c

            x_list = []
            prm_names = self.train_history_c.prm_names
            for prm_name in prm_names:
                x_list.append(np.linspace(
                    df[prm_name + '_lower_bound'].values[0],
                    df[prm_name + '_upper_bound'].values[0],
                    20
                ))

            for i in range(len(x_list)):
                for j in range(i, len(x_list)):
                    if i == j:
                        continue

                    # i=0
                    # j=1

                    xx = np.meshgrid(x_list[i], x_list[j])

                    x_plot = np.array([[x[0]] * 400 for x in x_list]).T
                    x_plot[:, i] = xx[0].ravel()
                    x_plot[:, j] = xx[1].ravel()

                    y_mean, y_std = self.pyfemtet_model_c.predict(x_plot)
                    # feasibility_cdf_threshold = self.feasibility_cdf_threshold
                    # feasibility_cdf_threshold = 0.5
                    cdf_threshold = 0.25  # 不明なところは pof が 1 近くにすればあとは ACQF がうまいことやってくれる
                    # feasibility_cdf_threshold = self.feasibility_cdf_threshold * 0.5
                    pof = 1 - norm.cdf(cdf_threshold, y_mean, y_std)

                    x1 = x_list[i]
                    x2 = x_list[j]
                    xx1 = xx[0]
                    xx2 = xx[1]

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
                            x=self._debug_df_c[prm_names[i]],
                            y=self._debug_df_c[prm_names[j]],
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
                            x=self._debug_df_c[prm_names[i]],
                            y=self._debug_df_c[prm_names[j]],
                            z=self._debug_df_c['feasibility'],
                            mode='markers',
                        )
                    )
                    fig.show()

            return

        x = np.array([[variable.value for variable in self.current_prm_values.values()]])

        f_mean, f_std = self.pyfemtet_model_c.predict(x)
        f_mean, f_std = f_mean[0][0], f_std[0][0]

        if isinstance(self.feasibility_cdf_threshold, float):
            cdf_threshold = self.feasibility_cdf_threshold
        else:
            raise NotImplementedError(
                f'self.cdf_threshold must be float, '
                f'passed {self.feasibility_cdf_threshold}'
            )

        pof = 1 - norm.cdf(cdf_threshold, f_mean, f_std)

        return pof

    def update(self) -> None:

        # BoTorchInterface.update() の前に PoF を計算する
        pof = self.calc_pof()
        if pof < self.pof_threshold:
            logger.info(
                _(
                    en_message='The surrogate model estimated '
                               'that the probability of '
                               'feasibility (PoF) is {pof}. '
                               'This is under {thresh}. '
                               'So this trial is processed as '
                               'a constraint violation.',
                    jp_message='サロゲートモデルは解の実行可能確率（PoF）が'
                               '{pof} であると予測しました。'
                               'これは閾値 {thresh} を下回っているので、'
                               '最適化試行においては拘束違反であると扱います。',
                    pof=pof,
                    thresh=self.pof_threshold,
                )
            )
            raise _HiddenConstraintViolation(f'PoF < {self.pof_threshold}')

        BoTorchInterface.update(self)
