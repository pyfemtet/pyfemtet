import numpy as np
import pandas as pd

from optuna.distributions import (
    FloatDistribution,
    CategoricalDistribution,
    BaseDistribution,
    CategoricalChoiceType
)
from optuna._transform import _SearchSpaceTransform

import torch

from v1.problem import *
from v1.history import *
from v1.optimizer.optuna_optimizer.pof_botorch.pof_botorch_sampler import setup_gp, SingleTaskGP


def get_bounds(df: pd.DataFrame, prm_name) -> tuple[float, float]:
    """Get param bounds with all bounds"""

    lb_name = CorrespondingColumnNameRuler.prm_lower_bound_name(prm_name)
    lb = df[lb_name].dropna().min()

    ub_name = CorrespondingColumnNameRuler.prm_upper_bound_name(prm_name)
    ub = df[ub_name].dropna().max()

    return float(lb), float(ub)


def get_choices(df: pd.DataFrame, prm_name) -> set[CategoricalChoiceType]:
    """Get param bounds with all choices"""

    choices_name = CorrespondingColumnNameRuler.prm_choices_name(prm_name)
    choices_values = df[choices_name].values
    choices = set()
    for choices_ in choices_values:
        choices.add(choices_)
    return choices


def get_params_list(df: pd.DataFrame, history: History) -> list[dict[str, ...]]:
    # get search space and parameters
    params_list: list[dict[str, ...]] = []

    # update params_list
    for i, row in df:

        # create search space and parameter values
        params = dict()

        # inspect each parameter
        for prm_name in history.prm_names:

            # parameter value
            params.update({prm_name: row[prm_name]})

        # append params_list
        params_list.append(params)

    return params_list


def get_params_list_from_ndarray(x: np.ndarray, history: History) -> list[dict[str, ...]]:
    # get search space and parameters
    params_list: list[dict[str, ...]] = []

    # update params_list
    for prm_values in x:

        # create search space and parameter values
        params = dict()

        # parameter value
        params.update(
            {name: value for name, value
             in zip(history.prm_names, prm_values)}
        )

        # append params_list
        params_list.append(params)

    return params_list


def get_search_space(df: pd.DataFrame, history: History) -> dict[str, BaseDistribution]:
    # get search_space
    search_space = dict()
    for prm_name in history.prm_names:
        if Record.is_numerical_parameter(prm_name):
            lb, ub = get_bounds(df, prm_name)
            search_space.update({
                prm_name: FloatDistribution(
                    low=lb, high=ub,
                )
            })
        elif Record.is_categorical_parameter(prm_name):
            choices: set = get_choices(df, prm_name)
            search_space.update({
                prm_name: CategoricalDistribution(
                    choices=tuple(choices)
                )
            })
        else:
            raise NotImplementedError
    return search_space


def get_transform_0_1(df: pd.DataFrame, history: History) -> _SearchSpaceTransform:
    # get search_space
    search_space = get_search_space(df, history)

    # get transform
    trans = _SearchSpaceTransform(search_space, transform_0_1=True)

    return trans


def get_transformed_params(
        df_or_x: pd.DataFrame | np.ndarray, history: History, trans: _SearchSpaceTransform
) -> np.ndarray:
    out = np.empty((0, len(history.prm_names)), dtype=float)

    # get list of params
    if isinstance(df_or_x, pd.DataFrame):
        params_list = get_params_list(df_or_x, history)
    elif isinstance(df_or_x, np.ndarray):
        params_list = get_params_list_from_ndarray(df_or_x, history)
    else:
        raise NotImplementedError

    # calc transformed value
    for params in params_list:
        trans_prm_values: np.ndarray = trans.transform(params)
        out = np.concatenate(
            [out, [trans_prm_values]],
            dtype=float,
            axis=0,
        )

    return out


class AbstractModel:

    def fit(self, x: np.ndarray, y: np.ndarray, bounds: np.ndarray = None, **kwargs): ...
    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ...


class SingleTaskGPModel(AbstractModel):

    KWARGS = dict(dtype=torch.float64, device='cpu')
    model: SingleTaskGP

    def fit(self, x: np.ndarray, y: np.ndarray, bounds: np.ndarray = None,
            observation_noise=None, likelihood_class=None):
        X = torch.tensor(x, **self.KWARGS)
        Y = torch.tensor(y, **self.KWARGS)
        B = torch.tensor(bounds, **self.KWARGS).transpose(1, 0)
        self.model = setup_gp(X, Y, B, observation_noise, likelihood_class)

    def predict(self, x: np.ndarray):
        X = torch.tensor(x, **self.KWARGS)
        post = self.model.posterior(X)
        with torch.no_grad():
            mean = post.mean.numpy()
            std = post.variance.sqrt().numpy()
        return mean, std


class PyFemtetModel:

    current_trans: _SearchSpaceTransform
    current_model: AbstractModel
    history: History

    def set_history(self, history: History):
        self.history = history

    def set_model(self, model: AbstractModel):
        self.current_model = model

    def fit(self, history: History, **kwargs):
        assert hasattr(self, 'current_model')

        # get df to create figure
        df = history.get_df(
            equality_filters={'sub_fidelity_name': MAIN_FIDELITY_NAME}
        )

        # set current trans
        self.current_trans = get_transform_0_1(df, history)

        # transform all values
        transformed_x = get_transformed_params(df, history, self.current_trans)

        # bounds as setup maximum range
        bounds = np.array([[0, 1]] * len(history.prm_names), dtype=float)

        y = df[history.obj_names].values
        self.current_model.fit(transformed_x, y, bounds, **kwargs)

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        assert hasattr(self, 'history')
        assert hasattr(self, 'current_trans')

        transformed_x = get_transformed_params(x, self.history, self.current_trans)
        return self.current_model.predict(transformed_x)
