import numpy as np
import pandas as pd

from optuna.distributions import (
    FloatDistribution,
    CategoricalDistribution,
    BaseDistribution,
    CategoricalChoiceType
)
from optuna._transform import _SearchSpaceTransform

from pyfemtet.opt.history import *

__all__ = [
    'get_bounds_containing_entire_bounds',
    'get_choices_containing_entire_bounds',
    'get_params_list',
    'get_params_list_from_ndarray',
    'get_search_space',
    'get_transform_0_1',
    'get_transformed_params',
]


def get_bounds_containing_entire_bounds(
        df: pd.DataFrame, prm_name
) -> tuple[float | None, float | None]:
    """Get param bounds with all bounds"""

    lb_name = CorrespondingColumnNameRuler.prm_lower_bound_name(prm_name)
    lb = df[lb_name].dropna().min()

    ub_name = CorrespondingColumnNameRuler.prm_upper_bound_name(prm_name)
    ub = df[ub_name].dropna().max()

    lb = float(lb)
    ub = float(ub)

    if np.isnan(lb):
        lb = None
    if np.isnan(ub):
        ub = None

    return lb, ub


def get_choices_containing_entire_bounds(df: pd.DataFrame, prm_name) -> set[CategoricalChoiceType]:
    """Get param bounds with all choices"""

    choices_name = CorrespondingColumnNameRuler.prm_choices_name(prm_name)
    choices_values = df[choices_name].values
    choices = set()
    for choices_ in choices_values:
        for c in choices_:
            choices.add(c)
    return choices


def get_params_list(df: pd.DataFrame, history: History) -> list[dict[str, ...]]:
    # get search space and parameters
    params_list: list[dict[str, ...]] = []

    # update params_list
    for i, row in df.iterrows():

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


def get_search_space(
        df: pd.DataFrame, history: History
) -> dict[str, BaseDistribution | None]:
    # get search_space
    search_space = dict()
    for prm_name in history.prm_names:
        if history._records.column_manager.is_numerical_parameter(prm_name):
            lb, ub = get_bounds_containing_entire_bounds(df, prm_name)

            if lb is None:
                lb = df[prm_name].dropna().min()
            if ub is None:
                ub = df[prm_name].dropna().max()

            if lb == ub:
                # 固定値は search_space に含めない
                pass

            else:
                search_space.update({
                    prm_name: FloatDistribution(
                        low=lb, high=ub,
                    )
                })

        elif history._records.column_manager.is_categorical_parameter(prm_name):
            choices: set = get_choices_containing_entire_bounds(df, prm_name)
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

    out = np.empty(
        (
            0,
            len(trans.encoded_column_to_column)  # ここ。choices ぶん増やす
        ),
        dtype=float
    )

    # get list of params
    if isinstance(df_or_x, pd.DataFrame):
        params_list = get_params_list(df_or_x, history)
    elif isinstance(df_or_x, np.ndarray):
        params_list = get_params_list_from_ndarray(df_or_x, history)
    else:
        raise NotImplementedError

    # calc transformed value
    for params in params_list:

        # search_space に含まれない params の key は
        # trans.transform() で無視される
        trans_prm_values: np.ndarray = trans.transform(params)
        out = np.concatenate(
            [out, [trans_prm_values]],
            dtype=float,
            axis=0,
        )

    return out
