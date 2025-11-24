# import
from packaging import version

import numpy as np
import optuna
if version.parse(optuna.version.__version__) < version.parse('4.0.0'):
    # noinspection PyUnresolvedReferences
    from optuna._hypervolume import WFG
    wfg = WFG()
    compute_hypervolume = wfg.compute
else:
    from optuna._hypervolume import wfg
    compute_hypervolume = wfg.compute_hypervolume

import datetime
from optuna.visualization._hypervolume_history import _get_hypervolume_history_info, _get_hypervolume_history_plot

try:
    from ._optimality import *
except ImportError:
    from _optimality import *


__all__ = [
    'calc_hypervolume',
]


def calc_hypervolume(
        y_internal: np.ndarray,
        feasibility: np.ndarray,
        ref_point: str | np.ndarray = 'optuna-nadir',
        rtol=0.01,
) -> np.ndarray:
    """

    Args:
        y_internal: n x m shaped 2d-array. float. Can contain nan.
        feasibility: n shaped 1d-array. bool.
        ref_point:
            'nadir-up-to-the-point',
            'worst-up-to-the-point',
            'nadir',
            'worst',
            or the fixed reference point of y_internal.
        rtol: Relative tolerance for optimality calculation. float.

    Returns: n shaped 1d-array. float.

    """

    # single objective
    if y_internal.shape[-1] == 1:
        return np.full_like(np.empty(len(y_internal)), np.nan, dtype=float)

    # multi objective
    if isinstance(ref_point, str):
        if ref_point.lower() == 'nadir-up-to-the-point':
            hv_values = calc_hypervolume_nadir_up_to_the_point(y_internal, feasibility, rtol)
        elif ref_point.lower() == 'worst-up-to-the-point':
            hv_values = calc_hypervolume_worst_up_to_the_point(y_internal, feasibility)
        elif ref_point.lower() == 'nadir':
            hv = calc_hypervolume_nadir(y_internal, feasibility)
            hv_values = hv * np.ones(len(y_internal)).astype(float)
        elif ref_point.lower() == 'worst':
            hv = calc_hypervolume_worst(y_internal, feasibility)
            hv_values = hv * np.ones(len(y_internal)).astype(float)
        elif ref_point.lower() == 'optuna-nadir':
            hv_values = calc_hypervolume_by_optuna(y_internal, feasibility, rtol)
        else:
            raise NotImplementedError

    elif isinstance(ref_point, np.ndarray):
        hv = calc_hypervolume_fixed_point(y_internal, feasibility, ref_point)
        hv_values = hv * np.ones(len(y_internal)).astype(float)

    else:
        raise NotImplementedError

    return hv_values


def get_pareto_set(
        y: np.ndarray,
        feasibility: np.ndarray,
        rtol,
) -> np.ndarray:
    optimality = calc_optimality(y, feasibility, rtol)
    non_dominated_solutions = y[optimality]
    assert not np.any(np.isnan(non_dominated_solutions))
    return non_dominated_solutions


def calc_hypervolume_nadir(y, feasibility, ref_point=None, rtol=0.01) -> float:
    """Use Nadir point as the ref_point.

    Args:
        y: (n, m) shaped 2d-array. float.
        feasibility (np.ndarray): n shaped 1d-array. bool.
        ref_point: (m) shaped array. float.
        rtol: Relative tolerance for optimality calculation. float.

    Returns: float.

    """

    pareto_set = get_pareto_set(y, feasibility, rtol)
    if len(pareto_set) == 0:
        return 0.

    if ref_point is None:
        nadir_point = pareto_set.max(axis=0)
        ref_point = nadir_point + 1e-8

    hv = compute_hypervolume(pareto_set, ref_point)

    return hv


def calc_hypervolume_nadir_up_to_the_point(y, feasibility, rtol) -> np.ndarray:
    """Use Nadir point up_to_the_point as the ref_point.

    Args:
        y: (n, m) shaped 2d-array. float.
        feasibility (np.ndarray): n shaped 1d-array. bool.
        rtol: Relative tolerance for optimality calculation. float.

    Returns: (n) shaped 1d-array. float.

    """

    out = []

    assert len(y) == len(feasibility)

    nadir_points = []
    for n in range(len(y)):
        y_up = y[:n]
        f_up = feasibility[:n]
        pareto_set = get_pareto_set(y_up, f_up, rtol)
        if len(pareto_set) == 0:
            continue
        nadir_points.append(pareto_set.max(axis=0))

    if len(nadir_points) == 0:
        return np.zeros(len(y))

    nadir_point = np.array(nadir_points).max(axis=0)
    ref_point = nadir_point + 1e-8

    for n in range(len(y)):
        y_up = y[:n]
        f_up = feasibility[:n]
        out.append(calc_hypervolume_nadir(y_up, f_up, ref_point=ref_point))

    return np.array(out).astype(float)


def calc_hypervolume_worst(y, feasibility) -> float:
    """Use Worst point as the ref_point.

    Args:
        y: (n, m) shaped 2d-array. float.
        feasibility (np.ndarray): n shaped 1d-array. bool.

    Returns: float.

    """

    feasible_solutions = y[feasibility]
    if len(feasible_solutions) == 0:
        return 0.

    worst_point = feasible_solutions.max(axis=0)
    ref_point = worst_point + 1e-8
    hv = compute_hypervolume(feasible_solutions, ref_point)

    return hv


def calc_hypervolume_worst_up_to_the_point(y, feasibility) -> np.ndarray:
    out = []

    assert len(y) == len(feasibility)
    for n in range(len(y)):
        y_up = y[:n]
        f_up = feasibility[:n]
        out.append(calc_hypervolume_worst(y_up, f_up))

    return np.array(out).astype(float)


def calc_hypervolume_fixed_point(y, feasibility, ref_point) -> float:

    feasible_solutions = y[feasibility]
    if len(feasible_solutions) == 0:
        return 0.

    hv = compute_hypervolume(feasible_solutions, ref_point)

    return hv


def calc_hypervolume_by_optuna(y, f, rtol):
    study = optuna.create_study(
        directions=['minimize'] * y.shape[-1]
    )
    number = 0
    for y_ in y:
        system_attrs = {}
        if np.isnan(y_).any():
            system_attrs.update({"constraints": [1.]})
            y_ = y[0]
        trial = optuna.trial.FrozenTrial(
            number=number,
            value=None,
            values=y_,
            state=optuna.trial.TrialState.COMPLETE,
            datetime_start=datetime.datetime.now(),
            datetime_complete=datetime.datetime.now(),
            params={},
            distributions={},
            user_attrs={},
            system_attrs=system_attrs,
            intermediate_values={},
            trial_id=number,
        )
        study.add_trial(trial)
        number += 1
    ref_point = get_pareto_set(y, f, rtol).max(axis=0) + 1e-8
    info = _get_hypervolume_history_info(study, reference_point=ref_point)
    return np.array(info.values)
