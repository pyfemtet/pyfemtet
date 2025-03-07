import numpy as np


__all__ = ['calc_optimality']


def calc_optimality(y_internal: np.ndarray, feasibility: np.ndarray) -> np.ndarray:
    """

    Args:
        y_internal (np.ndarray): n x m shaped 2d-array. Can contain np.nan. Minimum value is optimal.
        feasibility (np.ndarray): n shaped 1d-array. bool.

    Returns:
        np.ndarray: Array if not optimal, dominated or Nan False, else True

    """

    # 非劣解の計算
    non_dominated = [
        (not np.isnan(y).any())  # feasible (duplicated)
        and
        (not (y > y_internal).all(axis=1).any())  # not dominated
        and
        feas  # feasible
        for y, feas in zip(y_internal, feasibility)
    ]

    return np.array(non_dominated).astype(bool)
