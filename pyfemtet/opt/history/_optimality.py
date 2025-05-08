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

    # 「全ての項目において
    # 重複する解を除いて
    # より優れているか又は同等である
    # 他の解が存在しない解」であるかどうかを判定
    y_values: np.ndarray
    another_y_values: np.ndarray
    optimality = []
    assert len(y_internal) == len(feasibility)
    for i, (y_values, feas) in enumerate(zip(y_internal, feasibility)):
        for j, (another_y_values, another_feas) in enumerate(zip(y_internal, feasibility)):

            # 自身が infeasible なら
            # 比較を終了して False
            if not feas:
                optimality.append(False)
                break

            # 比較相手が infeasible なら
            # 比較しない
            elif not another_feas:
                continue

            # 自身との比較はしない
            elif i == j:
                continue

            # 重複した解なら比較しない
            elif np.allclose(y_values, another_y_values, atol=0, rtol=0.01):
                assert np.all(~np.isnan(y_values))
                assert np.all(~np.isnan(another_y_values))
                continue

            # 全項目について another のほうが
            # 優れているか又は同等であるなら
            # 比較を終了して False
            elif all(another_y_values <= y_values):
                optimality.append(False)
                break

            # その他の場合、比較を続行
            else:
                pass

        # 自身以外のすべての解と比較して
        # optimality が False になるような
        # ことがなかったので True
        else:
            optimality.append(True)

    return np.array(optimality).astype(bool)


if __name__ == '__main__':
    _optimality = calc_optimality(
        (_y := np.random.rand(10, 2)),
        (_feas := (np.random.rand(10) > 0.25).astype(bool))
    )

    print(_y)
    print(_feas)
    print(_optimality)
