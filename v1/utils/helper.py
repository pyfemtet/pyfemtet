import pandas as pd


__all__ = [
    'float_',
    'apply_partial_df',
    'get_partial_df',
]


def float_(value: str | None | float) -> str | float | None:
    if value is None:
        return value
    try:
        return float(value)
    except ValueError:
        return value


def get_index(df, equality_filters):
    # フィルタ条件に一致する行のインデックスを取得
    # noinspection PyUnresolvedReferences
    return (df[list(equality_filters.keys())] == pd.Series(equality_filters)).all(axis=1)


def get_partial_df(df: pd.DataFrame, equality_filters: dict):
    return df[get_index(df, equality_filters)]


def apply_partial_df(df: pd.DataFrame, partial_df: pd.DataFrame, equality_filters: dict):

    idx = get_index(df, equality_filters)

    # インデクスに対応する部分を上書き
    assert len(df[idx]) == len(partial_df)
    df[idx] = partial_df

    return df
