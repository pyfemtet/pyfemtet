from math import isnan
import pandas as pd


__all__ = [
    'apply_partial_df',
    'get_partial_df',
]


def get_index(df, equality_filters):
    # フィルタ条件に一致する行のインデックスを取得

    # na との == での比較は常に False なので別処理するために別リストを作る
    want_na_keys = []
    for key, value in equality_filters.items():
        if isinstance(value, float):
            if isnan(value):
                want_na_keys.append(key)
    [equality_filters.pop(key) for key in want_na_keys]

    # na 以外の比較
    # noinspection PyUnresolvedReferences
    out: pd.Series = (df[list(equality_filters.keys())] == pd.Series(equality_filters)).all(axis=1)

    # na との比較
    for key in want_na_keys:
        out = out & df[key].isna()

    return out


def get_partial_df(df: pd.DataFrame, equality_filters: dict):
    return df[get_index(df, equality_filters)]


def apply_partial_df(df: pd.DataFrame, partial_df: pd.DataFrame, equality_filters: dict):

    idx = get_index(df, equality_filters)

    # インデクスに対応する部分を上書き
    assert len(df[idx]) == len(partial_df), ('equality_filters の実行結果と'
                                             '与えられた partial_df の長さが一致しません。')
    df[idx] = partial_df

    return df
