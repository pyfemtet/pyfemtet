from typing import TypeAlias, Literal
from math import isnan
import pandas as pd


__all__ = [
    'apply_partial_df',
    'get_partial_df',
]


_Method: TypeAlias = Literal['all', 'any', 'all-exclude']


def get_index(df, equality_filters, method: _Method = 'all'):
    # 値渡しに変換
    equality_filters = equality_filters.copy()

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
    if 'all' in method.lower():
        out: pd.Series = (df[list(equality_filters.keys())] == pd.Series(equality_filters)).all(axis=1)
    elif 'any' in method.lower():
        out: pd.Series = (df[list(equality_filters.keys())] == pd.Series(equality_filters)).any(axis=1)
    else:
        raise NotImplementedError(f'Unknown method: {method}')

    # na との比較
    for key in want_na_keys:
        out = out & df[key].isna()

    if 'exclude' in method:
        out = ~out

    return out


def get_partial_df(df: pd.DataFrame, equality_filters: dict, method: _Method = 'all'):
    return df[get_index(df, equality_filters, method=method)]


def apply_partial_df(df: pd.DataFrame, partial_df: pd.DataFrame, equality_filters: dict):

    idx = get_index(df, equality_filters)

    # インデクスに対応する部分を上書き
    assert len(df[idx]) == len(partial_df), ('equality_filters の実行結果と'
                                             '与えられた partial_df の長さが一致しません。')
    df[idx] = partial_df

    return df
