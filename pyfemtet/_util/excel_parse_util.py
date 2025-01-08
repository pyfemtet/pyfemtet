import os
import numpy as np
import pandas as pd


__all__ = [
    'ParseAsParameter',
    'ParseAsConstraint',
    'ParseAsObjective',
    'search_index',
    'search_r',
    'search_c',
]


def parse_excel(book_path, sheet_name, keyword, required, optional) -> pd.DataFrame:
    """Excel シートからパラメータを取得します。

    シートのパースプロセスは以下の通りです。

    1. シート全体 (A1 セルから、値が入力されている最終セルまで) をデータに取り込みます。
    2. すべてのセルが空白である列をデータから除きます。
    3. すべてのセルが空白である行をデータから除きます。
    4. 最も左上（上が優先）にある keyword に一致するセルより上および左の行・列をデータから除きます。

    Args:
        book_path: Excel book のパス。
        sheet_name: シート名。
        keyword (str): 必ず含まれるべき、表データの最初の列名として使う文字列。
        required (list[str]): 必ず含まれるべき、表データの列名として使う文字列のリスト。
        optional (list[str]): 表データの列名として使ってよい文字列のリスト。

    Returns:

    """

    # 存在チェック
    if not os.path.exists(book_path):
        raise FileNotFoundError(book_path)

    # 読み込み
    df = pd.read_excel(book_path, sheet_name, header=None)

    # NaN のみからなる列を削除する
    valid_columns = [col for col in df.columns if df[col].notna().sum()]
    df = df[valid_columns]

    # NaN のみからなる行を削除する
    valid_rows = [row for row in df.index if df.loc[row].notna().sum()]
    df = df.loc[valid_rows]

    # 「変数名」を左上とする表にする
    df: pd.DataFrame
    idx = np.where(df.values == keyword)
    r = idx[0][0]
    c = idx[1][0]
    df = pd.DataFrame(df.iloc[1+r:, c:].values, columns=df.iloc[r, c:].values)

    # パースが成功しているかチェックする
    lack = True
    for col in df.columns:
        lack *= col == keyword
        lack *= col in required
    if lack:
        raise RuntimeError(f'Some required keywords are lacked. '
                           f'Required keywords are {keyword} (required and must be first) and '
                           f'{required} (required), '
                           f'and {optional} is optional.')

    return df


class ParseBase:
    KEYWORD = ''
    REQUIRED_COLUMNS = []
    OPTIONAL_COLUMNS = []

    @classmethod
    def parse(cls, book_path, sheet_name) -> pd.DataFrame:
        return parse_excel(book_path, sheet_name, cls.KEYWORD, cls.REQUIRED_COLUMNS, cls.OPTIONAL_COLUMNS)


class ParseAsParameter(ParseBase):
    name = '変数名'
    value = '値'
    lb = '下限'
    ub = '上限'
    step = 'ステップ'
    use = '使用'
    KEYWORD = name
    REQUIRED_COLUMNS = [value, lb, ub]
    OPTIONAL_COLUMNS = [step, use]


class ParseAsObjective(ParseBase):
    name = '目的名'
    value = '値'
    direction = '目標'
    use = '使用'
    KEYWORD = name
    REQUIRED_COLUMNS = [value, direction]
    OPTIONAL_COLUMNS = [use]


class ParseAsConstraint(ParseBase):
    name = '拘束名'
    value = '値'
    lb = '下限'
    ub = '上限'
    strict = '厳守'
    use = '使用'
    calc_before_solve = 'ソルブ前に計算'
    KEYWORD = name
    REQUIRED_COLUMNS = [value]
    OPTIONAL_COLUMNS = [lb, ub, strict, use, calc_before_solve]


def search_index(book_path, sheet_name, value):
    df = pd.read_excel(book_path, sheet_name, header=None)
    idx = np.where(df.values == value)
    r = idx[0][0]
    c = idx[1][0]
    return r, c


def search_r(book_path, sheet_name, value):
    return search_index(book_path, sheet_name, value)[0]


def search_c(book_path, sheet_name, value):
    return search_index(book_path, sheet_name, value)[1]


if __name__ == '__main__':
    os.chdir(os.path.dirname(__file__))
    print(ParseAsParameter.parse('sample.xlsx', 'Sheet1'))
    print(search_r('sample.xlsx', 'Sheet1', 'X2'))
    print(search_c('sample.xlsx', 'Sheet1', '値'))
