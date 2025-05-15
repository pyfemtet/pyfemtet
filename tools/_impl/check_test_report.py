"""Pre-release check.

テスト実行中フラグより後にできた
テスト成功レポートが存在するか
"""

import os
import sys
import datetime
from glob import glob
try:
    from run_tests import (
        test_result_yaml_timestamp_format as ts_format,
    )
    from test_done import (
        get_test_start,
        success_report_file_suffix as report_suffix,
    )
except ModuleNotFoundError:
    from .run_tests import (
        test_result_yaml_timestamp_format as ts_format,
    )
    from .test_done import (
        get_test_start,
        success_report_file_suffix as report_suffix,
    )


if __name__ == '__main__':
    flag_path = sys.argv[1]

    # エラーハンドリング
    if not os.path.isfile(flag_path):
        raise FileNotFoundError(f'"{flag_path}" does not exist.')

    # テスト開始時刻を取得する
    test_start = get_test_start(flag_path)

    # テスト成功レポートのフォーマットに合致する
    # ファイル名を取得する
    valid_paths = []
    print(report_suffix)
    for path in glob(f'*{report_suffix}.yaml'):

        # ファイル名から suffix を除き
        # 時刻フォーマットで生成時刻を取得する
        timestamp = os.path.splitext(os.path.basename(path))[0].removesuffix(report_suffix)
        generated = datetime.datetime.strptime(timestamp, ts_format)

        # 終了レポートの生成時刻がテスト開始時刻より後なら OK
        if generated >= test_start:
            valid_paths.append(path)

    # 有効な終了レポートがあれば終了
    if len(valid_paths) > 0:
        return_code = 0
    else:
        return_code = 1
    sys.exit(return_code)
