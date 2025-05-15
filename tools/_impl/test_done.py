"""Check results and done test.

If all test results are passed,
    create a simple report file,
    remove the .pytest_cache file,
    and return 0.
Else, leave it existing anr return 1.

Args:
    argv[1]: Path to '.test-running' flag file.

"""

import os
import sys
import datetime
import subprocess
from glob import glob

import yaml

try:
    from notify import send_mail
    from run_tests import (
        test_result_yaml_timestamp_format as ts_format,
        test_result_yaml_filename_suffix as yaml_suffix
    )
except ModuleNotFoundError:
    from .notify import send_mail
    from .run_tests import (
        test_result_yaml_timestamp_format as ts_format,
        test_result_yaml_filename_suffix as yaml_suffix
    )


success_report_file_suffix = '_pyfemtet_final_test_record'


def collect_yaml_paths(test_start_):

    out = []

    for filepath in glob(f'*{yaml_suffix}.yaml'):

        # ファイル名から日時をパースする（例: 20250515155320.yaml など）
        filename = os.path.basename(filepath)
        timestamp = filename.removesuffix(f'{yaml_suffix}.yaml')

        # noinspection PyBroadException
        try:
            # 日時部分の抽出（例: 先頭14文字が日時）
            file_datetime = datetime.datetime.strptime(
                timestamp, ts_format
            )

        except Exception:
            continue  # フォーマットに合わないファイルは無視する

        if file_datetime >= test_start_:
            print(filepath)
            out.append(filepath)

    return out


def check_yamls(yaml_paths_):

    # collect results of done tests
    results: dict[str, bool] = dict()
    for yaml_path in yaml_paths_:

        # load
        with open(yaml_path) as f_:
            data = yaml.safe_load(f_)

        # summarize
        done_test_names = data.get('items', []) or []
        done_results = data.get('results', []) or []

        for done_name, done_result in zip(done_test_names, done_results):
            result: str = done_result.get('call', 'failed') or 'failed'
            results[done_name] = True if result == 'passed' else False

    # get all test names from pytest command
    cmd = [sys.executable, '-m', 'pytest', '--collect-only', '-q']
    comp_proc: subprocess.CompletedProcess = \
        subprocess.run(cmd, capture_output=True, text=True, check=True)

    # pytest の出力結果からテスト名を取得
    # 例: pytest の --collect-only -q はテスト名を1行ずつ表示するので
    all_test_names = [l.strip().split('::')[-1]
                      for l in comp_proc.stdout.splitlines()
                      if l.strip()][:-1]

    # check
    all_results = {name: results.get(name, False) for name in all_test_names}
    out = all(all_results.values())

    print('All test names:', all_test_names)
    print('Done results:', results)
    print('All results:', all_results)
    print('Test passed:', out)

    return out, all_results


def create_report(
    test_start_: datetime.datetime,
    all_results_dict_: dict,
    _yaml_file_paths_: list[str],
):
    """Create report yaml file."""

    test_end_ = datetime.datetime.now()
    test_start_str = test_start_.strftime('%Y/%m/%d %H:%M')
    test_end_str = test_end_.strftime('%Y/%m/%d %H:%M')

    report_path = test_end_.strftime(
        f'{ts_format}{success_report_file_suffix}.yaml'
    )

    content: dict = {
        'Start': test_start_str,
        'End': test_end_str,
        'All Results': all_results_dict_,
    }

    with open(report_path, 'w') as f_:
        yaml.safe_dump(
            content,
            f_,
            allow_unicode=True,
            sort_keys=False,
        )


def get_test_start(flag_path_):
    # flag_path の内容一行目が
    # 2025/05/15 15:53:20.60 というフォーマットで
    # テスト開始時刻を表す（日本版 OS のみ）
    with open(flag_path_) as f:
        line = f.readline()
        out: datetime.datetime = datetime.datetime.strptime(
            line[:-4],  # remove . 6 0 \n
            '%Y/%m/%d %H:%M:%S',
        )
    return out


if __name__ == '__main__':

    flag_path = sys.argv[1]
    # flag_path = r'E:\pyfemtet\pyfemtet\.test-running'

    # エラーハンドリング
    if not os.path.isfile(flag_path):
        raise FileNotFoundError(f'"{flag_path}" does not exist.')

    # yaml filename フォーマットに合致する
    # ファイル名をパースして、テスト開始時刻以降に実施された
    # すべてのファイルを取得する
    test_start = get_test_start(flag_path)
    yaml_file_paths = collect_yaml_paths(test_start)

    # エラーハンドリング
    if len(yaml_file_paths) == 0:

        # メール送信
        raise FileNotFoundError(
            'No yaml files generated after '
            f'{test_start} '
            'are found. Run test first.'
        )

    # yaml 内のファイルを確認し、
    # すべてのテストが合格か判定する
    all_tests_passed, all_results_dict = check_yamls(yaml_file_paths)  # この関数は実装しなくていいです

    # 終了
    if all_tests_passed:

        # 終了レポートを作成
        create_report(
            test_start,
            all_results_dict,
            yaml_file_paths,
        )

        return_code = 0
    else:
        return_code = 1
    sys.exit(return_code)
