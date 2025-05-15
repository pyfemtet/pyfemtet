"""Run pytest with -lf option.

Args:
    argv[1]: Test target. 'module', 'sample' or 'all'
    argv[2]: Use Femtet. 1 or 0.
    argv[3]: Use CAD. 1 or 0.
    argv[4]: Run manual tests. 1 or 0.
    argv[5]: With removing pytest cache before run. 1 or 0.

Examples:
    > run_test.py "module" 0 0 0 0

"""
import sys
import datetime
import subprocess
from pathlib import Path
from contextlib import suppress

import yaml

try:
    from notify import send_mail
except ModuleNotFoundError:
    from .notify import send_mail


test_result_yaml_filename_suffix = '_test_result'
test_result_yaml_timestamp_format = '%Y%m%d_%H%M%S'
project_root = Path(__file__).parent.parent.parent


if __name__ == '__main__':

    # process arguments
    # -----------------
    print('args:', sys.argv[1:])
    mode = sys.argv[1].lower()
    use_femtet = bool(int(sys.argv[2]))
    use_cad = bool(int(sys.argv[3]))
    manual_only = bool(int(sys.argv[4]))
    cache_clear = bool(int(sys.argv[5]))
    # mode = 'module'
    # use_femtet = True
    # use_cad = False
    # manual_only = False
    # cache_clear = True

    assert mode in ('module', 'sample', 'all')

    # construct marks
    # ---------------
    if manual_only:
        marks = ['manual']
    else:
        marks = ['(not manual)']
    if not use_femtet:
        marks.append('(not femtet)')
    if not use_cad:
        marks.append('(not cad)')

    # construct options
    # -----------------
    options = ''

    # -s
    if manual_only:  # requires '-s' to use input().
        options += ' -s'

    # --cache-clear
    if cache_clear:
        options += ' --cache-clear'

    # --last-failed
    options += ' --last-failed'

    # -m
    if len(marks) > 0:
        marks_str = ' and '.join(marks)
        options += f' -m "{marks_str}"'


    # --progress-path
    without_ext: str = datetime.datetime.now().strftime(
        test_result_yaml_timestamp_format + test_result_yaml_filename_suffix
    )
    yaml_path = without_ext + '.yaml'
    options += f' --progress-path="{yaml_path}"'

    # --junit-path
    options += f' --junit-xml="{without_ext}.xml"'

    # --report (pytest-reporter-html-dots)
    options += (f' --report="{without_ext}.html"'
                f' --template="html-dots/index.html"')

    # file_or_dir (final)
    if mode == 'module':
        path = str(project_root / 'tests/module_test')
        options += f' "{path}"'
    elif mode == 'sample':
        path = str(project_root / 'tests/femprj_sample_tests')
        options += f' "{path}"'
    else:
        path = str(project_root / 'tests/module_test')
        options += f' "{path}"'
        path = str(project_root / 'tests/femprj_sample_tests')
        options += f' "{path}"'

    # run pytest
    # ----------
    print(options)

    # run
    with suppress(Exception):
        subprocess.run(
            'uv'
            ' --offline '
            ' run'
            ' --no-sync'
            ' pytest'
            f'{options}',
            shell=True,
        )

    # load
    # ----

    # load
    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    # summarize
    all_names = data['items']

    # error handling
    if all_names is None:
        send_mail(
            'PyFemtet Test Error!',
            f'No tests are collected.\noptions: {options}'
        )
        sys.exit(1)

    # summarize
    passed_list = [None for _ in all_names]
    for i, result in enumerate(data.get('results', []) or []):
        # noinspection PyTypeChecker
        passed_list[i] = result.get('call', None) == 'passed'
    all_test_passed = all([passed or False for passed in passed_list])

    # send mail
    subject = f'({"✅PASSED" if all_test_passed else "🔥FAILED"}) PyFemtet Test Result'
    body = ''
    for name, passed in zip(all_names, passed_list):
        if passed is True:
            symbol = '✅'
        elif passed is False:
            symbol = '🔥'
        else:
            symbol = '❔'
        body += f'{symbol}: {name}\n'
    send_mail(subject, body)

    # return_code
    if all_test_passed:
        return_code = 0
    else:
        return_code = 1

    sys.exit(return_code)
