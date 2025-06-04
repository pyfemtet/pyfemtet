import sys
import signal
from time import sleep
from pathlib import Path
import subprocess
import pytest
import pyfemtet


project_root = Path(pyfemtet.__file__).parent.parent
docs_root = project_root / 'docs'
build_script = docs_root / 'pre-build.ps1'


@pytest.mark.manual
def test_check_visualization_tutorial():
    popen = subprocess.Popen(
        f'{sys.executable} '
        '-m '
        'pyfemtet.opt.visualization.history_viewer',
        shell=True,
    )

    print('起動中。10秒待ちます。')
    sleep(10)

    print('')


    input('チュートリアルモードで動作確認。\nEnter でモニターを終了。')
    popen.send_signal(signal.CTRL_C_EVENT)

    assert input('結果が NG なら NG と入力\n>>> ') != 'NG'


if __name__ == '__main__':
    test_check_visualization_tutorial()
