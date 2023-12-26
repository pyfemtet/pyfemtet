import os
import subprocess
from time import sleep
import psutil
import ray
from pyfemtet.tools.DispatchUtils import _get_pid
from pyfemtet.opt import OptimizerOptuna ,FemtetInterface


here, me = os.path.split(__file__)
# os.chdir(here)  # テストでは効かない？

FEMTET_EXE_PATH = r'C:\Program Files\Femtet_Ver2023_64bit_inside\Program\Femtet.exe'


# 後片付け
def _destruct(pids):
    ray.shutdown()
    for pid in pids:
        psutil.Process(pid).terminate()


def _lunch_femtet(femprj='test1/test1.femprj'):
    # 後片付けのために Femtet の pid を取得する
    pid_list_before = [p.pid for p in psutil.process_iter(attrs=["name"]) if p.info["name"] == 'Femtet.exe']

    # Femtet の起動, 普通に起動するのと同じやり方でやらないと ProjectPath が '' になる
    subprocess.Popen([FEMTET_EXE_PATH, os.path.abspath(femprj)], cwd=os.path.dirname(FEMTET_EXE_PATH))

    # pid_list の差から起動した Femtet の pid を取得する
    pid_list_after = [p.pid for p in psutil.process_iter(attrs=["name"]) if p.info["name"] == 'Femtet.exe']
    mypid = [x for x in pid_list_after if x not in pid_list_before][-1]

    # 一応 Femtet の起動を待つ
    sleep(5)

    return mypid


def test_1_1():
    """
    テストしたい状況
        ユーザーが希望の Femtet を開いており、
        Optimizer() によって接続する
    パラメータ
        Femtet
            ユーザーが正しいプロジェクト・モデルを起動
        プロジェクト
            指定しない
        モデル
            指定しない
        接続方法
            指定しない
    結果
        起動した Femtet と接続する。
    """
    # Femtet 起動
    mypid = _lunch_femtet(os.path.join(here, 'test1/test1.femprj'))

    # 接続を試みる
    femopt = OptimizerOptuna()

    # ちゃんと接続できているか確認
    try:
        assert femopt.fem.Femtet.Project == os.path.abspath(os.path.join(here, 'test1/test1.femprj'))
        assert femopt.fem.Femtet.AnalysisModelName == 'test1'
    except AssertionError as e:
        _destruct([mypid])
        raise e
    else:
        _destruct([mypid])


def test_1_2():
    """
    テストしたい状況
        ユーザーは既存の Femtet を開いているが、
        FemtetInterface(femprj, name, 'new') によって処理を始めたい
    パラメータ
        Femtet
            ユーザーが異なるプロジェクト・モデルを起動
        プロジェクト
            指定する
        モデル
            指定する
        接続方法
            指定する
    結果
        起動した Femtet と接続する。
    """
    # Femtet 起動
    mypid = _lunch_femtet(os.path.join(here, 'test1/test1_another.femprj'))

    # 接続を試みる（これは相対パスで指定できたほうが便利）
    fem = FemtetInterface(
        os.path.join(here, 'test1/test1.femprj'),
        'test1',
        connect_method='new'
    )
    femopt = OptimizerOptuna(fem)

    pid = _get_pid(femopt.fem.Femtet.hWnd)

    # ちゃんと接続できているか確認
    try:
        assert femopt.fem.Femtet.Project == os.path.abspath(os.path.join(here, 'test1/test1.femprj'))
        assert femopt.fem.Femtet.AnalysisModelName == 'test1'
    except AssertionError as e:
        _destruct([mypid, pid])
        raise e
    else:
        _destruct([mypid, pid])


if __name__ == '__main__':
    test_1_2()
