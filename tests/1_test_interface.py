import os
from pyfemtet.opt.interface import Femtet, NoFEM


here, me = os.path.split(__file__)


if __name__ == '__main__':
    fem1 = NoFEM()

    path = os.path.join(here, f'{me.replace(".py", "")}/some_project.femprj')

    # fem2 = Femtet(femprj_path=None, model_name=None, connect_method='auto')
    # fem4 = Femtet(femprj_path=path, model_name=None, connect_method='catch')
    # fem3 = Femtet(femprj_path=None, model_name=None, connect_method='new')  # 新しい Fmetet で開かなければ OK
    fem5 = Femtet(femprj_path=path, model_name=None, connect_method='new')  # 新しい Fmetet で開けば OK
    # fem6 = Femtet(femprj_path=path, model_name='解析モデル2', connect_method='new')  # 新しい Fmetet で開けば OK
    # fem7 = Femtet(femprj_path=path, model_name=None, connect_method='auto')  # 何等かの接続可能 Femtet を開いていればそれを乗っ取って開き、開いていなければ新しい Femtet を立てる。
