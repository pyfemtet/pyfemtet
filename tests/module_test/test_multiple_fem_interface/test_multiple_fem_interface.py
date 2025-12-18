import os
from contextlib import closing

import pandas as pd

from pyfemtet.opt import FEMOpt
from pyfemtet.opt.optimizer import AbstractOptimizer, OptunaOptimizer
from pyfemtet.opt.interface import FemtetInterface, NoFEM, FEMListInterface
from pyfemtet.opt.exceptions import SolveError
from pyfemtet.logger import get_module_logger

here = os.path.dirname(__file__)
logger = get_module_logger('opt.test.multiple_fem_interface', debug=True)


class _SimpleFEM(NoFEM):
    # テスト用ダミー FEM
    def __init__(self, target):
        super().__init__()
        self.target = target
        self.internal_value: float = float('nan')

    def update(self) -> None:
        try:
            self.internal_value = self.current_prm_values[self.target].value ** 2
        except KeyError:
            raise SolveError(f'Parameter "{self.target}" is not defined in current_prm_values.')


class _SimpleFEMWithPrePost(_SimpleFEM):
    name: str

    def trial_preprocess(self) -> None:
        # ダミーファイルがないことを確認
        assert not os.path.exists("dummy_preprocess.txt")
        logger.debug(f"✅ 【{self.name}】 dummy_preprocess.txt does not exist before preprocess.")

    def update(self) -> None:
        super().update()
        # ダミーファイルを作成して追記
        with open('dummy_preprocess.txt', 'a') as f:
            f.write(f'Updated with target={self.target}\n')
        logger.debug(f"📝 【{self.name}】 wrote to dummy_preprocess.txt.")

    @staticmethod
    def _postprocess_after_recording(
            dask_scheduler,
            trial_name: str,
            df: pd.DataFrame,
            **kwargs
    ) -> None:
        # ダミーファイルがまだ存在することを確認
        assert os.path.exists('dummy_preprocess.txt')
        logger.debug("✅ 【name not available】 dummy_preprocess.txt exists during postprocess.")

        # ダミーファイルに 2 行書き込まれていることを確認
        with open('dummy_preprocess.txt', 'r') as f:
            lines = f.readlines()
            logger.debug(lines)
            assert len(lines) == 2
        logger.debug("✅ 【name not available】 dummy_preprocess.txt has 2 lines during postprocess.")

    def trial_postprocess(self) -> None:
        # ダミーファイルがあったら削除
        if os.path.exists("dummy_preprocess.txt"):
            os.remove('dummy_preprocess.txt')
            logger.debug(f"✅ 【{self.name}】 dummy_preprocess.txt has been removed.")
        else:
            logger.debug(f"✅ 【{self.name}】 dummy_preprocess.txt is not exists.")


def test_multiple_fem_interface_basic_flow():
    # 2 つの簡易 FEM を作る
    fem1 = _SimpleFEM(target="x1")
    fem2 = _SimpleFEM(target="x2")

    # optimizer を作る
    opt = AbstractOptimizer()

    # MultipleFEMInterface に登録
    opt.fems.append(fem1)
    opt.fems.append(fem2)

    # parameter を登録（optimizer に注入される想定）
    opt.add_parameter('x1', 5, -10, 10)
    opt.add_parameter('x2', 7, -10, 10)

    # objective を登録（fem オブジェクトを受け取って内部値を返す関数）
    opt.add_objective('y1', lambda fems_: fems_[0].internal_value, direction='minimize')
    opt.add_objective('y2', lambda fems_: fems_[1].internal_value, direction='minimize')

    # setup
    opt._finalize()

    # variable が追加されていること
    vm = opt.variable_manager.variables
    assert 'x1' in vm
    assert 'x2' in vm

    # objective が追加されていること
    assert "y1" in opt.objectives
    assert "y2" in opt.objectives

    # solve
    x = opt.get_variables(format="raw")
    f_return = opt._get_solve_set().solve(x)
    y: tuple[float, ...] = [obj_res.value for obj_res in f_return[0].values()]  # type: ignore

    print(f'{y=}')
    assert abs(y[0] - 25.0) < 0.001
    assert abs(y[1] - 49.0) < 0.001


def test_multiple_fem_interface_basic_femtet():
    # 2 つの簡易 FEM を作る
    fem1 = FemtetInterface(femprj_path=os.path.join(here, 'fem1.femprj'))
    fem2 = FemtetInterface(femprj_path=os.path.join(here, 'fem2.femprj'))

    # 目的関数の設定
    fem1.use_parametric_output_as_objective(1)
    fem2.use_parametric_output_as_objective(1)

    with closing(fem1), closing(fem2):

        def user_obj(_: FEMListInterface):
            return 1.

        # optimizer を作る
        opt = AbstractOptimizer()

        # MultipleFEMInterface に登録
        opt.fems.append(fem1)
        opt.fems.append(fem2)

        # parameter を登録（optimizer に注入される想定）
        opt.add_parameter('x1', 5, 2, 10)
        opt.add_parameter('x2', 7, 2, 10)

        # objectives を登録
        opt.add_objective('user_defined', user_obj)

        # setup
        opt._finalize()

        # variable が追加されていること
        vm = opt.variable_manager.variables
        assert 'x1' in vm
        assert 'x2' in vm

        # objective が追加されていること
        # opt.objectives にはユーザー定義の目的関数のみ含まれる
        # FEM 由来の目的関数は各 FEMContext.objectives に含まれる
        print(f'{tuple(opt.objectives)=}')
        assert 'user_defined' in opt.objectives

        # FEMContext の目的関数を確認
        all_objectives = list(opt.objectives.keys())
        print(f'{all_objectives=}')
        # assert 'user_defined' in all_objectives
        # assert '応力[Pa] / 静水圧 / 最大値 / 全てのボディ属性' in all_objectives
        # assert '0: 定常解析 / 温度[deg] / 最小値 / 全てのボディ属性' in all_objectives
        assert all_objectives == [
            '応力[Pa] / 静水圧 / 最大値 / 全てのボディ属性',
            '0: 定常解析 / 温度[deg] / 最小値 / 全てのボディ属性',
            'user_defined',
        ]

        # solve
        x = opt.get_variables(format="raw")
        f_return = opt._get_solve_set().solve(x)
        y_dict: dict[str, float] = {name: obj_res.value for name, obj_res in f_return[0].items()}

        print(f'{y_dict=}')
        # 期待される値:
        # y_dict={
        #   '応力[Pa] / 静水圧 / 最大値 / 全てのボディ属性': 1.0000026284781436,
        #   '0: 定常解析 / 温度[deg] / 最小値 / 全てのボディ属性': 30.123154397344265,
        #   'user_defined': 1.0
        # }
        assert abs(y_dict['user_defined'] - 1.0) < 0.001
        assert abs(y_dict['応力[Pa] / 静水圧 / 最大値 / 全てのボディ属性'] - 1.0000026284781436) < 0.001
        assert abs(y_dict['0: 定常解析 / 温度[deg] / 最小値 / 全てのボディ属性'] - 30.123154397344265) < 0.001


def test_multiple_fem_interface_on_error():
    # 2 つの簡易 FEM を作る
    fem1 = _SimpleFEM(target="x1")
    fem2 = _SimpleFEM(target="undefined parameter")  # 片方はエラーになるようにしておく

    # optimizer を作る
    opt = AbstractOptimizer()

    # MultipleFEMInterface に登録
    opt.fems.append(fem1)
    opt.fems.append(fem2)

    # parameter を登録（optimizer に注入される想定）
    opt.add_parameter('x1', 5, -10, 10)
    opt.add_parameter('x2', 7, -10, 10)

    # objective を登録（fem オブジェクトを受け取って内部値を返す関数）
    opt.add_objective('y1', lambda fems_: fems_[0].internal_value, direction='minimize')
    opt.add_objective('y2', lambda fems_: fems_[1].internal_value, direction='minimize')

    # setup
    opt._finalize()

    # variable が追加されていること
    vm = opt.variable_manager.variables
    assert 'x1' in vm
    assert 'x2' in vm

    # objective が追加されていること
    assert "y1" in opt.objectives
    assert "y2" in opt.objectives

    # # solve
    # x = opt.get_variables(format="raw")
    # f_return = opt._get_solve_set().solve(x)
    # y: tuple[float, ...] = [obj_res.value for obj_res in f_return[0].values()]  # type: ignore

    # solve_or_raise
    x = opt.get_variables(format="raw")
    try:
        f_return = opt._get_solve_set()._solve_or_raise(opt, x)
    except SolveError as e:
        print(f'Caught expected error: {e}')
        return
    else:
        print(f_return)
        assert False, "Expected an error but none was raised."


def test_multiple_fem_prepost():

    femopt = FEMOpt(fem=NoFEM(), opt=AbstractOptimizer())

    # 2 つの簡易 FEM を作る
    fem1 = _SimpleFEMWithPrePost(target="x1")
    fem2 = _SimpleFEMWithPrePost(target="x2")
    fem1.name = "FEM1"
    fem2.name = "FEM2"

    # optimizer を作る
    opt = OptunaOptimizer()

    # MultipleFEMInterface に登録
    opt.fems.append(fem1)
    opt.fems.append(fem2)

    # femopt に追加
    femopt.opt = opt

    # parameter を登録（optimizer に注入される想定）
    opt.add_parameter('x1', 5, -10, 10)
    opt.add_parameter('x2', 7, -10, 10)

    # objective を登録（fem オブジェクトを受け取って内部値を返す関数）
    opt.add_objective('y1', lambda fems_: fems_[0].internal_value, direction='minimize')
    opt.add_objective('y2', lambda fems_: fems_[1].internal_value, direction='minimize')

    # 実行。assertionError が出なければ成功
    femopt.optimize(
        n_trials=3,
        confirm_before_exit=False,
    )


class _FEMWithParamCheck(NoFEM):
    """テスト用 FEM: _check_param_and_raise の呼び出しを記録する"""

    def __init__(self, registered_params: list[str]):
        super().__init__()
        self.registered_params = registered_params
        self.checked_params: list[str] = []

    def _check_param_and_raise(self, prm_name) -> None:
        self.checked_params.append(prm_name)
        if prm_name not in self.registered_params:
            raise RuntimeError(f'Parameter "{prm_name}" is not registered in this FEM.')


def test_check_param_and_raise_with_ctx():
    """ctx.add_parameter で追加した変数は対応する FEM でのみチェックされる"""

    # 2 つの FEM を作る（それぞれ異なる変数を登録）
    fem1 = _FEMWithParamCheck(registered_params=['x1'])
    fem2 = _FEMWithParamCheck(registered_params=['x2'])

    # optimizer を作る
    opt = AbstractOptimizer()

    # FEM を登録して FEMContext を取得
    ctx1 = opt.fems.append(fem1)
    ctx2 = opt.fems.append(fem2)

    # 各 FEMContext に対応する変数を登録
    ctx1.add_parameter('x1', 5, -10, 10)
    ctx2.add_parameter('x2', 7, -10, 10)

    # objective を登録
    opt.add_objective('y', lambda fems_: fems_[0].internal_value + fems_[1].internal_value)

    # setup（_check_param_and_raise が呼ばれる）
    opt._finalize()

    # fem1 は x1 のみチェックされる
    assert fem1.checked_params == ['x1'], f"Expected ['x1'], got {fem1.checked_params}"
    # fem2 は x2 のみチェックされる
    assert fem2.checked_params == ['x2'], f"Expected ['x2'], got {fem2.checked_params}"

    print("✅ test_check_param_and_raise_with_ctx passed")


def test_check_param_and_raise_without_ctx():
    """opt.add_parameter で追加した変数はチェックされない"""

    # FEM を作る（変数は登録されていない）
    fem1 = _FEMWithParamCheck(registered_params=[])
    fem2 = _FEMWithParamCheck(registered_params=[])

    # optimizer を作る
    opt = AbstractOptimizer()

    # FEM を登録
    opt.fems.append(fem1)
    opt.fems.append(fem2)

    # opt 経由で変数を登録（FEMContext 経由ではない）
    opt.add_parameter('x1', 5, -10, 10)
    opt.add_parameter('x2', 7, -10, 10)

    # objective を登録
    opt.add_objective('y', lambda fems_: 1.0)

    # setup（_check_param_and_raise が呼ばれるが、どの ctx にも属さないのでチェックされない）
    opt._finalize()

    # どの FEM もチェックされない
    assert fem1.checked_params == [], f"Expected [], got {fem1.checked_params}"
    assert fem2.checked_params == [], f"Expected [], got {fem2.checked_params}"

    print("✅ test_check_param_and_raise_without_ctx passed")


def test_check_param_and_raise_error():
    """ctx に登録した変数が FEM に存在しない場合はエラーになる"""

    # FEM を作る（x1 は登録されていない）
    fem1 = _FEMWithParamCheck(registered_params=[])  # x1 が存在しない

    # optimizer を作る
    opt = AbstractOptimizer()

    # FEM を登録して FEMContext を取得
    ctx1 = opt.fems.append(fem1)

    # ctx1 に x1 を登録（しかし fem1 には x1 が存在しない）
    ctx1.add_parameter('x1', 5, -10, 10)

    # objective を登録
    opt.add_objective('y', lambda fems_: 1.0)

    # setup でエラーが発生するはず
    try:
        opt._finalize()
    except RuntimeError as e:
        print(f'Caught expected error: {e}')
        assert 'x1' in str(e)
        print("✅ test_check_param_and_raise_error passed")
        return

    assert False, "Expected RuntimeError but none was raised"


if __name__ == '__main__':
    # test_multiple_fem_interface_basic_flow()
    # test_multiple_fem_interface_basic_femtet()
    # test_multiple_fem_interface_on_error()
    # test_multiple_fem_prepost()
    test_check_param_and_raise_with_ctx()
    test_check_param_and_raise_without_ctx()
    test_check_param_and_raise_error()
