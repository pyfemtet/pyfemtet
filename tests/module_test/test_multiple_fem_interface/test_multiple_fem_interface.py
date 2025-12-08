import os
from contextlib import closing
from pyfemtet.opt.optimizer import AbstractOptimizer
from pyfemtet.opt.interface import FemtetInterface, NoFEM, MultipleFEMInterface
from pyfemtet.opt.exceptions import SolveError

here = os.path.dirname(__file__)


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


def test_multiple_fem_interface_basic_flow():
    # 2 つの簡易 FEM を作る
    fem1 = _SimpleFEM(target="x1")
    fem2 = _SimpleFEM(target="x2")

    # optimizer を作る
    opt = AbstractOptimizer()

    # MultipleFEMInterface に登録
    opt._append_fem(fem1)
    opt._append_fem(fem2)

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

        def user_obj(_: MultipleFEMInterface):
            return 1.

        # optimizer を作る
        opt = AbstractOptimizer()

        # MultipleFEMInterface に登録
        opt._append_fem(fem1)
        opt._append_fem(fem2)

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
    opt._append_fem(fem1)
    opt._append_fem(fem2)

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
    

if __name__ == '__main__':
    test_multiple_fem_interface_basic_flow()
    test_multiple_fem_interface_basic_femtet()
    test_multiple_fem_interface_on_error()
