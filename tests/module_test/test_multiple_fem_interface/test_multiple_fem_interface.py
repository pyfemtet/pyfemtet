import os
from pyfemtet.opt.optimizer import AbstractOptimizer
from pyfemtet.opt.interface import FemtetInterface, NoFEM, MultipleFEMInterface

here = os.path.dirname(__file__)


class _SimpleFEM(NoFEM):
    # テスト用ダミー FEM
    def __init__(self, target):
        super().__init__()
        self.target = target
        self.internal_value: float = float('nan')

    def update(self) -> None:
        self.internal_value = self.current_prm_values[self.target].value ** 2


def test_multiple_fem_interface_basic_flow():
    # 2 つの簡易 FEM を作る
    fem1 = _SimpleFEM(target="x1")
    fem2 = _SimpleFEM(target="x2")

    # MultipleFEMInterface に登録
    fems = MultipleFEMInterface()
    fems.add(fem1)
    fems.add(fem2)

    # optimizer を作る
    opt = AbstractOptimizer()
    opt.fem = fems

    # parameter を登録（optimizer に注入される想定）
    opt.add_parameter('x1', 5, -10, 10)
    opt.add_parameter('x2', 7, -10, 10)

    # objective を登録（fem オブジェクトを受け取って内部値を返す関数）
    opt.add_objective('y1', lambda fems_: fems_._fems[0].internal_value, direction='minimize')
    opt.add_objective('y2', lambda fems_: fems_._fems[1].internal_value, direction='minimize')

    # setup
    opt._finalize()
    opt._setup_before_parallel()
    opt._setup_after_parallel()

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
    y: tuple[float, ...] = tuple(f_return[0].values())

    assert y[0] == 25.0
    assert y[1] == 49.0


if __name__ == '__main__':
    test_multiple_fem_interface_basic_flow()
