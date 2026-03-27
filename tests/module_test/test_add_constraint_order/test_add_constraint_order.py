from numpy.char import lower
from pyfemtet.opt import FEMOpt
from pyfemtet.opt.interface import NoFEM


class MyFEM(NoFEM):
    def __init__(self):
        super().__init__()
        self.process_order = []
    
    def update(self):
        self.process_order.append("update")


def constraint(fem: MyFEM):
    fem.process_order.append("constraint")
    return 1.


def test_add_constraint_order():
    fem = MyFEM()
    femopt = FEMOpt(fem=fem)
    femopt.add_parameter(name="prm", initial_value=1., lower_bound=0, upper_bound=10.)
    femopt.add_objective(name="obj", fun=lambda *a: 1.)
    femopt.add_constraint(name="cns", fun=constraint, lower_bound=0)
    femopt.optimize(
        seed=42,
        n_trials=1,
        confirm_before_exit=False,
        with_monitor=False,
    )
    assert fem.process_order == ["constraint", "update"]


if __name__ == "__main__":
    test_add_constraint_order()
