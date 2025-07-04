from pyfemtet.opt.optimizer import AbstractOptimizer
from pyfemtet.opt.interface import NoFEM
from pyfemtet.opt.problem.problem import Function, FunctionResult


def other_output_1(_, arg):
    return 1 + arg


def test_add_other_output():
    opt = AbstractOptimizer()
    opt.fem = NoFEM()

    # add_other_output
    opt.add_other_output('oo1', other_output_1, args=(1,))

    # other_output が追加されている
    other_output: Function = tuple(opt.other_outputs.values())[0]
    assert isinstance(other_output, Function)

    # other_output が計算できる
    result = other_output.eval(opt.fem)
    print(f'Evaluated value: {result}')
    assert result == 2


if __name__ == '__main__':
    test_add_other_output()
