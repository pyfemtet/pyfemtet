from ray.util import inspect_serializability
from pyfemtet.opt import OptimizerOptuna, Femtet, NoFEM


def fun(Femtet):
    return 1


if __name__ == '__main__':
    fem = Femtet()
    femopt = OptimizerOptuna(fem)

    femopt.add_parameter('w', 1)
    femopt.add_objective(fun, 'a')

    result, something = inspect_serializability(femopt)  # False

    for k in femopt.__dir__():
        some = eval(f'femopt.{k}')
        ret, something = inspect_serializability(some)
        if not ret:
            print('★★★★★★★★★★★★★★★★★★',k)

    pass


