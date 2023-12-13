from time import sleep
from win32com.client import Dispatch
from minimum import OptimizerOptuna


def parabola(x):
    sleep(1)
    return (x ** 2).sum()


if __name__ == '__main__':
    app = Dispatch('excel.application')

    opt = OptimizerOptuna()
    opt.app = app
    opt.add_parameter('x', .5, -1, 1)
    opt.add_parameter('y', .5, -1, 1)
    opt.add_parameter('z', .5, -1, 1)
    opt.add_objective('parabola', parabola)

    # study = opt.main(n_trials=60, n_parallel=1)  # monitor が動くか
    study = opt.main(n_trials=60, n_parallel=3)  # 並列が動くか

    print(opt.history.data)
