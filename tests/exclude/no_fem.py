"""UI のテストなどを目的とした sandbox コードです。"""

from numpy import sin, cos, pi
from optuna.samplers import RandomSampler, BruteForceSampler
from optuna_integration import BoTorchSampler
from pyfemtet.opt import FEMOpt, OptunaOptimizer, NoFEM

from test_utils.hyper_sphere import HyperSphere


N = 10
hs = HyperSphere(N)
pi = 3.1415926


def main():

    fem = NoFEM()  # サンプルのため Femtet を使わない
    opt = OptunaOptimizer(
        sampler_class=BruteForceSampler,
        # sampler_class=BoTorchSampler,
        # sampler_kwargs=dict(
        #     n_startup_trials=10,
        # )
    )

    femopt = FEMOpt(fem, opt)

    femopt.add_parameter("r", 0.5, 0.5, 1, 0.5)
    for i in range(N - 2):
        femopt.add_parameter(f"theta{i}", 0, 0, pi, pi/4)
    femopt.add_parameter(f"theta{N-2}", 0, 0, 2 * pi, pi/4)

    for i in range(N):
        femopt.add_objective(hs.x, args=(femopt.opt, i))

    femopt.set_random_seed(42)
    femopt.optimize(
        # n_trials=30,
        n_parallel=1,
        wait_setup=True,
        confirm_before_exit=False,
    )


if __name__ == "__main__":
    main()
