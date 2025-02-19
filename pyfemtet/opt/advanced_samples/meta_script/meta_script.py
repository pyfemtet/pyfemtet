# for meta script
import os
import pyfemtet

if __name__ == '__main__':
    print(f'pyfemtet {pyfemtet.__version__} starting.')

from pyfemtet._message.messages import encoding

from fire import Fire
import yaml

# for concrete script
from pyfemtet.opt import FEMOpt
from pyfemtet.opt.interface import *
from pyfemtet.opt.optimizer import *
from optuna.samplers import *
from pyfemtet.opt.interface._femtet_excel import FemtetWithExcelSettingsInterface


# fore debug
class ContentContext:

    def __init__(self, content_):
        self.content = content_

    def __enter__(self):
        return self.content

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


def main(
        yaml_path: str = None,

        interface_class: str = None,  # including parameter definition excel
        interface_kwargs: str = None,  # including Parametric Analysis Output
        optimizer_class: str = None,
        optimizer_kwargs: str = None,
        femopt_kwargs: str = None,
        seed: str = 'null',
        optimize_kwargs: str = None,
):
    """

    Args:
        yaml_path:
            If this argument is passed, the other arguments will be ignored
            and load by .yaml file.
            The yaml file must contain the other arguments.

        interface_class: FemtetWithExcelSettingsInterface or SurrogateModelInterface.
        interface_kwargs: See documentation of each interface class.
        optimizer_class: OptunaOptimizer or ScipyOptimizer.
        optimizer_kwargs: See documentation of each optimizer class.
        femopt_kwargs: See documentation of FEMOpt.
        seed: int or None.
        optimize_kwargs: See documentation of FEMOpt.optimize().

    """

    if yaml_path is not None:
        if os.path.isfile(yaml_path):
            context = open(yaml_path, 'r', encoding='utf-8')
        else:
            print('debug mode')
            context = ContentContext(yaml_path)
        with context as f:
            d = yaml.safe_load(f)
            interface_class = yaml.safe_dump(d['interface_class'], allow_unicode=True)
            interface_kwargs = yaml.safe_dump(d['interface_kwargs'], allow_unicode=True)
            optimizer_class = yaml.safe_dump(d['optimizer_class'], allow_unicode=True)
            optimizer_kwargs = yaml.safe_dump(d['optimizer_kwargs'], allow_unicode=True)
            femopt_kwargs = yaml.safe_dump(d['femopt_kwargs'], allow_unicode=True)
            seed = yaml.safe_dump(d['seed'], allow_unicode=True)
            optimize_kwargs = yaml.safe_dump(d['optimize_kwargs'], allow_unicode=True)

    Interface = eval(yaml.safe_load(interface_class))
    interface_kwargs_ = yaml.safe_load(interface_kwargs)

    Optimizer = eval(yaml.safe_load(optimizer_class))
    optimizer_kwargs_ = yaml.safe_load(optimizer_kwargs)

    femopt_kwargs_ = yaml.safe_load(femopt_kwargs)

    seed_ = yaml.safe_load(seed)

    optimize_kwargs_ = yaml.safe_load(optimize_kwargs)

    fem = Interface(**interface_kwargs_)
    opt = Optimizer(**optimizer_kwargs_)
    femopt = FEMOpt(fem=fem, opt=opt, **femopt_kwargs_)
    femopt.set_random_seed(seed_)
    femopt.optimize(**optimize_kwargs_)


# if __name__ == '__main__':
#     Fire(main)


# for debugging file input
if __name__ == '__main__':
    os.chdir(os.path.dirname(__file__))
    path = 'sample.yaml'
    main(yaml_path=path)


# for debugging yaml input
# if __name__ == '__main__':
#     content = r"""
# interface_class: FemtetInterface
# interface_kwargs:
#   femprj_path: C:\日本語ファイル.femprj
# optimizer_class: OptunaOptimizer
# optimizer_kwargs:
#   sampler_class: TPESampler
#   sampler_kwargs:
#     n_startup_trials: 10
# femopt_kwargs:
#   history_path: sample.csv
# optimize_kwargs:
#   n_trials: 15
#   confirm_before_exit: False
# seed: null
# """
#
#     main(yaml_path=content)


# for debugging CLI input
# if __name__ == '__main__':
#
#     interface_kwargs = dict(
#         femprj_path=r'sample.femprj',
#         parametric_output_indexes_use_as_objective={0: 0, 1: 'minimize'}
#     )
#     optimizer_kwargs = dict(
#         sampler_class='TPESampler',
#         sampler_kwargs=dict(
#             n_startup_trials=10,
#         )
#     )
#     femopt_kwargs = dict(
#         history_path='sample.csv'
#     )
#
#     seed = None
#
#     optimize_kwargs = dict(
#         n_trials=15,
#         confirm_before_exit=False,
#     )
#
#     main(
#         interface_class='FemtetWithExcelSettingsInterface',
#         interface_kwargs=yaml.safe_dump(interface_kwargs, allow_unicode=True),
#         optimizer_class='OptunaOptimizer',
#         optimizer_kwargs=yaml.safe_dump(optimizer_kwargs, allow_unicode=True),
#         femopt_kwargs=yaml.safe_dump(femopt_kwargs, allow_unicode=True),
#         seed=yaml.safe_dump(femopt_kwargs, allow_unicode=True),
#         optimize_kwargs=yaml.safe_dump(optimize_kwargs, allow_unicode=True),
#     )
