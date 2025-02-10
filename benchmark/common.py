import numpy as np
from numpy import pi, sin, cos, sqrt
from numpy.linalg import norm
import pandas as pd

from optuna.samplers import BaseSampler
from pyfemtet.core import SolveError
from pyfemtet.opt.optimizer import PoFBoTorchSampler, PoFConfig

from pyfemtet.opt import NoFEM, FEMOpt, OptunaOptimizer

import os

os.chdir(os.path.dirname(__file__))


__all__ = [
    'np',
    'pi',
    'sin',
    'cos',
    'norm',
    'sqrt',
    'pd',

    'Problem',
    'hidden_constraint',
    'Algorithm',
    'create_history_path',
    'parse_history_path',
    'main',

    'PoFBoTorch',
    'TPE',
    'NSGA2',
    'NSGA3',
    'Rand',
    'GP',
]


# ===== common =====
class Problem:

    femopt: FEMOpt

    def setup(self, femopt_: FEMOpt):
        self.femopt = femopt_
        self.setup_parameters()
        self.setup_objectives()
        self.setup_constraints()
        self.setup_hv_reference()

    def setup_parameters(self):
        raise NotImplementedError

    def setup_objectives(self):
        raise NotImplementedError

    def setup_constraints(self):
        raise NotImplementedError

    @property
    def hv_reference(self) -> str or np.ndarray:
        return 'dynamic-pareto'

    def setup_hv_reference(self):
        self.femopt._hv_reference = self.hv_reference


def hidden_constraint(constraint_fun, lower_bound=None, upper_bound=None):

    if lower_bound is None:
        lower_bound = -np.inf
    if upper_bound is None:
        upper_bound = np.inf

    def f_(f):

        def f__(problem: Problem):

            f___ = getattr(problem, constraint_fun)
            if lower_bound <= f___() <= upper_bound:
                return f(problem)
            else:
                raise SolveError

        return f__

    return f_


class Algorithm:

    def __init__(self):
        self.config: dict = dict()

    @property
    def sampler_class(self) -> type[BaseSampler]:
        raise NotImplementedError

    @property
    def sampler_kwargs(self) -> dict:
        raise NotImplementedError

    @property
    def opt(self) -> OptunaOptimizer:
        return OptunaOptimizer(sampler_class=self.sampler_class, sampler_kwargs=self.sampler_kwargs)


def create_history_path(a: Algorithm, p: Problem, ext='.csv', forced_counter=None):

    def path_wo_ext():
        problem_name = type(p).__name__
        algorithm_name = type(a).__name__
        algorithm_config = '_'.join([f'{''.join(term[0].upper() for term in k.split('_'))}={str(v)}' for k, v in a.config.items()])

        if algorithm_config == '':
            algorithm_config = ' '

        if not os.path.exists('results'):
            os.mkdir('results')

        if not os.path.exists(os.path.join('results', problem_name)):
            os.mkdir(os.path.join('results', problem_name))

        path_without_ext = os.path.join(
            'results',
            problem_name,
            f'{problem_name}__{algorithm_name}__{algorithm_config}'
        )

        return path_without_ext

    counter = 1
    while True:
        path = f'{path_wo_ext()}__{counter}{ext}'
        counter += 1
        if not os.path.exists(path):
            break

    return path, counter


def parse_history_path(result_path):
    base_name_wo_ext = os.path.splitext(os.path.basename(result_path))[0]
    problem_name, algorithm_name, str_algorithm_config, seed = base_name_wo_ext.split('__')
    if str_algorithm_config != ' ':
        algorithm_config = {key_value.split('=')[0]: key_value.split('=')[1] for key_value in str_algorithm_config.split('_')}
    else:
        algorithm_config = {}
    return problem_name, algorithm_name, algorithm_config, seed


def main(algorithm: Algorithm, problem: Problem, n_trials=1, timeout=None):

    fem = NoFEM()
    opt = algorithm.opt
    history_path, seed = create_history_path(algorithm, problem)
    femopt = FEMOpt(fem=fem, opt=opt, history_path=history_path)
    femopt.set_random_seed(seed)
    problem.setup(femopt)
    df = femopt.optimize(
        n_trials=n_trials,
        timeout=n_trials * 60 if timeout is None else timeout,
        confirm_before_exit=False,
    )

    history = femopt.history
    if len(history.obj_names) == 1:
        target_values = df[history.obj_names[0]].dropna().values

    else:
        indices = (~df[history.obj_names].isna()).sum(axis=1).astype(bool)
        pdf = df[indices]
        target_values = pdf['hypervolume'].values

    target_values: np.ndarray
    txt_path, _ = create_history_path(algorithm, problem, '.txt')
    np.savetxt(txt_path, target_values)

    return target_values, txt_path


# ===== Algorithms =====
class PoFBoTorch(Algorithm):

    pof_config: PoFConfig = PoFConfig()

    def __init__(self, n_startup_trials=None, gamma=None):
        super().__init__()

        if n_startup_trials is not None:
            self.config.update(dict(n_startup_trials=n_startup_trials))

        if gamma is not None:
            self.config.update(dict(gamma=gamma))

    @property
    def sampler_class(self) -> type[BaseSampler]:
        return PoFBoTorchSampler

    @property
    def sampler_kwargs(self) -> dict:

        if 'gamma' in self.config:
            self.pof_config.gamma = self.config['gamma']

        d = dict(
            n_startup_trials=self.config['n_startup_trials'],
            pof_config=self.pof_config,
        )
        return d


class TPE(Algorithm):

    def __init__(self, n_startup_trials=None):
        super().__init__()

        if n_startup_trials is not None:
            self.config.update(dict(n_startup_trials=n_startup_trials))

    @property
    def sampler_class(self) -> type[BaseSampler]:
        from optuna.samplers import TPESampler
        return TPESampler

    @property
    def sampler_kwargs(self) -> dict:
        d = dict(
            n_startup_trials=self.config['n_startup_trials'],
        )
        return d


class Rand(Algorithm):


    @property
    def sampler_class(self) -> type[BaseSampler]:
        from optuna.samplers import RandomSampler
        return RandomSampler

    @property
    def sampler_kwargs(self) -> dict:
        d = dict()
        return d


class NSGA2(Algorithm):

    def __init__(self, population_size=None):
        super().__init__()
        if population_size is not None:
            self.config['population_size'] = population_size

    @property
    def sampler_class(self) -> type[BaseSampler]:
        from optuna.samplers import NSGAIISampler
        return NSGAIISampler

    @property
    def sampler_kwargs(self) -> dict:
        d = dict(
            population_size=self.config['population_size'],
        )
        return d


class NSGA3(Algorithm):

    def __init__(self, population_size=None):
        super().__init__()
        if population_size is not None:
            self.config['population_size'] = population_size

    @property
    def sampler_class(self) -> type[BaseSampler]:
        from optuna.samplers import NSGAIIISampler
        return NSGAIIISampler

    @property
    def sampler_kwargs(self) -> dict:
        d = dict(
            population_size=self.config['population_size'],
        )
        return d


class GP(Algorithm):

    def __init__(self):
        super().__init__()

    @property
    def sampler_class(self) -> type[BaseSampler]:
        from optuna.samplers import GPSampler
        return GPSampler

    @property
    def sampler_kwargs(self) -> dict:
        d = dict(
            deterministic_objective=True,
        )
        return d


class CmaEs(Algorithm):

    def __init__(self):
        super().__init__()

    @property
    def sampler_class(self) -> type[BaseSampler]:
        from optuna.samplers import CmaEsSampler
        return CmaEsSampler

    @property
    def sampler_kwargs(self) -> dict:
        d = dict(
        )
        return d
