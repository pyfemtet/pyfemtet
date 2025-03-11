from __future__ import annotations

import warnings
import datetime
from contextlib import closing

import numpy as np

import optuna

from v1.history import *
from v1.problem import *
from v1.interface import *
from v1.exceptions import *
from v1.variable_manager import *
from v1.logger import get_optuna_logger, remove_all_output

from v1.optimizer.optimizer import *
from v1.optimizer.optuna_optimizer.optuna_attribute import OptunaAttribute

remove_all_output(get_optuna_logger())

warnings.filterwarnings('ignore', 'set_metric_names', optuna.exceptions.ExperimentalWarning)


class OptunaOptimizer(AbstractOptimizer):

    current_trial: optuna.trial.Trial | None

    def _create_infeasible_constraints(self, opt_: AbstractOptimizer = None) -> tuple:
        opt_ = opt_ if opt_ is not None else self
        count = 0
        for name, cns in opt_.constraints.items():
            if cns.lower_bound is not None:
                count += 1
            if cns.upper_bound is not None:
                count += 1
        return tuple(np.ones(count, dtype=np.float64))

    def _constraint(self, trial: optuna.trial.FrozenTrial):
        key = OptunaAttribute(self).key
        value = trial.user_attrs[key]
        return OptunaAttribute.get_violation_from_trial_attr(value)

    def _objective(self, trial: optuna.trial.Trial):

        self.current_trial = trial

        with self._logging():

            vm = self.variable_manager

            # check interruption
            self._check_and_raise_interruption()

            # parameter suggestion
            params = vm.get_variables(filter='parameter')
            for name, prm in params.items():

                if prm.properties.get('fix', False):  # default is False
                    continue

                if isinstance(prm, NumericParameter):
                    prm.value = trial.suggest_float(
                        name,
                        prm.lower_bound,
                        prm.upper_bound,
                        step=prm.step,
                        log=prm.properties.get('log', __default := False),
                    )
                elif isinstance(prm, CategoricalParameter):
                    prm.value = trial.suggest_categorical(
                        name, prm.choices
                    )
                else:
                    raise NotImplementedError

            # evaluate expressions
            vm.evaluate()

            # check interruption
            self._check_and_raise_interruption()

            # construct TrialInput
            x = vm.get_variables(filter='parameter')
            x_pass_to_fem: TrialInput = vm.get_variables(filter='pass_to_fem')


            def solve(
                    opt_: AbstractOptimizer = self
            ) -> tuple[float] | None:

                # check interruption
                self._check_and_raise_interruption()

                # declare output
                y_internal_: tuple[float] | None = None

                # prepare attribute
                optuna_attr = OptunaAttribute(opt_)

                # if opt_ is not self, update variable manager
                opt_.variable_manager = vm

                # start solve
                datetime_start = datetime.datetime.now()
                try:
                    _, dict_or_None_y_internal, c, record = opt_.f(x, x_pass_to_fem, self.history, datetime_start)

                    # convert dict or None to tuple or None
                    y_internal_ = dict_or_None_y_internal if dict_or_None_y_internal is None else tuple(dict_or_None_y_internal.values())

                # if (hidden) constraint violation, set trial attribute
                except (HardConstraintViolation, HiddenConstraintViolation) as e:
                    optuna_attr.pf_state = TrialState.get_corresponding_state_from_exception(e)
                    optuna_attr.v_values = self._create_infeasible_constraints(opt_)

                # if skipped
                except SkipSolve:
                    optuna_attr.pf_state = TrialState.skipped

                # if succeeded
                else:

                    # convert constraint to **sorted** violation
                    assert len(c) == len(opt_.constraints)
                    v = {}
                    for cns_name, cns in opt_.constraints.items():
                        # This is {lower or upper: violation_value} dict
                        violation: dict[str, float] = c[cns_name].calc_violation()
                        for l_or_u, violation_value in violation.items():
                            key_ = cns_name + '_' + l_or_u
                            v.update({key_: violation_value})

                    # register results
                    optuna_attr.v_values = tuple(v.values())
                    optuna_attr.y_values = y_internal_
                    optuna_attr.pf_state = record.state

                # update trial attribute
                trial.set_user_attr(optuna_attr.key, optuna_attr.value)

                # check interruption
                self._check_and_raise_interruption()

                return y_internal_


            # process main fidelity model
            y_internal: tuple[float] | None = solve()

            # process sub_fidelity_models
            for sub_fidelity_name, sub_opt in self.sub_fidelity_models.items():
                solve(sub_opt)

            # check interruption
            self._check_and_raise_interruption()

            # clear trial
            self.current_trial = None

            # To avoid trial FAILED with hard constraint
            # violation, check pf_state and raise TrialPruned.
            key = OptunaAttribute(self).key
            value = trial.user_attrs[key]
            state = OptunaAttribute.get_pf_state_from_trial_attr(value)
            if state in [
                TrialState.hard_constraint_violation,
                TrialState.model_error,
                TrialState.mesh_error,
                TrialState.solve_error,
                TrialState.post_error,
            ]:
                raise optuna.TrialPruned

            # if main solve skipped, y_internal is empty.
            # this should be processed as FAIL.
            elif state == TrialState.skipped:
                return None

            return y_internal

    def run(self):

        # quit FEM even if abnormal termination
        with closing(self.fem):

            # sub fidelity
            if self.sub_fidelity_models is None:
                self.sub_fidelity_models = SubFidelityModels()
                for sub_fidelity_model in self.sub_fidelity_models.values():
                    assert sub_fidelity_model.objectives.keys() == self.objectives.keys()
                    assert sub_fidelity_model.constraints.keys() == self.constraints.keys()

            # finalize
            self._finalize_history()

            # optuna
            from v1.optimizer.optuna_optimizer.pof_botorch.pof_botorch_sampler import PoFBoTorchSampler
            from v1.optimizer.optuna_optimizer.pof_botorch.pof_botorch_sampler import PoFConfig, PartialOptimizeACQFConfig
            sampler = PoFBoTorchSampler(
                n_startup_trials=5,
                seed=42,
                constraints_func=self._constraint,
                pof_config=PoFConfig(
                    # consider_pof=False,
                    # feasibility_threshold='mean',
                ),
                partial_optimize_acqf_kwargs=PartialOptimizeACQFConfig(
                    # gen_candidates='scipy',
                    timeout_sec=5.,
                    # method='SLSQP'  # 'COBYLA, COBYQA, SLSQP or trust-constr
                    tol=0.1,
                    # scipy_minimize_kwargs=dict(),
                ),
            )
            # from optuna_integration import BoTorchSampler
            # sampler = BoTorchSampler(n_startup_trials=5)

            if isinstance(sampler, PoFBoTorchSampler):
                sampler.pyfemtet_optimizer = self  # FIXME: multi-fidelity に対応できない?

            study = optuna.create_study(
                directions=['minimize'] * len(self.objectives),
                sampler=sampler,
            )

            study.set_metric_names(list(self.objectives.keys()))

            study.optimize(
                self._objective,
                n_trials=100,
                catch=InterruptOptimization
            )

    def _check_and_raise_interruption(self):
        try:
            AbstractOptimizer._check_and_raise_interruption(self)
        except InterruptOptimization as e:
            if self.current_trial is not None:
                self.current_trial.study.stop()
            raise e


if __name__ == '__main__':

    from v1.exceptions import PostProcessError

    def _parabola(_fem: AbstractFEMInterface, _opt: AbstractOptimizer) -> float:
        x = _opt.get_variables('values')
        # if _cns(_fem, _opt) < 0:
        #     raise PostProcessError
        return (x ** 2).sum()

    def _parabola2(_fem: AbstractFEMInterface, _opt: AbstractOptimizer) -> float:
        x = _opt.get_variables('values')
        return ((x-0.1) ** 2).sum()

    def _cns(_fem: AbstractFEMInterface, _opt: AbstractOptimizer) -> float:
        x = _opt.get_variables('values')
        return x[0]

    _fem = NoFEM()
    _opt = OptunaOptimizer()
    _opt.fem = _fem
    _opt.add_parameter('x1', 1, -1, 1, step=0.1)
    _opt.add_parameter('x2', 1, -1, 1, step=0.1)
    _opt.add_constraint('cns', _cns, lower_bound=0.8, args=(_fem, _opt))
    _opt.add_objective('obj1', _parabola, args=(_fem, _opt))
    # _opt.add_objective('obj2', _parabola2, args=(_fem, _opt))


    # # ===== sub-fidelity =====
    # __fem = NoFEM()
    # __opt = SubFidelityModel()
    # __opt.fem = __fem
    # __opt.add_objective('obj1', _parabola, args=(__fem, __opt))
    # __opt.add_objective('obj2', _parabola2, args=(__fem, __opt))
    #
    # _opt.add_sub_fidelity_model(name='low-fidelity', sub_fidelity_model=__opt, fidelity=0.5)
    #
    # def _solve_condition(_history: History):
    #
    #     sub_fidelity_df = _history.get_df(
    #         {'sub_fidelity_name': 'low-fidelity'}
    #     )
    #     idx = sub_fidelity_df['state'] == TrialState.succeeded
    #     pdf = sub_fidelity_df[idx]
    #
    #     return len(pdf) % 5 == 0
    #
    # _opt.set_solve_condition(_solve_condition)


    # _opt.history.path = 'restart-test.csv'
    _opt.run()

    # import plotly.express as px
    # _df = _opt.history.get_df()
    # px.scatter_3d(_df, x='x1', y='x2', z='obj', color='fidelity', opacity=0.5).show()

    _opt.history.save()
