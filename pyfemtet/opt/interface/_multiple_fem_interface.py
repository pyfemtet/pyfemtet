from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Any

from ._base_interface import AbstractFEMInterface
from pyfemtet.opt.problem.problem import TrialInput


if TYPE_CHECKING:
    from pyfemtet.opt.optimizer._base_optimizer import (
        AbstractOptimizer,
        GlobalOptimizationData,
        OptimizationDataPerFEM,
    )


class FEMListInterface(AbstractFEMInterface):

    def __init__(self):
        self._fems: list[AbstractFEMInterface] = []

    def __iter__(self):
        return iter(self._fems)

    def __len__(self):
        return len(self._fems)

    def __getitem__(self, index: int) -> AbstractFEMInterface:
        return self._fems[index]

    def append(self, fem: AbstractFEMInterface):
        self._fems.append(fem)

    def remove(self, fem: AbstractFEMInterface):
        self._fems.remove(fem)

    def pop(self, index: int) -> AbstractFEMInterface:
        return self._fems.pop(index)

    def update_parameter(self, x: TrialInput) -> None:
        for fem in self._fems:
            fem.update_parameter(x)

    def update(self):
        for fem in self._fems:
            fem.update()

    def trial_preprocess(self) -> None:
        for fem in self._fems:
            fem.trial_preprocess()

    def trial_postprocess(self) -> None:
        for fem in self._fems:
            fem.trial_postprocess()

    def trial_preprocess_per_fidelity(self) -> None:
        for fem in self._fems:
            fem.trial_preprocess_per_fidelity()

    def trial_postprocess_per_fidelity(self) -> None:
        for fem in self._fems:
            fem.trial_postprocess_per_fidelity()

    @property
    def object_pass_to_fun(self):
        return [fem.object_pass_to_fun for fem in self._fems]

    def reopen(self):
        for fem in self._fems:
            fem.reopen()

    def _setup_before_parallel(self, scheduler_address=None) -> None:
        for fem in self._fems:
            fem._setup_before_parallel()

    def _setup_after_parallel(self, opt: AbstractOptimizer) -> None:
        for fem in self._fems:
            fem._setup_after_parallel(opt)

    def _check_param_and_raise(self, prm_name) -> None:
        for fem in self._fems:
            fem._check_param_and_raise(prm_name)

    def load_variables(self, opt: OptimizationDataPerFEM):
        for fem in self._fems:
            fem.load_variables(opt)

    def load_objectives(self, opt: OptimizationDataPerFEM):
        for fem in self._fems:
            fem.load_objectives(opt)

    def load_constraints(self, opt: OptimizationDataPerFEM):
        for fem in self._fems:
            fem.load_constraints(opt)

    def contact_to_optimizer(
            self,
            opt: AbstractOptimizer,
            global_data: GlobalOptimizationData,
            ctx: OptimizationDataPerFEM,
    ):
        for fem in self._fems:
            fem.contact_to_optimizer(opt, global_data, ctx)

    def close(self, *args, **kwargs):
        # TODO: 引数の取り扱い
        for fem in self._fems:
            fem.close()

    def _check_using_fem(self, fun: Callable) -> bool:
        return any(fem._check_using_fem(fun) for fem in self._fems)

    def _create_postprocess_args(self) -> dict[str, Any]:
        out = {}
        for i, fem in enumerate(self._fems):
            kwargs = fem._create_postprocess_args()
            kwargs.update(
                {'__postprocess_fun__': type(fem)._postprocess_after_recording}
            )
            out.update({
                f"fem{i}kwargs": kwargs
            })
        return out

    @staticmethod
    def _postprocess_after_recording(
        dask_scheduler, trial_name: str, df: Any, **kwargs
    ) -> ...:
        for kwargs_per_fem in kwargs.values():
            postprocess = kwargs_per_fem.pop('__postprocess_fun__', None)
            if postprocess is None:
                continue
            postprocess(
                dask_scheduler, trial_name, df, **kwargs_per_fem
            )

    def _get_additional_data(self) -> dict:
        data = {}
        for i, fem in enumerate(self._fems):
            data.update(fem._get_additional_data())
        return data
