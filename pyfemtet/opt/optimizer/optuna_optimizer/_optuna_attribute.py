from __future__ import annotations

from typing import TypedDict, Sequence

import optuna.trial

from pyfemtet.opt.problem.problem import *
from pyfemtet.opt.history import *
from pyfemtet.opt.optimizer._base_optimizer import *


class OptunaAttribute:
    """Manage optuna user attribute.

    By `set_user_attr_to_trial`,
        key (str):
            {sub_fidelity_name}(_{subsampling_idx})
        value (dict):
            fidelity: ...
            internal_y_values: ...
            violation_values: ...
            pf_state: ...

    """
    class AttributeStructure(TypedDict):
        fidelity: Fidelity | None
        internal_y_values: Sequence[float] | None
        violation_values: Sequence[float] | None
        pf_state: TrialState | None

    def __init__(
            self,
            opt: AbstractOptimizer,
            subsampling_idx: SubSampling | None = None,
    ):
        # key
        self.sub_fidelity_name: str = opt.sub_fidelity_name
        self.subsampling_idx: SubSampling | None = subsampling_idx
        # value
        self.fidelity: Fidelity = opt.fidelity
        self.y_values: Sequence[float] | None = None
        self.v_values: Sequence[float] | None = None
        self.pf_state: TrialState | None = None

    @property
    def key(self) -> str:
        key = self.sub_fidelity_name
        if self.subsampling_idx is not None:
            key += f'_{self.subsampling_idx}'
        return key

    @property
    def value(self) -> AttributeStructure:
        out = self.AttributeStructure(
            fidelity=self.fidelity,
            internal_y_values=self.y_values,
            violation_values=self.v_values,
            pf_state=self.pf_state,
        )
        return out

    def set_user_attr_to_trial(self, trial: optuna.trial.Trial):
        trial.set_user_attr(self.key, self.value)
