from __future__ import annotations

from v1.problem import *
from v1.optimizer.optimizer import *


class OptunaAttribute:
    """Manage optuna user attribute

    user attributes are:
        sub_fidelity_name:
            fidelity: ...
            OBJECTIVE_ATTR_KEY: ...
            PYFEMTET_STATE_ATTR_KEY: ...
            CONSTRAINT_ATTR_KEY: ...

    """

    OBJECTIVE_KEY = 'internal_objective'
    CONSTRAINT_KEY = 'constraint'
    PYFEMTET_TRIAL_STATE_KEY = 'pyfemtet_trial_state'

    sub_fidelity_name: str  # key
    fidelity: Fidelity
    v_values: tuple  # violation
    y_values: tuple  # internal objective
    pf_state: tuple  # PyFemtet state

    def __init__(self, opt: AbstractOptimizer):
        self.sub_fidelity_name = opt.sub_fidelity_name
        self.fidelity = None
        self.v_values = None
        self.y_values = None
        self.pf_state = None

    # noinspection PyPropertyDefinition
    @classmethod
    @property
    def main_fidelity_key(cls):
        return MAIN_FIDELITY_NAME

    @property
    def key(self):
        return self.sub_fidelity_name

    @property
    def value(self):
        d = {
            'fidelity': self.fidelity,
            self.OBJECTIVE_KEY: self.y_values,
            self.CONSTRAINT_KEY: self.v_values,
            self.PYFEMTET_TRIAL_STATE_KEY: self.pf_state,
        }
        return d

    @staticmethod
    def get_fidelity(optuna_attribute: OptunaAttribute):
        return optuna_attribute.value['fidelity']

    @staticmethod
    def get_violation(optuna_attribute: OptunaAttribute):
        return optuna_attribute.value[OptunaAttribute.CONSTRAINT_KEY]

    @staticmethod
    def get_violation_from_trial_attr(trial_attr: dict):  # value is OptunaAttribute.value
        return trial_attr[OptunaAttribute.CONSTRAINT_KEY]

    @staticmethod
    def get_pf_state_from_trial_attr(trial_attr: dict):  # value is OptunaAttribute.value
        return trial_attr[OptunaAttribute.PYFEMTET_TRIAL_STATE_KEY]
