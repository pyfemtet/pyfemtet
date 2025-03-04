from typing import Callable


__all__ = [
    'Parameter',
    'Variable',
    'NumericVariable',
    'NumericParameter',
    'NumericExpression',
    'CategoricalVariable',
    'CategoricalParameter',
    'CategoricalExpression',
    'VariableManager',
]

import numpy as np


class Variable:
    name: str
    value: ...
    pass_to_fem: bool
    properties: dict[str, ...]

    def __init__(self):
        self.value = None
        self.properties = {}

    def __repr__(self):
        return str(self.value)


class Parameter(Variable): ...


class NumericVariable(Variable):
    value: float


class NumericParameter(NumericVariable, Parameter):
    value: float
    lower_bound: float | None
    upper_bound: float | None
    step: float | None


class CategoricalVariable(Variable):
    value: str


class CategoricalParameter(CategoricalVariable, Parameter):
    choices: list[str]


class NumericExpression(NumericVariable):
    expression: str
    fun: Callable[[...], float]


class CategoricalExpression(CategoricalVariable):
    expression: str
    fun: Callable[[...], str]


class VariableManager:

    variables: dict[str, Variable]

    def __init__(self):
        super().__init__()
        self.variables = dict()

    def resolve(self):
        ...

    def evaluate(self):
        ...

    def get_variables(
            self,
            *,
            filter: tuple | None = None,  # 'pass_to_fem' and 'parameter' (or filter)
            format: str = None,  # None, 'dict' and 'values'
    ) -> (
        dict[str, Variable]
        | dict[str, float]
        | np.ndarray
    ):

        raw = {}

        for name, var in self.variables.items():

            if filter is not None:
                if 'pass_to_fem' in filter:
                    if var.pass_to_fem:
                        raw.update({name: var})

                if 'parameter' in filter:
                    if isinstance(var, Parameter):
                        raw.update({name: var})

            else:
                raw.update({name: var})

        if format is None:
            return raw

        elif format == 'dict':
            return {name: var.value for name, var in raw.items()}

        elif format == 'values':
            return np.array([var.value for var in raw.values()], dtype=np.float64)
