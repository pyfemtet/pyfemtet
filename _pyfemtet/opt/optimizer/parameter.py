from graphlib import TopologicalSorter
from dataclasses import dataclass
import inspect
from typing import Optional, Callable, Any, Tuple, Dict

import numpy as np
import pandas as pd


@dataclass
class Variable:
    name: str
    value: float
    pass_to_fem: Optional[bool] = True
    properties: Optional[dict[Any]] = None


@dataclass
class Parameter(Variable):
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None
    step: Optional[float] = None


@dataclass
class Expression(Variable):
    # fun に params を自動で代入するので positional args は実装しない
    fun: Optional[Callable] = None
    kwargs: Optional[Dict] = None


class ExpressionEvaluator:
    def __init__(self):
        self.variables = {}  # Parameter 又は計算された Expression が入る
        self.parameters = {}
        self.expressions = {}
        self.dependencies = {}
        self.evaluation_order = []

    def add_parameter(self, prm: Parameter):
        self.variables[prm.name] = prm
        self.parameters[prm.name] = prm
        self.dependencies[prm.name] = set()

    def add_expression(self, exp: Expression):
        self.expressions[exp.name] = exp

        # params は Python 変数として使える文字のみからなる文字列のリスト
        params = inspect.signature(exp.fun).parameters
        self.dependencies[exp.name] = set(params) - set(exp.kwargs.keys())

    def resolve(self):
        ts = TopologicalSorter(self.dependencies)
        self.evaluation_order = list(ts.static_order())

    def evaluate(self):
        # order 順に見ていき、expression なら計算して variables を更新する
        for var_name in self.evaluation_order:
            if var_name in self.expressions.keys():
                # 現在の expression に関して parameter 部分の引数 kwargs を作成
                kwargs = {param: self.variables[param].value for param in self.dependencies[var_name]}

                # fun に すべての kwargs を入れて expression の value を更新
                exp: Expression = self.expressions[var_name]
                kwargs.update(exp.kwargs)
                exp.value = exp.fun(**kwargs)

                # 計算済み variables に追加
                self.variables[var_name] = exp

    def get_variables(self, format='dict', filter_pass_to_fem=False, filter_parameter=False):
        """format: dict, values, df, raw(list of Variable object)"""

        # リストを作成
        vars = [self.variables[name] for name in self.evaluation_order if name in self.variables]

        # 必要なら FEM に直接使うもののみ取り出し
        if filter_pass_to_fem:
            vars = [var for var in vars if var.pass_to_fem]

        # 必要なら parameter のみ取り出し
        if filter_parameter:
            vars = [var for var in vars if isinstance(var, Parameter)]

        if format == 'raw':
            return vars

        elif format == 'dict':
            return {var.name: var.value for var in vars}

        elif format == 'values':
            return np.array([var.value for var in vars]).astype(float)

        elif format == 'df':
            data = dict(
                    name=[var.name for var in vars],
                    value=[var.value for var in vars],
                    properties=[var.properties for var in vars],
                )
            if filter_parameter:
                data.update(
                    dict(
                        lower_bound=[var.lower_bound for var in vars],
                        upper_bound=[var.upper_bound for var in vars],
                    )
                )
            return pd.DataFrame(data)

        else:
            raise NotImplementedError(f'invalid format: {format}. Valid formats are `dict`, `values`, `df` and `raw`(= list of Variables).')

    def get_parameter_names(self):
        return list(self.parameters.keys())
