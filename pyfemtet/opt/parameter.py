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
    is_direct_to_fem: Optional[bool] = True
    properties: Optional[dict[Any]] = None


@dataclass
class Parameter(Variable):
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None
    step: Optional[float] = None


@dataclass
class Expression(Variable):
    # FIXME: dataclass をやめる。__init__ の継承が必要なので。
    # params かどうか判断するために args は実装しない
    fun: Optional[Callable] = None
    kwargs: Optional[Dict] = None


class ExpressionEvaluator:
    def __init__(self):
        self.variables = {}  # 計算された Parameter 又は Expression が入る
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



        # params = inspect.signature(exp.fun).parameters  # FIXME: この実装だと特殊文字が使えない
        params = get_accessed_keys(exp.fun)  # FIXME: この実装だと問題が多い



        self.dependencies[exp.name] = set(params)
        # self.variables[exp.name] = exp

    def resolve(self):
        ts = TopologicalSorter(self.dependencies)
        self.evaluation_order = list(ts.static_order())

    def evaluate(self):
        # order 順に見ていき、expression なら計算して variables を更新する
        for var_name in self.evaluation_order:
            if var_name in self.expressions.keys():

                # FIXME: この実装だと特殊文字が使えない
                # # 現在の variable(即ち expression)に関して param 部分の引数を作成
                # prm_kwargs = {param: self.variables[param].value for param in self.dependencies[var_name]}
                # # expression の value を更新
                # exp: Expression = self.expressions[var_name]
                # kwargs = exp.kwargs
                # kwargs.update(prm_kwargs)
                # exp.value = exp.fun(**exp.kwargs)

                # FIXME: この実装だと問題が多い
                exp: Expression = self.expressions[var_name]
                exp.value = exp.fun(self.get_variables(format='dict'), **exp.kwargs)

                self.variables[var_name] = exp

    def get_variables(self, format='dict', direct_to_fem_only=False, parameter_only=False):
        """format: dict, values, df, list_of_Variables"""

        # リストを作成
        vars = [self.variables[name] for name in self.evaluation_order if name in self.variables]

        # 必要なら FEM に直接使うもののみ取り出し
        if direct_to_fem_only:
            vars = [var for var in vars if var.is_direct_to_fem]

        # 必要なら parameter のみ取り出し
        if parameter_only:
            vars = [var for var in vars if isinstance(var, Parameter)]

        if format == 'list_of_Variables':
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
            if parameter_only:
                data.update(
                    dict(
                        lower_bound=[var.lower_bound for var in vars],
                        upper_bound=[var.upper_bound for var in vars],
                    )
                )
            return pd.DataFrame(data)

        else:
            raise NotImplementedError(f'invalid format: {format}. Valid formats are `dict`, `values`, `df` and `list_of_Variables`.')

    def get_parameter_names(self):
        return list(self.parameters.keys())


import inspect
import ast


class DictKeyVisitor(ast.NodeVisitor):
    # TODO: 引数の dict に対して実行している気配がないので、 visit する関数の中で辞書にアクセスしたら全部ハンドルしてしまうのではないか？を確認する。

    def __init__(self):
        self.keys = set()
        self.variables = {}

    def visit_Assign(self, node):
        """与えられた関数内で割り当てられた変数を取得する"""
        if isinstance(node.value, ast.Constant):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.variables[target.id] = node.value.value
        self.generic_visit(node)

    def visit_Subscript(self, node):
        """アクセスされる key をハンドルするメソッド"""
        if isinstance(node.value, ast.Name) and isinstance(node.slice, (ast.Constant, ast.Index, ast.Name)):

            # handle different ast structures for different python versions
            if isinstance(node.slice, ast.Constant):
                """In case of d['key']"""
                # self.keys.append(node.slice.value)
                key = node.slice.value

            elif isinstance(node.slice, ast.Index) and isinstance(node.slice.value, ast.Constant):
                # self.keys.append(node.slice.value.value)
                key = node.slice.value.value

            elif isinstance(node.slice, ast.Index) and isinstance(node.slice.value, ast.Str):
                # self.keys.append(node.slice.value.s)
                key = node.slice.value.s

            elif isinstance(node.slice, ast.Name):
                key = self.variables.get(node.slice.id)

            else:
                return self.generic_visit(node)

            if key is not None:
                self.keys.add(key)

        self.generic_visit(node)

    def visit_Call(self, node):
        if isinstance(node.func, ast.Attribute) and node.func.attr == 'get':
            if isinstance(node.func.value, ast.Name):
                if len(node.args) > 0 and isinstance(node.args[0], ast.Constant):
                    self.keys.add(node.args[0].value)
                elif len(node.args) > 0 and isinstance(node.args[0], ast.Str):
                    self.keys.add(node.args[0].s)
        self.generic_visit(node)


def get_accessed_keys(fun):
    # FIXME: lambda 式が入ってくると getsource したときに lambda 式の入っている行すべてが入ってくるから parse でエラーになる
    # FIXME: インデントされたスコープで関数が定義されていると、unexpected indent で parse でエラーになる。当然クラスメソッドもダメだと思う。

    source = inspect.getsource(fun)
    tree = ast.parse(source)
    visitor = DictKeyVisitor()
    visitor.visit(tree)
    return visitor.keys


if __name__ == '__main__':
    KEY_4 = 'key4'


    def function_using_dict(d: dict):
        key_5 = 'key5'
        value1 = d['key1']
        value2 = d['key2']
        value3 = d.get('key3', None)
        value4 = d[KEY_4]  # 未対応
        value5 = d[key_5]

    accessed_keys = get_accessed_keys(function_using_dict)
    print(accessed_keys)

    # # lambda は未対応
    # accessed_keys = get_accessed_keys(lambda d: d['key6'])
    # print(accessed_keys)

    # unexpected indent が出る。インデントされた関数を get_accessed_keys に入れると ast の パースに失敗する。
