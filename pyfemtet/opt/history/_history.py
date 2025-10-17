from __future__ import annotations

from typing import TYPE_CHECKING, TypeAlias, Literal

import os
import csv
import ast
import math
import json
import datetime
import dataclasses
from time import sleep, time

import numpy as np
import pandas as pd

import pyfemtet

from pyfemtet._i18n import *
from pyfemtet._util.helper import generate_random_id
from pyfemtet._util.df_util import *
from pyfemtet._util.dask_util import *
from pyfemtet._util.str_enum import StrEnum
from pyfemtet.opt.exceptions import *
from pyfemtet.opt.problem.problem import *
from pyfemtet.opt.problem.variable_manager import *
from pyfemtet.logger import get_module_logger

from pyfemtet.opt.history._optimality import *
from pyfemtet.opt.history._hypervolume import *

if TYPE_CHECKING:
    from pyfemtet.opt.interface import AbstractFEMInterface


__all__ = [
    'TrialState',
    'History',
    'ColumnOrderMode',
    'Record',
    'create_err_msg_from_exception',
    'CorrespondingColumnNameRuler',
    'MAIN_FILTER',
]

MAIN_FILTER: dict = {
    'sub_fidelity_name': MAIN_FIDELITY_NAME,
    'sub_sampling': float('nan')
}


logger = get_module_logger('opt.history', False)
logger_dask = get_module_logger('opt.dask', False)


def _assert_locked_with_timeout(lock, assertion_message=None, timeout=10.):
    start = time()
    while not lock.locked():
        sleep(0.5)
        if time() - start > timeout:
            assert False, assertion_message or "Lock is not acquired."
        logger_dask.debug("Lock is not acquired. Retry to check locked.")


class MetaColumnNames(StrEnum):
    prm_num_value = 'prm.num.value'
    prm_cat_value = 'prm.cat.value'


def create_err_msg_from_exception(e: Exception):
    """:meta private:"""
    additional = ' '.join(map(str, e.args))
    if additional == '':
        return type(e).__name__
    else:
        return type(e).__name__ + f'({additional})'


class TrialState(StrEnum):

    succeeded = 'Success'
    skipped = 'Skip'
    hard_constraint_violation = 'Hard constraint violation'
    soft_constraint_violation = 'Soft constraint violation'

    # Hidden Constraint
    model_error = 'Model error'
    mesh_error = 'Mesh error'
    solve_error = 'Solve error'
    post_error = 'Post-processing error'

    unknown_error = 'Unknown error'
    undefined = 'undefined'

    @staticmethod
    def get_corresponding_state_from_exception(e: Exception) -> TrialState:
        """:meta private:"""
        if isinstance(e, ModelError):
            state = TrialState.model_error
        elif isinstance(e, MeshError):
            state = TrialState.mesh_error
        elif isinstance(e, SolveError):
            state = TrialState.solve_error
        elif isinstance(e, PostProcessError):
            state = TrialState.post_error
        elif isinstance(e, HardConstraintViolation):
            state = TrialState.hard_constraint_violation
        elif isinstance(e, SkipSolve):
            state = TrialState.skipped
        else:
            state = TrialState.unknown_error
        return state

    @staticmethod
    def get_corresponding_exception_from_state(state: TrialState) -> Exception | None:
        """:meta private:"""
        if state == TrialState.model_error:
            e = ModelError()
        elif state == TrialState.mesh_error:
            e = MeshError()
        elif state == TrialState.solve_error:
            e = SolveError()
        elif state == TrialState.post_error:
            e = PostProcessError()
        elif state == TrialState.unknown_error:
            e = Exception()
        elif state == TrialState.hard_constraint_violation:
            e = HardConstraintViolation()
        elif state == TrialState.skipped:
            e = SkipSolve()
        else:
            e = None
        return e

    @classmethod
    def get_hidden_constraint_violation_states(cls):
        """:meta private:"""
        return [cls.get_corresponding_state_from_exception(exception_type())
                for exception_type in _HiddenConstraintViolation.__pyfemtet_subclasses__]


class DataFrameWrapper:
    """:meta private:"""

    __df: pd.DataFrame
    _lock_name = 'edit-df'
    _dataset_name: str
    _scheduler_address: str

    def __init__(self, df: pd.DataFrame):
        self._dataset_name = 'df-' + generate_random_id()
        self._scheduler_address = None
        self.set_df(df)

    def __len__(self):
        return len(self.get_df())

    def __str__(self):
        return self.get_df().__str__()

    @property
    def lock(self):
        return Lock(self._lock_name)

    def get_df(self, equality_filters: dict = None) -> pd.DataFrame:
        """

        Args:
            equality_filters (dict, optional):
                {column: value} formatted dict.
                Each condition is considered as
                an 'and' condition.

                Defaults to no filter.

        Returns (pd.DataFrame):

        """

        client = get_client(self._scheduler_address)

        # dask クラスターがある場合
        if client is not None:

            # あるけど with を抜けている場合
            if client.scheduler is None:
                df = self.__df

            # 健在の場合
            else:

                self._scheduler_address = client.scheduler.address

                df = None

                with Lock('access_dataset_df'):
                    # datasets 内に存在する場合
                    if self._dataset_name in client.list_datasets():
                        df = client.get_dataset(self._dataset_name)

                    # 存在しない場合は publish する
                    else:
                        df = self.__df
                        client.publish_dataset(**{self._dataset_name: df})
                        sleep(0.1)

                assert df is not None

        # dask クラスターがない場合
        else:
            df = self.__df

        # filter に合致するものを取得
        if equality_filters is not None:
            df = get_partial_df(df, equality_filters)

        return df

    def set_df(self, df, equality_filters: dict = None):
        """

        Args:
            df:
            equality_filters (dict, optional):
                {column: value} formatted dict.
                Each condition is considered as
                an 'and' condition.
                Only the indexed rows will be updated.

                Defaults to no filter.

        Returns (pd.DataFrame):

        """

        # フィルタを適用
        # partial_df を get_df した時点のものから
        # 変わっていたらエラーになる
        if equality_filters is not None:
            _assert_locked_with_timeout(self.lock, 'set_df() with equality_filters must be called with locking.')
            partial_df = df
            df = self.get_df()
            apply_partial_df(df, partial_df, equality_filters)

        # dask クラスター上のデータを更新
        client = get_client(self._scheduler_address)
        if client is not None:
            if client.scheduler is not None:

                self._scheduler_address = client.scheduler.address

                with Lock('access_dataset_df', client):

                    # datasets 上に存在する場合は削除（上書きができない）
                    if self._dataset_name in client.list_datasets():

                        # remove
                        client.unpublish_dataset(self._dataset_name)

                    # update
                    client.publish_dataset(**{self._dataset_name: df})
                    sleep(0.1)

        # local のデータを更新
        self.__df = df

    def start_dask(self):
        # Register the df initialized before dask context.
        self.set_df(self.__df)

    def end_dask(self):
        # Get back the df on dask to use the value outside
        # dask context.
        self.__df = self.get_df()


class CorrespondingColumnNameRuler:
    """:meta private:"""

    @staticmethod
    def cns_lower_bound_name(cns_name):
        return cns_name + '_lower_bound'

    @staticmethod
    def cns_upper_bound_name(cns_name):
        return cns_name + '_upper_bound'

    @staticmethod
    def direction_name(obj_name):
        return obj_name + '_direction'

    @staticmethod
    def prm_lower_bound_name(prm_name):
        return prm_name + '_lower_bound'

    @staticmethod
    def prm_upper_bound_name(prm_name):
        return prm_name + '_upper_bound'

    @staticmethod
    def prm_choices_name(prm_name):
        return prm_name + '_choices'

    @staticmethod
    def prm_step_name(prm_name):
        return prm_name + '_step'


class ColumnOrderMode(StrEnum):
    """The order rule of the history csv columns."""
    per_category = 'per_category'  #: Sort per each object.
    important_first = 'important_first'  #: The values of parameters and objectives first.


ColumnOrderModeStr: TypeAlias = Literal['per_category', 'important_first']


class DuplicatedColumnNameError(Exception):
    """:meta private:"""


class NoDuplicateDict(dict):
    def update(self, m: dict, /, **kwargs):
        for key_ in m.keys():
            if key_ in self.keys():
                raise DuplicatedColumnNameError(
                    _(
                        en_message='The name `{name}` is duplicated. '
                                   'Please use another name.',
                        jp_message='名前 「{name}」 が重複しています。'
                                   '別の名前を使ってください。',
                        name=key_,
                    )
                )

        super().update(m, **kwargs)


class ColumnManager:
    """:meta private:"""

    parameters: TrialInput
    y_names: list[str]
    c_names: list[str]
    other_output_names: list[str]
    column_dtypes: dict[str, type]
    meta_columns: list[str]

    @staticmethod
    def columns_to_keep_even_if_nan():
        return [
            'messages',
        ]

    def initialize(
            self,
            parameters: TrialInput,
            y_names,
            c_names,
            other_output_names,
            additional_data: dict,
            column_order_mode: str = ColumnOrderMode.per_category,
    ):
        self.parameters = parameters
        self.y_names = y_names
        self.c_names = c_names
        self.other_output_names=other_output_names
        self.set_full_sorted_column_information(
            additional_data=additional_data,
            column_order_mode=column_order_mode,
        )

    def set_full_sorted_column_information(
            self,
            extra_parameters: TrialInput = None,
            extra_y_names: list[str] = None,
            extra_c_names: list[str] = None,
            extra_other_output_names: list[str] = None,
            additional_data: dict = None,
            column_order_mode: str = ColumnOrderMode.per_category,
    ):
        extra_parameters = extra_parameters or TrialInput()
        extra_y_names = extra_y_names or []
        extra_c_names = extra_c_names or []
        extra_other_output_names = extra_other_output_names or []

        # column name になるので重複は許されない
        column_dtypes: dict = NoDuplicateDict()
        meta_columns: list = []
        column_dtypes_later: dict = NoDuplicateDict()
        meta_columns_later: list = []

        if column_order_mode == ColumnOrderMode.per_category:
            target_cds: dict = column_dtypes
            target_mcs: list = meta_columns
        elif column_order_mode == ColumnOrderMode.important_first:
            target_cds: dict = column_dtypes_later
            target_mcs: list = meta_columns_later
        else:
            assert False, f'Unknown {column_order_mode=}'

        # noinspection PyUnresolvedReferences
        keys = Record.__dataclass_fields__.copy().keys()
        for key in keys:
            # Note:
            #   as_df() で空欄になりうるカラムには
            #   Nan や '' を許容する dtype を指定すること
            #   例えば、 trial に int を指定してはいけない
            #
            # Note:
            #   pandas は column_dtypes に str を受け付けない
            #   (object にキャストされる模様)

            if key == 'x':
                for prm_name in self.parameters.keys():

                    param = self.parameters[prm_name]

                    if isinstance(param, NumericParameter):
                        # important
                        column_dtypes.update({prm_name: float})
                        meta_columns.append(MetaColumnNames.prm_num_value.value)

                        # later
                        f = CorrespondingColumnNameRuler.prm_lower_bound_name
                        target_cds.update({f(prm_name): float})
                        target_mcs.append('prm.num.lower_bound')

                        f = CorrespondingColumnNameRuler.prm_upper_bound_name
                        target_cds.update({f(prm_name): float})
                        target_mcs.append('prm.num.upper_bound')

                        f = CorrespondingColumnNameRuler.prm_step_name
                        target_cds.update({f(prm_name): float})
                        target_mcs.append('prm.num.step')

                    elif isinstance(param, CategoricalParameter):
                        # important
                        column_dtypes.update({prm_name: object})
                        meta_columns.append(MetaColumnNames.prm_cat_value.value)

                        # later
                        f = CorrespondingColumnNameRuler.prm_choices_name
                        target_cds.update({f(prm_name): object})
                        target_mcs.append('prm.cat.choices')

                    else:
                        raise NotImplementedError

                for extra_prm_name, extra_param in extra_parameters.items():

                    if isinstance(extra_param, NumericParameter):
                        # later
                        target_cds.update({extra_prm_name: float})
                        target_mcs.append('')

                        f = CorrespondingColumnNameRuler.prm_lower_bound_name
                        target_cds.update({f(extra_prm_name): object})
                        target_mcs.append('')

                        f = CorrespondingColumnNameRuler.prm_upper_bound_name
                        target_cds.update({f(extra_prm_name): object})
                        target_mcs.append('')

                    elif isinstance(extra_param, CategoricalParameter):
                        target_cds.update({extra_prm_name: object})
                        target_mcs.append('')

                        f = CorrespondingColumnNameRuler.prm_choices_name
                        target_cds.update({f(extra_prm_name): object})
                        target_mcs.append('')

                    else:
                        raise NotImplementedError

            elif key == 'y':
                f = CorrespondingColumnNameRuler.direction_name
                for name in self.y_names:
                    # important
                    column_dtypes.update({name: float})
                    meta_columns.append('obj')

                    # later
                    target_cds.update({f(name): object})  # str | float
                    target_mcs.append('obj.direction')

                for name in extra_y_names:
                    # later
                    target_cds.update({name: float})
                    target_mcs.append('')

                    # later
                    target_cds.update({f(name): object})  # str | float
                    target_mcs.append('')

            elif key == 'c':

                f_lb = CorrespondingColumnNameRuler.cns_lower_bound_name
                f_ub = CorrespondingColumnNameRuler.cns_upper_bound_name

                for name in self.c_names:
                    # important
                    column_dtypes.update({name: float})
                    meta_columns.append('cns')

                    # later
                    target_cds.update({f_lb(name): float})
                    target_mcs.append('cns.lower_bound')

                    # later
                    target_cds.update({f_ub(name): float})
                    target_mcs.append('cns.upper_bound')

                for name in extra_c_names:
                    # later
                    target_cds.update({name: float})
                    target_mcs.append('')

                    # later
                    target_cds.update({f_lb(name): float})
                    target_mcs.append('')

                    # later
                    target_cds.update({f_ub(name): float})
                    target_mcs.append('')

            elif key == 'other_outputs':
                for name in self.other_output_names:
                    # important
                    column_dtypes.update({name: float})
                    meta_columns.append('other_output.value')

                for name in extra_other_output_names:
                    # later
                    target_cds.update({name: float})
                    target_mcs.append('')

            # additional_data を入れる
            elif key == self._get_additional_data_column():
                # important
                column_dtypes.update({key: object})
                meta_columns.append(json.dumps(additional_data or dict()))

            elif key in (
                'feasibility',
                'optimality',
                'sub_sampling',
                'sub_fidelity_name',
            ):
                # important
                column_dtypes.update({key: object})
                meta_columns.append('')

            else:
                # later
                target_cds.update({key: object})
                target_mcs.append('')

        column_dtypes.update(column_dtypes_later)
        meta_columns.extend(meta_columns_later)

        self.column_dtypes = dict(**column_dtypes)
        self.meta_columns = meta_columns

    @staticmethod
    def _get_additional_data_column():
        return 'trial'

    @classmethod
    def _get_additional_data(cls, columns, meta_columns) -> dict:
        for column, meta_column in zip(columns, meta_columns):
            if column == cls._get_additional_data_column():
                if meta_column:
                    return json.loads(meta_column)
                else:
                    return json.loads('{}')
        else:
            raise RuntimeError(f'"{cls._get_additional_data_column()}" is not found in given columns.')

    @staticmethod
    def _filter_columns(meta_column, columns, meta_columns) -> list[str]:
        out = []
        assert len(columns) == len(meta_columns), f'{len(columns)=} and {len(meta_columns)=}'

        for i, (column_, meta_column_) in enumerate(zip(columns, meta_columns)):
            if meta_column_ == meta_column:
                out.append(column_)
        return out

    @classmethod
    def _filter_prm_names(cls, columns, meta_columns) -> list[str]:
        return (
                cls._filter_columns('prm.num.value', columns, meta_columns)
                + cls._filter_columns('prm.cat.value', columns, meta_columns)
            )

    def filter_columns(self, meta_column) -> list[str]:
        columns = list(self.column_dtypes.keys())
        return self._filter_columns(meta_column, columns, self.meta_columns)

    def get_prm_names(self) -> list[str]:
        return (
            self.filter_columns('prm.num.value')
            + self.filter_columns('prm.cat.value')
        )

    def get_obj_names(self) -> list[str]:
        return self.filter_columns('obj')

    def get_cns_names(self) -> list[str]:
        return self.filter_columns('cns')

    def get_other_output_names(self) -> list[str]:
        return self.filter_columns('other_output')

    @staticmethod
    def _is_numerical_parameter(prm_name, columns, meta_columns):
        col_index = tuple(columns).index(prm_name)
        meta_column = meta_columns[col_index]
        return meta_column == MetaColumnNames.prm_num_value

    @staticmethod
    def _is_categorical_parameter(prm_name, columns, meta_columns):
        col_index = tuple(columns).index(prm_name)
        meta_column = meta_columns[col_index]
        return meta_column == MetaColumnNames.prm_cat_value

    def is_numerical_parameter(self, prm_name) -> bool:
        return self._is_numerical_parameter(prm_name, tuple(self.column_dtypes.keys()), self.meta_columns)

    def is_categorical_parameter(self, prm_name) -> bool:
        return self._is_categorical_parameter(prm_name, tuple(self.column_dtypes.keys()), self.meta_columns)

    @staticmethod
    def _get_parameter(prm_name: str, df: pd.DataFrame, meta_columns) -> Parameter:
        if ColumnManager._is_numerical_parameter(prm_name, df.columns, meta_columns):
            out = NumericParameter()
            out.name = prm_name
            out.value = float(df[prm_name].dropna().values[-1])

            # lower_bound
            key = CorrespondingColumnNameRuler.prm_lower_bound_name(prm_name)
            if key in df.columns:
                out.lower_bound = float(df[key].dropna().values[-1])
            else:
                out.lower_bound = None

            # upper bound
            key = CorrespondingColumnNameRuler.prm_upper_bound_name(prm_name)
            if key in df.columns:
                out.upper_bound = float(df[key].dropna().values[-1])
            else:
                out.upper_bound = None

            # step
            key = CorrespondingColumnNameRuler.prm_step_name(prm_name)
            if key in df.columns:
                out.step = float(df[key].dropna().values[-1])
            else:
                out.step = None

        elif ColumnManager._is_categorical_parameter(prm_name, df.columns, meta_columns):
            out = CategoricalParameter()
            out.name = prm_name
            out.value = str(df[prm_name].dropna().values[-1])
            out.choices = df[CorrespondingColumnNameRuler.prm_choices_name(prm_name)].dropna().values[-1]

        else:
            raise NotImplementedError

        return out

    @staticmethod
    def _reconvert_objects(df: pd.DataFrame, meta_columns: list[str]):
        for column, meta_column in zip(df.columns, meta_columns):
            # messages は df の段階で _RECORD_MESSAGE_DELIMITER
            # separated な str なのでここで restore してはいけない

            # choices list は csv を経由することで str になるので restore
            if meta_column == 'prm.cat.choices':
                df[column] = [ast.literal_eval(d) for d in df[column]]

    @staticmethod
    def _get_sub_fidelity_names(df: pd.DataFrame) -> list[str]:

        if 'sub_fidelity_name' not in df.columns:
            return [MAIN_FIDELITY_NAME]

        else:
            return np.unique(df['sub_fidelity_name'].values).tolist()


_RECORD_MESSAGE_DELIMITER = ' | '


@dataclasses.dataclass
class Record:
    """:meta private:"""

    # x, y, c のみ特殊で、データの展開や関連情報の
    # 列への展開を必要とするが、他の field は
    # ここに定義すればよい

    trial: int = None
    trial_id: int = None
    sub_sampling: SubSampling | None = None
    sub_fidelity_name: str = None
    fidelity: Fidelity = None
    x: TrialInput = dataclasses.field(default_factory=TrialInput)
    y: TrialOutput = dataclasses.field(default_factory=TrialOutput)
    c: TrialConstraintOutput = dataclasses.field(default_factory=TrialConstraintOutput)
    other_outputs: TrialFunctionOutput = dataclasses.field(default_factory=TrialFunctionOutput)
    state: TrialState = TrialState.undefined
    datetime_start: datetime.datetime = dataclasses.field(default_factory=datetime.datetime.now)
    datetime_end: datetime.datetime = dataclasses.field(default_factory=datetime.datetime.now)
    messages: list = dataclasses.field(default_factory=list)
    hypervolume: float | None = None
    feasibility: bool | None = None
    optimality: bool | None = None

    def as_df(self, dtypes: dict = None):

        # noinspection PyUnresolvedReferences
        keys = self.__dataclass_fields__.copy().keys()
        d = {key: getattr(self, key) for key in keys if getattr(self, key) is not None}

        x: TrialInput = d.pop('x')
        y: TrialOutput = d.pop('y')
        c: TrialConstraintOutput = d.pop('c')
        other_outputs: TrialFunctionOutput = d.pop('other_outputs')

        # prm
        for prm_name, param in x.items():
            d.update({prm_name: param.value})
            if isinstance(param, NumericParameter):
                f = CorrespondingColumnNameRuler.prm_lower_bound_name
                d.update({f(prm_name): param.lower_bound})
                f = CorrespondingColumnNameRuler.prm_upper_bound_name
                d.update({f(prm_name): param.upper_bound})
                f = CorrespondingColumnNameRuler.prm_step_name
                d.update({f(prm_name): param.step})
            elif isinstance(param, CategoricalParameter):
                f = CorrespondingColumnNameRuler.prm_choices_name
                d.update({f(prm_name): param.choices})
            else:
                raise NotImplementedError
        
        # messages to str
        messages_str = _RECORD_MESSAGE_DELIMITER.join(d['messages'])
        d.update({'messages': messages_str})

        # obj
        d.update(**{k: v.value for k, v in y.items()})
        d.update(**{f'{CorrespondingColumnNameRuler.direction_name(k)}': v.direction for k, v in y.items()})

        # cns
        d.update(**{k: v.value for k, v in c.items()})
        f_lb = CorrespondingColumnNameRuler.cns_lower_bound_name
        d.update(**{f'{f_lb(k)}': v.lower_bound
                    for k, v in c.items()})
        f_ub = CorrespondingColumnNameRuler.cns_upper_bound_name
        d.update(**{f'{f_ub(k)}': v.upper_bound
                    for k, v in c.items()})

        # function
        d.update(**{k: v.value for k, v in other_outputs.items()})

        df = pd.DataFrame(
            {k: [v] for k, v in d.items()},
            columns=tuple(dtypes.keys())
        )

        if dtypes:
            df = df.astype(dtypes)

        return df

    @staticmethod
    def get_state_str_from_series(row: pd.Series):
        state: TrialState = TrialState.undefined
        if 'state' in row:
            state = row['state']
        return state


class EntireDependentValuesCalculator:
    """:meta private:"""

    def __init__(
            self,
            records: Records,
            equality_filters: dict,
            entire_df: pd.DataFrame,
    ):

        self.records = records
        self.equality_filters = equality_filters
        self.entire_df: pd.DataFrame = entire_df
        self.partial_df: pd.DataFrame = get_partial_df(entire_df, equality_filters)

        _assert_locked_with_timeout(self.records.df_wrapper.lock)

        # get column names
        obj_names = self.records.column_manager.get_obj_names()
        f = CorrespondingColumnNameRuler.direction_name
        obj_direction_names = [f(name) for name in obj_names]

        # get values
        all_obj_values = self.partial_df[obj_names].values
        all_obj_directions = self.partial_df[obj_direction_names].values
        feasibility = self.partial_df['feasibility']

        # convert values as minimization problem
        y_internal = np.empty(all_obj_values.shape)
        for i, (obj_values, obj_directions) \
                in enumerate(zip(all_obj_values.T, all_obj_directions.T)):
            y_internal[:, i] = np.array(
                list(
                    map(
                        lambda args: Objective._convert(*args),
                        zip(obj_values, obj_directions)
                    )
                )
            )

        self.partial_y_internal = y_internal
        self.partial_feasibility = feasibility

    def update_optimality(self):

        _assert_locked_with_timeout(self.records.df_wrapper.lock)

        # calc optimality
        optimality = calc_optimality(
            self.partial_y_internal,
            self.partial_feasibility,
        )

        # update
        self.partial_df.loc[:, 'optimality'] = optimality

    def update_hypervolume(self):

        _assert_locked_with_timeout(self.records.df_wrapper.lock)

        # calc hypervolume
        hv_values = calc_hypervolume(
            self.partial_y_internal,
            self.partial_feasibility,
            ref_point='optuna-nadir',
        )

        # update
        self.partial_df.loc[:, 'hypervolume'] = hv_values

    def update_trial_number(self):

        _assert_locked_with_timeout(self.records.df_wrapper.lock)

        # calc trial
        trial_number = 1 + np.arange(len(self.partial_df)).astype(int)

        # update
        self.partial_df.loc[:, 'trial'] = trial_number


class Records:
    """:meta private:

    最適化の試行全体の情報を格納するモデルクラス
    """
    df_wrapper: DataFrameWrapper
    column_manager: ColumnManager

    def __init__(self):
        self.df_wrapper = DataFrameWrapper(pd.DataFrame())
        self.column_manager = ColumnManager()
        self.loaded_meta_columns = None
        self.loaded_df = None

    def __str__(self):
        return self.df_wrapper.__str__()

    def __len__(self):
        return len(self.df_wrapper)

    def initialize(self):
        with self.df_wrapper.lock:
            # 新しく始まる場合に備えカラムを設定
            # load の場合はあとで上書きされる
            df = pd.DataFrame([], columns=list(self.column_manager.column_dtypes.keys()))
            self.df_wrapper.set_df(df)

    def load(self, path: str):

        for encoding in (ENCODING, 'utf-8'):
            try:
                with open(path, 'r', encoding=encoding, newline='\n') as f:
                    reader = csv.reader(f, delimiter=',')
                    # load meta_column
                    loaded_meta_columns = reader.__next__()
                    reader.__next__()  # empty line
                    # load df from line 3
                    loaded_df = pd.read_csv(f, encoding=encoding, header=0)
                break

            except UnicodeDecodeError:
                continue

        # df を csv にする過程で失われる list などのオブジェクトを restore
        ColumnManager._reconvert_objects(loaded_df, loaded_meta_columns)

        # この段階では column_dtypes が setup されていない可能性があるので
        # compatibility check をしない。よって set_df しない。
        self.loaded_meta_columns = loaded_meta_columns
        self.loaded_df = loaded_df

    def check_problem_compatibility(self):

        # 読み込んだデータがないのであれば何もしない
        if self.loaded_df is None:
            return

        # 順番が違ってもいいが、
        # 構成に変更がないこと。
        # ただし obj は減っていてもいい。
        loaded_columns, loaded_meta_columns = self.loaded_df.columns, self.loaded_meta_columns

        # prm_names が過不足ないか
        loaded_prm_names = set(
            self.column_manager._filter_prm_names(
                loaded_columns, loaded_meta_columns
            )
        )
        prm_names = set(self.column_manager.get_prm_names())
        if not (len(loaded_prm_names - prm_names) == len(prm_names - loaded_prm_names) == 0):
            raise RuntimeError('Incompatible parameter setting.')

        # obj_names が増えていないか
        loaded_obj_names = set(self.column_manager._filter_columns('obj', loaded_columns, loaded_meta_columns))
        obj_names = set(self.column_manager.get_obj_names())
        if len(obj_names - loaded_obj_names) > 0:
            raise RuntimeError('Incompatible objective setting.')

        # cns_names が過不足ないか
        # TODO: cns の上下限は変更されてはならない。
        loaded_cns_names = set(self.column_manager._filter_columns('cns', loaded_columns, loaded_meta_columns))
        cns_names = set(self.column_manager.get_cns_names())
        if not (len(loaded_cns_names - cns_names) == len(cns_names - loaded_cns_names) == 0):
            raise RuntimeError('Incompatible constraint setting.')

    def reinitialize_record_with_loaded_data(self, column_order_mode: str = ColumnOrderMode.per_category):

        # 読み込んだデータがないのであれば何もしない
        if self.loaded_df is None:
            return

        loaded_columns, loaded_meta_columns = self.loaded_df.columns, self.loaded_meta_columns
        loaded_prm_names = set(self.column_manager._filter_prm_names(loaded_columns, loaded_meta_columns))
        loaded_obj_names = set(self.column_manager._filter_columns('obj', loaded_columns, loaded_meta_columns))
        loaded_cns_names = set(self.column_manager._filter_columns('cns', loaded_columns, loaded_meta_columns))
        loaded_other_output_names = set(self.column_manager._filter_columns('other_output.value', loaded_columns, loaded_meta_columns))

        # loaded df に存在するが Record に存在しないカラムを Record に追加
        extra_parameters = {}
        extra_y_names = []
        extra_c_names = []
        extra_oo_names = []
        for l_col, l_meta in zip(loaded_columns, loaded_meta_columns):

            # 現在の Record に含まれないならば
            if l_col not in self.column_manager.column_dtypes.keys():

                # それが prm_name ならば
                if l_col in loaded_prm_names:

                    # それが Categorical ならば
                    if CorrespondingColumnNameRuler.prm_choices_name(l_col) in loaded_columns:
                        param = CategoricalParameter()
                        param.name = l_col
                        param.value = ''
                        param.choices = []

                    # それが Numeric ならば
                    elif CorrespondingColumnNameRuler.prm_lower_bound_name(l_col) in loaded_columns:
                        param = NumericParameter()
                        param.name = l_col
                        param.value = np.nan
                        param.lower_bound = np.nan
                        param.upper_bound = np.nan

                    else:
                        raise NotImplementedError

                    extra_parameters.update({l_col: param})

                # obj_name ならば
                elif l_col in loaded_obj_names:
                    extra_y_names.append(l_col)

                # cns_name ならば
                elif l_col in loaded_cns_names:
                    extra_c_names.append(l_col)

                # other_output_name ならば
                elif l_col in loaded_other_output_names:
                    extra_oo_names.append(l_col)

        # additional data を取得
        a_data = self.column_manager._get_additional_data(loaded_columns, loaded_meta_columns)

        self.column_manager.set_full_sorted_column_information(
            extra_parameters=extra_parameters,
            extra_y_names=extra_y_names,
            extra_c_names=extra_c_names,
            extra_other_output_names=extra_oo_names,
            additional_data=a_data,
            column_order_mode=column_order_mode,
        )

        # worker に影響しないように loaded_df のコピーを作成
        df: pd.DataFrame = self.loaded_df.copy()

        # loaded df に存在しないが Record に存在するカラムを追加
        for col in self.column_manager.column_dtypes.keys():
            if col not in df.columns:
                # column ごとの default 値を追加
                if col == 'sub_fidelity_name':
                    df[col] = MAIN_FIDELITY_NAME
                else:
                    df[col] = np.nan

        # column_dtypes を設定
        # 与える column_dtypes のほうが多い場合
        # エラーになるので余分なものを削除
        # 与える column_dtypes が少ない分には
        # (pandas としては) 問題ない
        dtypes = {k: v for k, v in self.column_manager.column_dtypes.items() if k in self.loaded_df.columns}
        df = df.astype(dtypes)

        # 並べ替え
        df = df[list(self.column_manager.column_dtypes.keys())].astype(self.column_manager.column_dtypes)

        # OK なので読み込んだデータを set_df する
        self.df_wrapper.set_df(df)

    def remove_nan_columns(
            self, df, meta_columns, columns_to_keep: str | list[str] = None
    ) -> tuple[pd.DataFrame, tuple[str]]:
        """

        Args:
            df:
            meta_columns:
            columns_to_keep: Allowing these columns to all NaN values.

        Returns:
            Removed DataFrame and corresponding meta_columns.

        """

        df = df.replace('', None)

        nan_columns = df.isna().all(axis=0)
        if columns_to_keep is None:
            columns_to_keep = self.column_manager.columns_to_keep_even_if_nan()
        nan_columns[columns_to_keep] = False

        fdf = df.loc[:, ~nan_columns]
        f_meta_columns = (np.array(meta_columns)[~nan_columns]).tolist()

        return fdf, f_meta_columns

    def save(self, path: str):

        # filter NaN columns
        df, meta_columns = self.remove_nan_columns(
            self.df_wrapper.get_df(), self.column_manager.meta_columns,
        )

        try:
            with open(path, 'w', encoding=ENCODING) as f:
                writer = csv.writer(f, delimiter=',', lineterminator="\n")
                # write meta_columns
                writer.writerow(meta_columns)
                writer.writerow([''] * len(meta_columns))  # empty line
                # write df from line 3
                df.to_csv(f, index=False, encoding=ENCODING, lineterminator='\n')
        except PermissionError:
            logger.warning(
                _(
                    en_message='History csv file ({path}) is in use and cannot be written to. '
                               'Please free this file before exiting the program, '
                               'otherwise history data will be lost.',
                    jp_message='履歴のCSVファイル（{path}）が使用中のため書き込みできません。'
                               'プログラムを終了する前にこのファイルを閉じてください。'
                               'そうしない場合、履歴データが失われます。',
                    path=path,
                )
            )

    def append(self, record: Record) -> pd.Series:

        # get row
        row = record.as_df(dtypes=self.column_manager.column_dtypes)

        # concat
        dfw = self.df_wrapper

        # append
        with dfw.lock:

            df = dfw.get_df()

            if len(df) == 0:
                # ここで空のカラムを削除しては
                # データの並びがおかしくなるケースが出る
                new_df = row
            else:
                # pandas の型推定の仕様変更対策で
                # 空のカラムは削除する
                row.dropna(axis=1, inplace=True, how='all')

                new_df = pd.concat(
                    [df, row],
                    axis=0,
                    ignore_index=True,
                )

            # calc entire-dependent values
            # must be in with block to keep
            # the entire data compatibility
            # during processing.
            self.update_entire_dependent_values(new_df)

            dfw.set_df(new_df)

            # postprocess after recording で使うために
            # 計算済み最終行を返す
            return new_df.iloc[-1]

    def update_entire_dependent_values(self, processing_df: pd.DataFrame):

        _assert_locked_with_timeout(self.df_wrapper.lock)

        # check trial_id is filled
        trial_processed = False
        if processing_df['trial_id'].notna().all():
            id_to_n: dict = {tid: i + 1 for i, tid
                             in enumerate(processing_df['trial_id'].unique())}
            processing_df['trial'] = processing_df['trial_id'].map(id_to_n)
            trial_processed = True

        # update main fidelity
        equality_filters = MAIN_FILTER
        mgr = EntireDependentValuesCalculator(
            self,
            equality_filters,
            processing_df,
        )
        mgr.update_optimality()
        mgr.update_hypervolume()
        if not trial_processed:
            mgr.update_trial_number()  # per_fidelity
        pdf = mgr.partial_df
        apply_partial_df(df=processing_df, partial_df=pdf, equality_filters=equality_filters)

        # update sub fidelity
        sub_fidelity_names: list = np.unique(processing_df['sub_fidelity_name']).tolist()
        if MAIN_FIDELITY_NAME in sub_fidelity_names:
            sub_fidelity_names.remove(MAIN_FIDELITY_NAME)
        for sub_fidelity_name in sub_fidelity_names:
            equality_filters = {'sub_fidelity_name': sub_fidelity_name}
            mgr = EntireDependentValuesCalculator(
                self,
                equality_filters,
                processing_df
            )
            if not trial_processed:
                mgr.update_trial_number()  # per_fidelity
        pdf = mgr.partial_df
        apply_partial_df(df=processing_df, partial_df=pdf, equality_filters=equality_filters)


class History:
    """最適化の試行の履歴を管理します。"""
    _records: Records
    prm_names: list[str]
    obj_names: list[str]
    cns_names: list[str]
    other_output_names: list[str]
    sub_fidelity_names: list[str]
    is_restart: bool
    additional_data: dict

    path: str
    """The existing or destination CSV path.

    If not specified, the CSV file is saved in the format
    "pyfemtet.opt_%Y%m%d_%H%M%S.csv"
    when the optimization process starts.
    """

    @property
    def all_output_names(self) -> list[str]:
        return self.obj_names + self.cns_names + self.other_output_names

    def __init__(self):
        self._records = Records()
        self.path: str | None = None
        self._finalized: bool = False
        self.is_restart = False
        self.additional_data = dict(version=pyfemtet.__version__)
        self.column_order_mode: ColumnOrderMode | ColumnOrderModeStr = ColumnOrderMode.per_category

    def __str__(self):
        return self._records.__str__()

    def __enter__(self):
        self._records.df_wrapper.start_dask()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._records.df_wrapper.end_dask()

    def load_csv(self, path, with_finalize=False):
        """:meta private:"""

        self.is_restart = True
        self.path = path
        self._records.load(self.path)

        if with_finalize:
            self._finalize_from_loaded_data()

    def _finalize_from_loaded_data(self):
        assert self.is_restart

        if not self._finalized:

            df = self._records.loaded_df
            meta_columns = self._records.loaded_meta_columns

            self.prm_names = ColumnManager._filter_prm_names(df.columns, meta_columns)
            self.obj_names = ColumnManager._filter_columns('obj', df.columns, meta_columns)
            self.cns_names = ColumnManager._filter_columns('cns', df.columns, meta_columns)
            self.other_output_names = ColumnManager._filter_columns('other_output.value', df.columns, meta_columns)
            self.sub_fidelity_names = ColumnManager._get_sub_fidelity_names(df)
            self.additional_data = ColumnManager._get_additional_data(df.columns, meta_columns)

            parameters: TrialInput = {}
            for prm_name in self.prm_names:
                param = ColumnManager._get_parameter(prm_name, df, meta_columns)
                parameters.update({prm_name: param})

            self.finalize(
                parameters,
                self.obj_names,
                self.cns_names,
                self.other_output_names,
                self.sub_fidelity_names,
                self.additional_data,
            )

    def finalize(
            self,
            parameters: TrialInput,
            obj_names,
            cns_names,
            other_output_names,
            sub_fidelity_names,
            additional_data,
    ):
        """:meta private:"""

        self.prm_names = list(parameters.keys())
        self.obj_names = list(obj_names)
        self.cns_names = list(cns_names)
        self.other_output_names = list(other_output_names)
        self.sub_fidelity_names = list(sub_fidelity_names)
        self.additional_data.update(additional_data)

        if not self._finalized:
            # ここで column_dtypes が決定する
            self._records.column_manager.initialize(
                parameters, self.obj_names, self.cns_names, self.other_output_names,
                self.additional_data, self.column_order_mode
            )

            # initialize
            self._records.initialize()

            if self.path is None:
                self.path = datetime.datetime.now().strftime("pyfemtet.opt_%Y%m%d_%H%M%S.csv")

            # load
            if os.path.isfile(self.path):
                self.load_csv(self.path)
                self._records.check_problem_compatibility()
                self._records.reinitialize_record_with_loaded_data(self.column_order_mode)

        self._finalized = True

    def get_df(self, equality_filters: dict = None) -> pd.DataFrame:
        """Returns the optimization history.

        Args:
            equality_filters (dict, optional):
                The {column: value} というフォーマットの
                matching filter.

        Returns: The optimization history.

        """
        return self._records.df_wrapper.get_df(equality_filters)

    @staticmethod
    def get_trial_name(trial=None, fidelity=None, sub_sampling=None, row: pd.Series = None):
        if row is not None:
            assert not math.isnan(row['trial'])
            trial = row['trial']
            fidelity = row['fidelity'] if not math.isnan(row['fidelity']) else None
            sub_sampling = row['sub_sampling'] if not math.isnan(row['sub_sampling']) else None

        name_parts = ['trial']
        if fidelity is not None:
            fid = str(fidelity)
            if fid != MAIN_FIDELITY_NAME:
                name_parts.append(fid)

        name_parts.append(str(trial))

        if sub_sampling is not None:
            name_parts.append(str(sub_sampling))

        trial_name = '_'.join(name_parts)

        return trial_name

    def recording(self, fem: AbstractFEMInterface):
        """:meta private:"""

        # noinspection PyMethodParameters
        class RecordContext:

            def __init__(self_):
                self_.record = Record()
                self_.record_as_df = None

            def __enter__(self_):
                return self_.record

            def append(self_):
                self_.record.datetime_end = self_.record.datetime_end \
                    if self_.record.datetime_end is not None \
                    else datetime.datetime.now()
                return self._records.append(self_.record)

            @staticmethod
            def postprocess_after_recording(row):

                client = get_client(self._records.df_wrapper._scheduler_address)

                trial_name = self.get_trial_name(row=row)

                # FIXME: メインフィデリティだけでなく、FEM に
                #   対応するフィデリティ又はサブサンプリングのみ
                #   フィルタした情報を提供するようにする。
                #   フィデリティの話は現在解析を実行している opt が
                #   必要なので、recording メソッドの引数に
                #   それを追加する
                df = self.get_df(equality_filters=MAIN_FILTER)

                if client is not None:
                    client.run_on_scheduler(
                        fem._postprocess_after_recording,
                        trial_name=trial_name,
                        df=df,
                        **(fem._create_postprocess_args()),
                    )

                else:
                    fem._postprocess_after_recording(
                        dask_scheduler=None,
                        trial_name=trial_name,
                        df=df,
                        **(fem._create_postprocess_args())
                    )

            def __exit__(self_, exc_type, exc_val, exc_tb):

                row: pd.Series | None = None

                # record feasibility
                # skipped -> None (empty)
                # succeeded -> True
                # else -> False
                if self_.record.state == TrialState.skipped:
                    self_.record.feasibility = None

                elif self_.record.state == TrialState.succeeded:
                    self_.record.feasibility = True

                else:
                    self_.record.feasibility = False

                # append
                if exc_type is None:
                    row = self_.append()
                # 1st argument of issubclass cannot be None
                elif issubclass(exc_type, ExceptionDuringOptimization):
                    row = self_.append()

                # if append is succeeded,
                # do fem.post_processing
                if row is not None:
                    self_.postprocess_after_recording(row)

                # save history if no FEMOpt
                client = get_client(self._records.df_wrapper._scheduler_address)
                if client is None:
                    self.save()

        return RecordContext()

    def save(self):
        """Export the optimization history.

        The destination path is :class:`History.path`.
        """

        # flask server 情報のように、最適化の途中で
        # 書き換えられるケースがあるので
        # additional data を再度ここで meta_columns に反映する
        cm = self._records.column_manager
        for i, column in enumerate(cm.column_dtypes.keys()):
            # additional_data を入れる
            if column == cm._get_additional_data_column():
                cm.meta_columns[i] = json.dumps(self.additional_data or dict())

        self._records.save(self.path)

    def _create_optuna_study_for_visualization(self):
        """出力は internal ではない値で、objective は出力という意味であり cns, other_output を含む。"""

        import optuna

        # create study
        kwargs: dict[str, ...] = dict(
            # storage='sqlite:///' + os.path.basename(self.path) + '_dummy.db',
            sampler=None, pruner=None, study_name='dummy',
        )
        if len(self.all_output_names) == 1:
            kwargs.update(dict(direction='minimize'))
        else:
            kwargs.update(dict(directions=['minimize']*len(self.all_output_names)))
        study = optuna.create_study(**kwargs)

        # add trial to study
        df = self.get_df(equality_filters=MAIN_FILTER)

        for i, row in df.iterrows():

            # trial
            trial_kwargs: dict = dict()

            # state
            state_str = Record.get_state_str_from_series(row)
            if state_str != TrialState.succeeded:
                continue
            state = optuna.trial.TrialState.COMPLETE
            trial_kwargs.update(dict(state=state))

            # params
            params = {prm_name: row[prm_name] for prm_name in self.prm_names}
            trial_kwargs.update(dict(params=params))

            # distribution
            distributions: dict[str, optuna.distributions.BaseDistribution] = dict()
            for prm_name in params.keys():

                # float
                if self._records.column_manager.is_numerical_parameter(prm_name):

                    lb_name = CorrespondingColumnNameRuler.prm_lower_bound_name(prm_name)
                    ub_name = CorrespondingColumnNameRuler.prm_upper_bound_name(prm_name)

                    if np.isnan(row[lb_name]):
                        low = df[prm_name].dropna().values.min()
                    else:
                        low = row[lb_name]

                    if np.isnan(row[ub_name]):
                        high = df[prm_name].dropna().values.max()
                    else:
                        high = row[ub_name]

                    dist = optuna.distributions.FloatDistribution(
                        low=low, high=high
                    )

                # categorical
                elif self._records.column_manager.is_categorical_parameter(prm_name):
                    choices_name = CorrespondingColumnNameRuler.prm_choices_name(prm_name)
                    dist = optuna.distributions.CategoricalDistribution(
                        choices=row[choices_name]
                    )

                else:
                    raise NotImplementedError

                distributions.update(
                    {prm_name: dist}
                )
            trial_kwargs.update(dict(distributions=distributions))

            # objective (+ constraints + other_outputs as objective)
            if len(self.all_output_names) == 1:
                if len(self.obj_names) == 1:
                    trial_kwargs.update(dict(value=row[self.obj_names].values[0]))
                elif len(self.cns_names) == 1:
                    trial_kwargs.update(dict(value=row[self.cns_names].values[0]))
                elif len(self.other_output_names) == 1:
                    trial_kwargs.update(dict(value=row[self.other_output_names].values[0]))
                else:
                    assert False
            else:
                values = row[self.all_output_names].values
                trial_kwargs.update(dict(values=values))

            # add to study
            trial = optuna.create_trial(**trial_kwargs)
            study.add_trial(trial)

        return study

    def is_numerical_parameter(self, prm_name: str) -> bool:
        """:meta private:"""
        return self._records.column_manager.is_numerical_parameter(prm_name)

    def is_categorical_parameter(self, prm_name) -> bool:
        """:meta private:"""
        return self._records.column_manager.is_categorical_parameter(prm_name)
