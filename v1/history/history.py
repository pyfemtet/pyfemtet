from __future__ import annotations

import os
import csv
import ast
import datetime
import dataclasses
from time import sleep
from contextlib import nullcontext

import numpy as np
import pandas as pd

from v1.i18n import *
from v1.utils.helper import *
from v1.problem import *
from v1.utils.dask_util import *
from v1.exceptions import *
from v1.history.optimality import *
from v1.history.hypervolume import *
from v1.utils.str_enum import StrEnum
from v1.variable_manager import *
from v1.logger import get_module_logger

logger = get_module_logger('opt.history', True)


__all__ = [
    'TrialState',
    'History',
    'Record',
    'create_err_msg_from_exception',
    'CorrespondingColumnNameRuler',
    'MAIN_FILTER',
]

MAIN_FILTER: dict = {'sub_fidelity_name': MAIN_FIDELITY_NAME}


def create_err_msg_from_exception(e: Exception):
    return type(e).__name__ + ' / ' + ' '.join(map(str, e.args))


class TrialState(StrEnum):

    succeeded = 'Success'
    skipped = 'Skip'
    hard_constraint_violation = 'Hard constraint violation'
    soft_constraint_violation = 'Soft constraint violation'

    model_error = 'Model error'
    mesh_error = 'Mesh error'
    solve_error = 'Solve error'
    post_error = 'Post-processing error'

    unknown_error = 'Unknown error'
    undefined = 'undefined'

    @staticmethod
    def get_corresponding_state_from_exception(e: Exception) -> TrialState:
        if isinstance(e, ModelError):
            state = TrialState.model_error
        elif isinstance(e, MeshError):
            state = TrialState.mesh_error
        elif isinstance(e, SolveError):
            state = TrialState.solve_error
        elif isinstance(e, PostProcessError):
            state = TrialState.post_error
        else:
            state = TrialState.unknown_error
        return state

    @staticmethod
    def get_corresponding_exception_from_state(state: TrialState) -> Exception | None:
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
        else:
            e = None
        return e

    @classmethod
    def get_hidden_constraint_violation_states(cls):  # utility function for user
        return [cls.get_corresponding_state_from_exception(exception_type())
                for exception_type in HiddenConstraintViolation.__subclasses__]


class DataFrameWrapper:
    __df: pd.DataFrame
    _lock_name = 'edit-df'
    _dataset_name = 'df'

    def __init__(self, df: pd.DataFrame):
        self.set_df(df)

    def __len__(self):
        return len(self.get_df())

    def __str__(self):
        return self.get_df().__str__()

    @property
    def lock(self):
        return Lock(self._lock_name)

    @property
    def lock_if_not_locked(self):
        if self.lock.locked():
            return nullcontext()
        else:
            return self.lock

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

        client = get_client()

        # dask クラスターがある場合
        if client is not None:

            # あるけど with を抜けている場合
            if client.scheduler is None:
                df = self.__df

            # 健在の場合
            else:
                with Lock('access_dataset_df'):

                    # mypy によるとこのスコープで
                    # 定義しないと定義漏れになる可能性がある
                    # そんなことないと思うけど将来の変更時に
                    # 警告を見落とさないようにするためここで定義
                    df = None

                    # datasets 内に存在する場合
                    if self._dataset_name in client.list_datasets():
                        df = client.get_dataset(self._dataset_name)

                    # set の前に get されることはあってはならない
                    else:
                        raise RuntimeError

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
            assert self.lock.locked(), 'set_df() with equality_filters must be called with locking.'
            partial_df = df
            df = self.get_df()
            apply_partial_df(df, partial_df, equality_filters)

        # dask クラスター上のデータを更新
        client = get_client()
        if client is not None:
            if client.scheduler is not None:
                with Lock('access_dataset_df'):

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


class ColumnManager:

    parameters: TrialInput
    y_names: list[str]
    c_names: list[str]
    dtypes: dict[str, type]
    meta_columns: list[str]

    @staticmethod
    def columns_to_keep_even_if_nan():
        return [
            'message',
        ]

    def initialize(self, parameters: TrialInput, y_names, c_names):
        self.parameters = parameters
        self.y_names = y_names
        self.c_names = c_names
        self.set_full_sorted_column_information()

    def set_full_sorted_column_information(
            self,
            extra_parameters: TrialInput = None,
            extra_y_names: list[str] = None,
            extra_c_names: list[str] = None,
    ):

        extra_parameters = extra_parameters or {}
        extra_y_names = extra_y_names or []
        extra_c_names = extra_c_names or []

        dtypes = {}
        meta_columns = []

        # noinspection PyUnresolvedReferences
        keys = Record.__dataclass_fields__.copy().keys()
        for key in keys:
            # Note:
            #   as_df() で空欄になりうるカラムには
            #   Nan や '' を許容する dtype を指定すること
            #   例えば、 trial に int を指定してはいけない
            #
            # Note:
            #   pandas は dtypes に str を受け付けない
            #   (object にキャストされる模様)

            if key == 'x':
                for prm_name in self.parameters.keys():

                    param = self.parameters[prm_name]

                    if isinstance(param, NumericParameter):
                        dtypes.update({prm_name: float})
                        meta_columns.append('prm')

                        f = CorrespondingColumnNameRuler.prm_lower_bound_name
                        dtypes.update({f(prm_name): float})
                        meta_columns.append('prm_lower_bound')

                        f = CorrespondingColumnNameRuler.prm_upper_bound_name
                        dtypes.update({f(prm_name): float})
                        meta_columns.append('prm_upper_bound')

                        f = CorrespondingColumnNameRuler.prm_step_name
                        dtypes.update({f(prm_name): float})
                        meta_columns.append('prm_step')

                    elif isinstance(param, CategoricalParameter):
                        dtypes.update({prm_name: object})
                        meta_columns.append('prm')

                        f = CorrespondingColumnNameRuler.prm_choices_name
                        dtypes.update({f(prm_name): object})
                        meta_columns.append('prm_choices')

                    else:
                        raise NotImplementedError

                for extra_prm_name, extra_param in extra_parameters.items():

                    if isinstance(extra_param, NumericParameter):
                        dtypes.update({extra_prm_name: float})
                        meta_columns.append('')

                        f = CorrespondingColumnNameRuler.prm_lower_bound_name
                        dtypes.update({f(extra_prm_name): object})
                        meta_columns.append('')

                        f = CorrespondingColumnNameRuler.prm_upper_bound_name
                        dtypes.update({f(extra_prm_name): object})
                        meta_columns.append('')

                    elif isinstance(extra_param, CategoricalParameter):
                        dtypes.update({extra_prm_name: object})
                        meta_columns.append('')

                        f = CorrespondingColumnNameRuler.prm_choices_name
                        dtypes.update({f(extra_prm_name): object})
                        meta_columns.append('')

                    else:
                        raise NotImplementedError

            elif key == 'y':
                f = CorrespondingColumnNameRuler.direction_name
                for name in self.y_names:
                    dtypes.update({name: float})
                    dtypes.update({f(name): object})  # str | float
                meta_columns.extend(['obj', f('obj')] * len(self.y_names))

                for name in extra_y_names:
                    dtypes.update({name: float})
                    dtypes.update({f(name): object})  # str | float
                meta_columns.extend(['', ''] * len(extra_y_names))

            elif key == 'c':
                dtypes.update({name: float for name in self.c_names})
                meta_columns.extend(['cns'] * len(self.c_names))

                dtypes.update({name: float for name in extra_c_names})
                meta_columns.extend([''] * len(extra_c_names))

            else:
                dtypes.update({key: object})
                meta_columns.append('')

        self.dtypes = dtypes
        self.meta_columns = meta_columns

    @staticmethod
    def _filter_columns(meta_column, columns, meta_columns) -> list[str]:
        out = []
        assert len(columns) == len(meta_columns), f'{len(columns)=} and {len(meta_columns)=}'

        for i, (column_, meta_column_) in enumerate(zip(columns, meta_columns)):
            if meta_column_ == meta_column:
                out.append(column_)
        return out

    def filter_columns(self, meta_column) -> list[str]:
        columns = list(self.dtypes.keys())
        return self._filter_columns(meta_column, columns, self.meta_columns)

    def get_prm_names(self) -> list[str]:
        return self.filter_columns('prm')

    def get_obj_names(self) -> list[str]:
        return self.filter_columns('obj')

    def get_cns_names(self) -> list[str]:
        return self.filter_columns('cns')

    @staticmethod
    def _is_numerical_parameter(prm_name, columns):
        prm_lb_name = CorrespondingColumnNameRuler.prm_lower_bound_name(prm_name)
        return prm_lb_name in columns

    @staticmethod
    def _is_categorical_parameter(prm_name, columns):
        prm_choices_name = CorrespondingColumnNameRuler.prm_choices_name(prm_name)
        return prm_choices_name in columns

    def is_numerical_parameter(self, prm_name) -> bool:
        return self._is_numerical_parameter(prm_name, tuple(self.dtypes.keys()))

    def is_categorical_parameter(self, prm_name) -> bool:
        return self._is_categorical_parameter(prm_name, tuple(self.dtypes.keys()))

    @staticmethod
    def _get_parameter(prm_name: str, df: pd.DataFrame) -> Parameter:
        if ColumnManager._is_numerical_parameter(prm_name, df.columns):
            out = NumericParameter()
            out.name = prm_name
            out.value = float(df[prm_name].dropna().values[-1])
            out.lower_bound = float(df[CorrespondingColumnNameRuler.prm_lower_bound_name(prm_name)].dropna().values[-1])
            out.upper_bound = float(df[CorrespondingColumnNameRuler.prm_upper_bound_name(prm_name)].dropna().values[-1])
            out.step = float(df[CorrespondingColumnNameRuler.prm_step_name(prm_name)].dropna().values[-1])

        elif ColumnManager._is_categorical_parameter(prm_name, df.columns):
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
            # list は csv を経由することで str になるので restore
            if meta_column == 'prm_choices':
                print(df[column])
                df[column] = [ast.literal_eval(d) for d in df[column]]

    @staticmethod
    def _get_sub_fidelity_names(df: pd.DataFrame) -> list[str]:

        if 'sub_fidelity_name' not in df.columns:
            return [MAIN_FIDELITY_NAME]

        else:
            return np.unique(df['sub_fidelity_name'].values).tolist()


@dataclasses.dataclass
class Record:
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
    state: TrialState = TrialState.undefined
    datetime_start: datetime.datetime = dataclasses.field(default_factory=datetime.datetime.now)
    datetime_end: datetime.datetime = dataclasses.field(default_factory=datetime.datetime.now)
    message: str = ''
    hypervolume: float | None = None
    feasibility: bool | None = None
    optimality: bool | None = None

    def _calc_feasibility(self):
        # skipped -> None (empty)
        # succeeded -> calc

        feasibility = True

        if self.state == TrialState.skipped:
            self.feasibility = None
            return

        for cns_result in self.c.values():
            v_dict = cns_result.calc_violation()
            if any([v > 0 for v in v_dict.values()]):
                feasibility = False

        self.feasibility = feasibility

    def as_df(self, dtypes: dict = None):

        self._calc_feasibility()

        # noinspection PyUnresolvedReferences
        keys = self.__dataclass_fields__.copy().keys()
        d = {key: getattr(self, key) for key in keys if getattr(self, key) is not None}

        x: TrialInput = d.pop('x')
        y: TrialOutput = d.pop('y')
        c: TrialConstraintOutput = d.pop('c')

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

        d.update(**{k: v.value for k, v in y.items()})
        d.update(**{f'{k}_direction': v.direction for k, v in y.items()})
        d.update(**{k: v.value for k, v in c.items()})

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

        assert self.records.df_wrapper.lock.locked()

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

        assert self.records.df_wrapper.lock.locked()

        # calc optimality
        optimality = calc_optimality(
            self.partial_y_internal,
            self.partial_feasibility,
        )

        # update
        self.partial_df.loc[:, 'optimality'] = optimality

    def update_hypervolume(self):

        assert self.records.df_wrapper.lock.locked()

        # calc hypervolume
        hv_values = calc_hypervolume(
            self.partial_y_internal,
            self.partial_feasibility,
            ref_point='nadir-up-to-the-point',
        )

        # update
        self.partial_df.loc[:, 'hypervolume'] = hv_values

    def update_trial_number(self):

        assert self.records.df_wrapper.lock.locked()

        # calc trial
        trial_number = np.arange(len(self.partial_df)).astype(int)

        # update
        self.partial_df.loc[:, 'trial'] = trial_number


class Records:
    """最適化の試行全体の情報を格納するモデルクラス"""
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
            df = pd.DataFrame([], columns=list(self.column_manager.dtypes.keys()))
            self.df_wrapper.set_df(df)

    def load(self, path: str):

        with open(path, 'r', encoding=ENCODING, newline='\n') as f:
            reader = csv.reader(f, delimiter=',')
            # load meta_column
            loaded_meta_columns = reader.__next__()
            reader.__next__()  # empty line
            # load df from line 3
            loaded_df = pd.read_csv(f, encoding=ENCODING, header=0)

        # choices は list だったものが str になるので型変換
        ColumnManager._reconvert_objects(loaded_df, loaded_meta_columns)

        # この段階では dtypes が setup されていない可能性があるので
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
        loaded_prm_names = set(self.column_manager._filter_columns('prm', loaded_columns, loaded_meta_columns))
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

    def reinitialize_record_with_loaded_data(self):

        # 読み込んだデータがないのであれば何もしない
        if self.loaded_df is None:
            return

        loaded_columns, loaded_meta_columns = self.loaded_df.columns, self.loaded_meta_columns
        loaded_prm_names = set(self.column_manager._filter_columns('prm', loaded_columns, loaded_meta_columns))
        loaded_obj_names = set(self.column_manager._filter_columns('obj', loaded_columns, loaded_meta_columns))
        loaded_cns_names = set(self.column_manager._filter_columns('cns', loaded_columns, loaded_meta_columns))

        # loaded df に存在するが Record に存在しないカラムを Record に追加
        extra_parameters = {}
        extra_y_names = []
        extra_c_names = []
        for l_col, l_meta in zip(loaded_columns, loaded_meta_columns):

            # 現在の Record に含まれないならば
            if l_col not in self.column_manager.dtypes.keys():

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

        self.column_manager.set_full_sorted_column_information(
            extra_parameters=extra_parameters,
            extra_y_names=extra_y_names,
            extra_c_names=extra_c_names,
        )

        # worker に影響しないように loaded_df のコピーを作成
        df: pd.DataFrame = self.loaded_df.copy()

        # loaded df に存在しないが Record に存在するカラムを追加
        for col in self.column_manager.dtypes.keys():
            if col not in df.columns:
                # column ごとの default 値を追加
                if col == 'sub_fidelity_name':
                    df[col] = MAIN_FIDELITY_NAME
                else:
                    df[col] = np.nan

        # dtypes を設定
        # 与える dtypes のほうが多い場合
        # エラーになるので余分なものを削除
        # 与える dtypes が少ない分には
        # (pandas としては) 問題ない
        dtypes = {k: v for k, v in self.column_manager.dtypes.items() if k in self.loaded_df.columns}
        df = df.astype(dtypes)

        # 並べ替え
        df = df[list(self.column_manager.dtypes.keys())].astype(self.column_manager.dtypes)

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

        with open(path, 'w', encoding=ENCODING) as f:
            writer = csv.writer(f, delimiter=',', lineterminator="\n")
            # write meta_columns
            writer.writerow(meta_columns)
            writer.writerow([''] * len(meta_columns))  # empty line
            # write df from line 3
            df.to_csv(f, index=False, encoding=ENCODING, lineterminator='\n')

    def append(self, record: Record):

        # get row
        row = record.as_df(dtypes=self.column_manager.dtypes)

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

    def update_entire_dependent_values(self, processing_df: pd.DataFrame):

        with self.df_wrapper.lock_if_not_locked:

            # update main fidelity
            equality_filters = MAIN_FILTER
            mgr = EntireDependentValuesCalculator(
                self,
                equality_filters,
                processing_df,
            )
            mgr.update_optimality()
            mgr.update_hypervolume()
            mgr.update_trial_number()
            pdf = mgr.partial_df
            apply_partial_df(df=processing_df, partial_df=pdf, equality_filters=equality_filters)

            # update sub fidelity
            entire_df = self.df_wrapper.get_df()
            sub_fidelity_names: list = np.unique(entire_df['sub_fidelity_name']).tolist()
            if MAIN_FIDELITY_NAME in sub_fidelity_names:
                sub_fidelity_names.remove(MAIN_FIDELITY_NAME)
            for sub_fidelity_name in sub_fidelity_names:
                equality_filters = {'sub_fidelity_name': sub_fidelity_name}
                mgr = EntireDependentValuesCalculator(
                    self,
                    equality_filters,
                    processing_df
                )
                mgr.update_trial_number()
            pdf = mgr.partial_df
            apply_partial_df(df=processing_df, partial_df=pdf, equality_filters=equality_filters)


class History:
    """最適化の試行全体の情報を操作するルールクラス"""
    _records: Records
    prm_names: list[str]
    obj_names: list[str]
    cns_names: list[str]
    sub_fidelity_names: list[str]
    is_restart: bool

    path: str

    def __init__(self):
        self._records = Records()
        self.path: str | None = None
        self._finalized: bool = False
        self.is_restart = False

    def __str__(self):
        return self._records.__str__()

    def __enter__(self):
        self._records.df_wrapper.start_dask()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._records.df_wrapper.end_dask()

    def load_csv(self, path, with_finalize=False):
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

            self.prm_names = ColumnManager._filter_columns('prm', df.columns, meta_columns)
            self.obj_names = ColumnManager._filter_columns('obj', df.columns, meta_columns)
            self.cns_names = ColumnManager._filter_columns('cns', df.columns, meta_columns)
            self.sub_fidelity_names = ColumnManager._get_sub_fidelity_names(df)

            parameters: TrialInput = {}
            for prm_name in self.prm_names:
                param = ColumnManager._get_parameter(prm_name, df)
                parameters.update({prm_name: param})

            self.finalize(
                parameters,
                self.obj_names,
                self.cns_names,
                self.sub_fidelity_names,
            )

    def finalize(
            self,
            parameters: TrialInput,
            obj_names,
            cns_names,
            sub_fidelity_names,
    ):

        self.prm_names = list(parameters.keys())
        self.obj_names = list(obj_names)
        self.cns_names = list(cns_names)
        self.sub_fidelity_names = list(sub_fidelity_names)

        if not self._finalized:
            # ここで dtypes が決定する
            self._records.column_manager.initialize(
                parameters, self.obj_names, self.cns_names
            )

            # initialize
            self._records.initialize()

            # load
            if self.path is None:
                self.path = datetime.datetime.now().strftime("pyfemtet.opt_%Y%m%d_%H%M%S.csv")
            if os.path.isfile(self.path):
                self.load_csv(self.path)

            # check
            self._records.check_problem_compatibility()

            # load 後だが worker ごとに実行が必要
            # Record のクラス変数 dtypes と meta_columns を再初期化するため
            self._records.reinitialize_record_with_loaded_data()

        self._finalized = True

    def get_df(self, equality_filters: dict = None):
        return self._records.df_wrapper.get_df(equality_filters)

    def recording(self):

        # noinspection PyMethodParameters
        class RecordContext:

            def __init__(self_):
                self_.record = Record()

            def __enter__(self_):
                return self_.record

            def append(self_):
                self_.record.datetime_end = self_.record.datetime_end \
                    if self_.record.datetime_end is not None \
                    else datetime.datetime.now()
                self._records.append(self_.record)

            def __exit__(self_, exc_type, exc_val, exc_tb):

                if exc_type is None:
                    self_.append()

                elif issubclass(exc_type, ExceptionDuringOptimization):
                    self_.append()

        return RecordContext()

    def save(self):
        self._records.save(self.path)

    def _create_optuna_study_for_visualization(self):
        import optuna

        # create study
        kwargs: dict[str, ...] = dict(
            # storage='sqlite:///' + os.path.basename(self.path) + '_dummy.db',
            sampler=None, pruner=None, study_name='dummy',
        )
        if len(self.obj_names) == 1:
            kwargs.update(dict(direction='minimize'))
        else:
            kwargs.update(dict(directions=['minimize']*len(self.obj_names)))
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
                    dist = optuna.distributions.FloatDistribution(
                        low=row[lb_name],
                        high=row[ub_name],
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

            # objective
            if len(self.obj_names) == 1:
                trial_kwargs.update(dict(value=row[self.obj_names].values[0]))
            else:
                trial_kwargs.update(dict(values=row[self.obj_names].values))

            # add to study
            trial = optuna.create_trial(**trial_kwargs)
            study.add_trial(trial)

        return study


def debug_standalone_history():
    history = History()
    history.load_csv(os.path.join(os.path.dirname(__file__), 'history_test.csv'), with_finalize=True)

    print(f'{history.prm_names=}')
    print(f'{history.obj_names=}')
    print(f'{history.cns_names=}')

    df = history.get_df()

    print(df)

    print(df[history.prm_names])

    df.to_csv(os.path.join(os.path.dirname(__file__), 'history_loaded.csv'))


if __name__ == '__main__':
    debug_standalone_history()
