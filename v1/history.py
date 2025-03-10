from __future__ import annotations

import os
import csv
import datetime
import dataclasses
from contextlib import nullcontext

import numpy as np
import pandas as pd

from v1.i18n import *
from v1.helper import *
from v1.problem import *
from v1.dask_util import *
from v1.exceptions import *
from v1.optimality import *
from v1.hypervolume import *
from v1.str_enum import StrEnum


__all__ = [
    'TrialState',
    'History',
    'Record',
    'create_err_msg_from_exception',
]


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

        # dask クラスターがある場合
        client = get_client()
        if client:

            # datasets 内に存在する場合
            if self._dataset_name in client.list_datasets():
                df = client.get_dataset(self._dataset_name)

            # set の前に get されることはあってはならない
            else:
                raise RuntimeError

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
        if equality_filters is not None:
            partial_df = df
            df = self.get_df()
            apply_partial_df(df, partial_df, equality_filters)


        # dask クラスター上のデータを更新
        client = get_client()
        if client:

            # datasets 上に存在する場合は削除（上書きができない）
            if self._dataset_name in client.list_datasets():

                # remove
                client.unpublish_dataset(self._dataset_name)

            # update
            client.publish_dataset(**{self._dataset_name: df})

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
    def create_direction_name(obj_name):
        return obj_name + '_direction'


@dataclasses.dataclass
class Record:
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

    def as_df(self):
        assert hasattr(self, 'dtypes'), 'Record is not setup.'

        self._calc_feasibility()

        # noinspection PyUnresolvedReferences
        keys = self.__dataclass_fields__.copy().keys()
        d = {key: getattr(self, key) for key in keys if getattr(self, key) is not None}

        x: TrialInput = d.pop('x')
        y: TrialOutput = d.pop('y')
        c: TrialConstraintOutput = d.pop('c')

        d.update(**{k: v.value for k, v in x.items()})
        d.update(**{k: v.value for k, v in y.items()})
        d.update(**{f'{k}_direction': v.direction for k, v in y.items()})
        d.update(**{k: v.value for k, v in c.items()})

        return pd.DataFrame(
            {k: [v] for k, v in d.items()},
            columns=tuple(self.dtypes.keys())
        ).astype(self.dtypes)

    @classmethod
    def initialize(cls, x_names, y_names, c_names):
        cls._set_full_sorted_column_information(x_names, y_names, c_names)

    @classmethod
    def _set_full_sorted_column_information(cls, x_names, y_names, c_names):
        dtypes = {}
        meta_columns = []

        # noinspection PyUnresolvedReferences
        keys = cls.__dataclass_fields__.copy().keys()
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
                dtypes.update({name: float for name in x_names})
                meta_columns.extend(['prm'] * len(x_names))

            elif key == 'y':
                f = CorrespondingColumnNameRuler.create_direction_name
                for name in y_names:
                    dtypes.update({name: float})
                    dtypes.update({f(name): object})  # str | float
                meta_columns.extend(['obj', f('obj')] * len(y_names))

            elif key == 'c':
                dtypes.update({name: float for name in c_names})
                meta_columns.extend(['cns'] * len(c_names))

            else:
                dtypes.update({key: object})
                meta_columns.append('')

        cls.dtypes = dtypes
        cls.meta_columns = meta_columns

    @staticmethod
    def filter_columns(meta_column, columns, meta_columns) -> list[str]:
        out = []
        for i, column in enumerate(columns):
            if meta_columns[i] == meta_column:
                out.append(column)
        return out

    @classmethod
    def _filter_columns(cls, meta_column) -> list[str]:
        columns = list(cls.dtypes.keys())
        return cls.filter_columns(meta_column, columns, cls.meta_columns)

    # noinspection PyPropertyDefinition
    @classmethod
    @property
    def prm_names(cls) -> list[str]:
        return cls._filter_columns('prm')

    # noinspection PyPropertyDefinition
    @classmethod
    @property
    def obj_names(cls) -> list[str]:
        return cls._filter_columns('obj')

    # noinspection PyPropertyDefinition
    @classmethod
    @property
    def cns_names(cls) -> list[str]:
        return cls._filter_columns('cns')


class EntireDependentValuesManager:

    y_internal: np.ndarray
    feasibility: np.ndarray

    def __init__(self, records: Records, equality_filters: dict):

        self.records = records
        self.equality_filters = equality_filters

        assert self.records.df_wrapper.lock.locked()

        # get df
        df = self.get_df()

        # get column names
        obj_names = Record.obj_names
        f = CorrespondingColumnNameRuler.create_direction_name
        obj_direction_names = [f(name) for name in obj_names]

        # get values
        all_obj_values = df[obj_names].values
        all_obj_directions = df[obj_direction_names].values
        feasibility = df['feasibility']

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

        self.y_internal = y_internal
        self.feasibility = feasibility

    def get_df(self):
        return self.records.df_wrapper.get_df(
            self.equality_filters
        )

    def set_df(self, df):
        self.records.df_wrapper.set_df(
            df,
            self.equality_filters
        )

    def update_optimality(self):

        assert self.records.df_wrapper.lock.locked()

        # get df
        df = self.get_df()

        # calc optimality
        optimality = calc_optimality(
            self.y_internal,
            self.feasibility,
        )

        # update
        df.loc[:, 'optimality'] = optimality
        self.set_df(df)

    def update_hypervolume(self):

        assert self.records.df_wrapper.lock.locked()

        # get df
        df = self.get_df()

        # calc hypervolume
        hv_values = calc_hypervolume(
            self.y_internal,
            self.feasibility,
            ref_point='nadir-up-to-the-point',
        )

        # update
        df.loc[:, 'hypervolume'] = hv_values
        self.set_df(df)

    def update_trial_number(self):

        assert self.records.df_wrapper.lock.locked()

        # get df
        df = self.get_df()

        # calc trial
        trial_number = np.arange(len(df)).astype(int)

        # update
        df.loc[:, 'trial'] = trial_number
        self.set_df(df)


class Records:
    """最適化の試行全体の情報を格納するモデルクラス"""
    df_wrapper: DataFrameWrapper

    def __init__(self):
        self.df_wrapper = DataFrameWrapper(pd.DataFrame())
        self.loaded_meta_columns = None
        self.loaded_df = None

    def __str__(self):
        return self.df_wrapper.__str__()

    def __len__(self):
        return len(self.df_wrapper)

    def initialize(self, prm_names, obj_names, cns_names):

        with self.df_wrapper.lock:
            df = self.df_wrapper.get_df()

            # 新しく始まる場合に備えカラムを設定
            # load の場合はあとで上書きされる
            df = pd.DataFrame([], columns=list(Record.dtypes.keys()))
            self.df_wrapper.set_df(df)

    def load(self, path: str):

        with open(path, 'r', encoding=ENCODING, newline='\n') as f:
            reader = csv.reader(f, delimiter=',')
            # load meta_column
            loaded_meta_columns = reader.__next__()
            reader.__next__()  # empty line
            # load df from line 3
            loaded_df = pd.read_csv(f, encoding=ENCODING, header=0)

        # この段階では Record が setup されていない可能性があるので
        # compatibility check をしない。よって set_df しない。
        self.loaded_meta_columns = loaded_meta_columns
        self.loaded_df = loaded_df

        # 文字列として扱いたいカラムに含まれる Nan を '' にする
        # 型は object にする（pandas は str 型を扱えない？）
        # 例:
        #   pd.DataFrame(dict(a=["123"])).astype(dict(a=str)).dtypes
        #   # dtype: object
        self.loaded_df['sub_fidelity_name'] = \
            self.loaded_df['sub_fidelity_name'].fillna('')


    def check_problem_compatibility(self):

        # 読み込んだデータがないのであれば何もしない
        if self.loaded_df is None:
            return

        # 順番が違ってもいいが、
        # 構成に変更がないこと。
        # ただし obj は減っていてもいい。
        loaded_columns, loaded_meta_columns = self.loaded_df.columns, self.loaded_meta_columns

        # prm_names が過不足ないか
        loaded_prm_names = set(Record.filter_columns('prm', loaded_columns, loaded_meta_columns))
        prm_names = set(Record.prm_names)
        if not (len(loaded_prm_names - prm_names) == len(prm_names - loaded_prm_names) == 0):
            raise RuntimeError('Incompatible parameter setting.')

        # obj_names が増えていないか
        loaded_obj_names = set(Record.filter_columns('obj', loaded_columns, loaded_meta_columns))
        obj_names = set(Record.obj_names)
        if len(obj_names - loaded_obj_names) > 0:
            raise RuntimeError('Incompatible objective setting.')

        # cns_names が過不足ないか
        # TODO: cns の上下限は変更されてはならない。
        loaded_cns_names = set(Record.filter_columns('cns', loaded_columns, loaded_meta_columns))
        cns_names = set(Record.cns_names)
        if not (len(loaded_cns_names - cns_names) == len(cns_names - loaded_cns_names) == 0):
            raise RuntimeError('Incompatible constraint setting.')

        # dtypes を設定
        # 与える dtypes のほうが多い場合
        # エラーになるので余分なものを削除
        # 与える dtypes が少ない分には
        # (pandas としては) 問題ない
        dtypes = {k: v for k, v in Record.dtypes.items() if k in self.loaded_df.columns}
        self.loaded_df = self.loaded_df.astype(dtypes)

        # OK なので読み込んだデータを set_df する
        self.df_wrapper.set_df(self.loaded_df)

    def save(self, path: str):

        df = self.df_wrapper.get_df()

        with open(path, 'w', encoding=ENCODING) as f:
            writer = csv.writer(f, delimiter=',', lineterminator="\n")
            # write meta_columns
            writer.writerow(Record.meta_columns)
            writer.writerow([''] * len(Record.meta_columns))  # empty line
            # write df from line 3
            df.to_csv(f, index=None, encoding=ENCODING, lineterminator='\n')

    def append(self, record: Record):

        # get row
        row = record.as_df()

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

            dfw.set_df(new_df)

            # calc entire-dependent values
            # must be in with block to keep
            # the entire data compatibility
            # during processing.
            self.update_entire_dependent_values()

    def update_entire_dependent_values(self):

        with self.df_wrapper.lock_if_not_locked:

            # update main fidelity
            mgr = EntireDependentValuesManager(
                self,
                {'sub_fidelity_name': MAIN_FIDELITY_NAME}
            )
            mgr.update_optimality()
            mgr.update_hypervolume()
            mgr.update_trial_number()

            # update sub fidelity
            entire_df = self.df_wrapper.get_df()
            sub_fidelity_names: list = np.unique(entire_df['sub_fidelity_name']).tolist()
            if MAIN_FIDELITY_NAME in sub_fidelity_names:
                sub_fidelity_names.remove(MAIN_FIDELITY_NAME)
            for sub_fidelity_name in sub_fidelity_names:
                mgr = EntireDependentValuesManager(
                    self,
                    {'sub_fidelity_name': sub_fidelity_name}
                )
                mgr.update_trial_number()


class History:
    """最適化の試行全体の情報を操作するルールクラス"""
    _records: Records
    prm_names: list[str]
    obj_names: list[str]
    cns_names: list[str]
    sub_fidelity_model_names: list[str]

    path: str

    def __init__(self):
        self._records = Records()
        self.path: str = None
        self._finalized: bool = False
        # self.current_trial_time_start: datetime.datetime = None

    def __str__(self):
        return self._records.__str__()

    def __enter__(self):
        self._records.df_wrapper.start_dask()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._records.df_wrapper.end_dask()

    # def _trial_start(self):
    #     self.current_trial_time_start = datetime.datetime.now()
    #
    # def _trial_end(self):
    #     self.current_trial_time_start = None
    #
    # def trial_recording(self):
    #
    #     class TrialContext:
    #
    #         # noinspection PyMethodParameters
    #         def __enter__(_self):
    #             self._trial_start()
    #
    #         # noinspection PyMethodParameters
    #         def __exit__(_self, exc_type, exc_val, exc_tb):
    #             self._trial_end()
    #
    #     return TrialContext()
    #
    def get_df(self, equality_filters: dict = None):
        return self._records.df_wrapper.get_df(equality_filters)

    def load_csv(self, path):
        self.path = path
        self._records.load(self.path)

    def finalize(
            self,
            prm_names,
            obj_names,
            cns_names,
            sub_fidelity_model_names=None,
    ):

        self.prm_names = list(prm_names)
        self.obj_names = list(obj_names)
        self.cns_names = list(cns_names)
        self.sub_fidelity_model_names = sub_fidelity_model_names or []

        # worker ごとに実行が必要な処理
        # ここで dtypes が決定する
        Record.initialize(prm_names, obj_names, cns_names)

        # worker で再処理されると困る処理
        if not self._finalized:

            # initialize
            self._records.initialize(prm_names, obj_names, cns_names)

            # load
            if self.path is None:
                self.path = datetime.datetime.now().strftime("pyfemtet.opt_%Y%m%d_%H%M%S.csv")
            if os.path.isfile(self.path):
                self.load_csv(self.path)

            # check
            self._records.check_problem_compatibility()

            self._finalized = True

    def record(
            self,
            datetime_start: datetime.datetime,
            x: TrialInput = None,
            y: TrialOutput = None,
            c: TrialConstraintOutput = None,
            state: TrialState = TrialState.undefined,
            sub_fidelity_name: str = None,
            fidelity: Fidelity | None = None,
            trial_id: int = None,
            sub_sampling: SubSampling | None = None,
            datetime_end: datetime.datetime | None = None,
            message: str = '',
    ):

        x = x or TrialInput()
        y = y or TrialOutput()
        c = c or TrialConstraintOutput()

        datetime_end = datetime_end if datetime_end is not None else datetime.datetime.now()

        record = Record(
            trial_id=trial_id,
            sub_sampling=sub_sampling,
            fidelity=fidelity,
            sub_fidelity_name=sub_fidelity_name,
            x=x, y=y, c=c, state=state,
            datetime_start=datetime_start,
            datetime_end=datetime_end,
            message=message,
        )

        self._records.append(record)

    def record2(self):

        # noinspection PyMethodParameters
        class RecordContext:

            def __init__(self_):
                self_.record = Record()

            def __enter__(self_):
                return self_.record

            def __exit__(self_, exc_type, exc_val, exc_tb):

                self_.record.datetime_end = self_.record.datetime_end \
                    if self_.record.datetime_end is not None \
                    else datetime.datetime.now()

                self._records.append(self_.record)

        return RecordContext()


    def save(self):
        self._records.save(self.path)


