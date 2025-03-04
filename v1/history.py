import os
import csv
import datetime
import dataclasses

import pandas as pd

from v1.problem import *
from v1.dask_util import *
from v1.str_enum import StrEnum
from v1.i18n import *

__all__ = [
    'TrialState',
    'History',
    'create_err_msg_from_exception',
]



def create_err_msg_from_exception(e: Exception):
    type(e).__name__ + ' / ' + ' '.join(map(str, e.args))


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

    def get_df(self):
        client = get_client()
        if client:
            if self._dataset_name in client.list_datasets():
                return client.get_dataset(self._dataset_name)
            else:
                raise RuntimeError
        else:
            return self.__df

    def set_df(self, df):
        client = get_client()
        if client:
            if self._dataset_name in client.list_datasets():
                client.unpublish_dataset(self._dataset_name)
            client.publish_dataset(**dict(df=df))
        self.__df = df

    def start_dask(self):
        # Register the df initialized before dask context.
        self.set_df(self.__df)

    def end_dask(self):
        # Get back the df on dask to use the value outside
        # dask context.
        self.__df = self.get_df()


@dataclasses.dataclass
class Record:
    trial: int = None
    trial_id: int = None
    sub_sampling: SubSampling | None = None
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

    @classmethod
    def setup(cls, x_names, y_names, c_names):
        cls._set_full_sorted_column_information(x_names, y_names, c_names)

    @classmethod
    def _set_full_sorted_column_information(cls, x_names, y_names, c_names):
        dtypes = {}
        meta_columns = []

        # noinspection PyUnresolvedReferences
        keys = cls.__dataclass_fields__.copy().keys()
        for key in keys:
            if key == 'x':
                dtypes.update({name: float for name in x_names})
                meta_columns.extend(['prm'] * len(x_names))
            elif key == 'y':
                dtypes.update({name: float for name in y_names})
                meta_columns.extend(['obj'] * len(y_names))
            elif key == 'c':
                dtypes.update({name: float for name in c_names})
                meta_columns.extend(['cns'] * len(c_names))
            else:
                dtypes.update({key: object})
                meta_columns.append('')

        cls.dtypes = dtypes
        cls.meta_columns = meta_columns

    def as_df(self):
        assert hasattr(self, 'dtypes'), 'Record is not setup.'

        self._calc_feasibility()

        # noinspection PyUnresolvedReferences
        keys = self.__dataclass_fields__.copy().keys()
        d = {key: getattr(self, key) for key in keys if getattr(self, key) is not None}

        x = d.pop('x')
        y = d.pop('y')
        c = d.pop('c')

        d.update(**{k: v.value for k, v in x.items()})
        d.update(**{k: v.value for k, v in y.items()})
        d.update(**{k: v.value for k, v in c.items()})

        return pd.DataFrame(
            {k: [v] for k, v in d.items()},
            columns=tuple(self.dtypes.keys())
        ).astype(self.dtypes)

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

    @classmethod
    @property
    def prm_names(cls) -> list[str]:
        return cls._filter_columns('prm')

    @classmethod
    @property
    def obj_names(cls) -> list[str]:
        return cls._filter_columns('obj')

    @classmethod
    @property
    def cns_names(cls) -> list[str]:
        return cls._filter_columns('cns')


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

    def save(self, path: str):

        df = self.df_wrapper.get_df()

        with open(path, 'w', encoding=ENCODING) as f:
            writer = csv.writer(f, delimiter=',', lineterminator="\n")
            # write meta_columns
            writer.writerow(Record.meta_columns)
            writer.writerow([''] * len(Record.meta_columns))  # empty line
            # write df from line 3
            df.to_csv(f, index=None, encoding=ENCODING, lineterminator='\n')

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

    def setup(self, prm_names, obj_names, cns_names):
        Record.setup(prm_names, obj_names, cns_names)

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

        # OK なので読み込んだデータを set_df する
        self.df_wrapper.set_df(self.loaded_df)

    def append(self, record: Record):

        # get row
        row = record.as_df()

        # concat
        dfw = self.df_wrapper

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
    def get_df(self):
        return self._records.df_wrapper.get_df()

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

        # worker ごとに実行が必要
        self._records.setup(prm_names, obj_names, cns_names)

        # worker で再処理されると困る処理
        if not self._finalized:
            if self.path is None:
                self.path = datetime.datetime.now().strftime("pyfemtet.opt_%Y%m%d_%H%M%S.csv")

            self._records.check_problem_compatibility()

            self._finalized = True

    def record(
            self,
            datetime_start: datetime.datetime,
            x: TrialInput = None,
            y: TrialOutput = None,
            c: TrialConstraintOutput = None,
            state: TrialState = TrialState.undefined,
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
            x=x, y=y, c=c, state=state,
            datetime_start=datetime_start,
            datetime_end=datetime_end,
            message=message,
        )

        self._records.append(record)

    def save(self):
        self._records.save(self.path)


