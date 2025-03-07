import csv
import datetime
import dataclasses
from contextlib import nullcontext

import numpy as np
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

    @property
    def lock_if_not_locked(self):
        if self.lock.locked():
            return nullcontext()
        else:
            return self.lock

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


class CorrespondingColumnNameRuler:
    @staticmethod
    def create_direction_name(obj_name):
        return obj_name + '_direction'


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
        self._update_optimality()

    @staticmethod
    def _calc_optimality(y_internal: np.ndarray, feasibility: np.ndarray = None) -> np.ndarray:
        """

        Args:
            y_internal (np.ndarray): n x m shaped 2d-array. Can contain np.nan. Minimum value is optimal.
            feasibility (np.ndarray): n shaped 1d-array. bool.

        Returns:
            np.ndarray: Array if not optimal, dominated or Nan False, else True

        """

        # verification
        feasibility = feasibility if feasibility is not None else np.ones(len(y_internal)).astype(bool)

        # 非劣解の計算
        non_dominated = [
            (not np.isnan(y).any())  # feasible (duplicated)
            and
            (not (y > y_internal).all(axis=1).any())  # not dominated
            and
            feas  # feasible
            for y, feas in zip(y_internal, feasibility)
        ]

        return np.array(non_dominated).astype(bool)

    def _update_optimality(self):
        """
        data = np.array([[0.6, 'minimize', 0.1, 'minimize'],
               [0.2, '0', 0.3, 'maximize'],
               [0.3, 'maximize', np.nan, np.nan],
               [np.nan, np.nan, 0.1, 'minimize'],
               [0.1, 'minimize', 0.6, 'minimize']], dtype=object)
        df = pd.DataFrame(data).astype({0: float, 1: object})
        all_obj_values = df.iloc[:, [0, 2]].values
        all_obj_directions = df.iloc[:, [1, 3]].values

        y_internal = np.empty(all_obj_values.shape)
        for i, (obj_values, obj_directions) in enumerate(zip(all_obj_values.T, all_obj_directions.T)):
            y_internal[:, i] = np.array(
                list(
                    map(
                        lambda args: Objective._convert(*args),
                        zip(obj_values, obj_directions)
                    )
                )
            )


        """

        with self.df_wrapper.lock_if_not_locked:

            # get df
            df = self.df_wrapper.get_df()

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

            # calc optimality
            optimality = self._calc_optimality(y_internal, feasibility)

            # update
            df['optimality'] = optimality
            self.df_wrapper.set_df(df)


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


