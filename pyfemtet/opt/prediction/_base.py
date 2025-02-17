from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from pyfemtet.opt._femopt_core import History


class PredictionModelBase(ABC):
    """Simple Abstract surrogate model class."""

    @abstractmethod
    def fit(self, x: np.ndarray, y: np.ndarray):
        """
        Args:
            x (np.ndarray): Input. (Point number) rows and (variable number) columns.
            y (np.ndarray): Output. (Point number) rows and (objective number) columns.
        """

    @abstractmethod
    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Args:
            x (np.ndarray): Input. (Point number) rows and (variable number) columns.
        Returns:
            np.ndarray: (Point number) rows and (objective number) columns. Index 0 is mean and index 1 is std.
        """


class PyFemtetPredictionModel:

    def __init__(self, history: History, df: pd.DataFrame, MetaModel: type):
        assert issubclass(MetaModel, PredictionModelBase)
        self.meta_model: PredictionModelBase = MetaModel()

        from pyfemtet.opt.prediction.single_task_gp import SingleTaskGPModel
        if isinstance(self.meta_model, SingleTaskGPModel):
            bounds = self.create_bounds_from_history(history, df)
            self.meta_model._set_bounds(bounds)

        self.obj_names = history.obj_names
        self.prm_names = history.prm_names
        self.fidelity_column_names = [history.get_fidelity_column_name(name) for name in history.sub_fidelity_names]
        self.sub_fidelity_obj_names_list: list[list[str]] = \
            [history.get_obj_names_of_sub_fidelity(name) for name in history.sub_fidelity_names]
        self.df = df

    def get_prm_index(self, prm_name):
        return self.prm_names.index(prm_name) if prm_name in self.prm_names else None

    def get_obj_index(self, obj_name):
        return self.obj_names.index(obj_name) if obj_name in self.obj_names else None

    def fit(self) -> None:
        from pyfemtet.opt.prediction.single_task_gp import SingleTaskGPModel, SingleTaskMultiFidelityGPModel

        if isinstance(self.meta_model, SingleTaskMultiFidelityGPModel):
            feasible = (~np.isnan(self.df[self.obj_names].values)).prod(axis=1).astype(bool)
            main_y = self.df[feasible][self.obj_names].values
            main_x = self.df[feasible][self.prm_names].values
            main_x = np.concatenate(
                [
                    np.ones((len(main_x), 1)),
                    main_x
                ],
                axis=1,
            )
            for fidelity_column_name, obj_names_per_fidelity in \
                    zip(self.fidelity_column_names, self.sub_fidelity_obj_names_list):



                feasible = (~np.isnan(self.df[obj_names_per_fidelity].values)).prod(axis=1).astype(bool)
                pdf = self.df[feasible]

                prm_values_per_fidelity = pdf[self.prm_names].values
                obj_values_per_fidelity = pdf[obj_names_per_fidelity].values
                fidelity_values = pdf[fidelity_column_name].values

                part_x = np.concatenate(
                    [
                        fidelity_values.reshape(-1, 1),
                        prm_values_per_fidelity,

                    ],
                    axis=1,
                )

                main_x = np.concatenate(
                    [
                        main_x,
                        part_x,
                    ],
                    axis=0,
                )

                main_y = np.concatenate(
                    [
                        main_y,
                        obj_values_per_fidelity,
                    ],
                    axis=0,
                )

            x = main_x
            y = main_y
            self.meta_model.fit(x, y)

        elif isinstance(self.meta_model, SingleTaskGPModel):
            feasible = (~np.isnan(self.df[self.obj_names].values)).prod(axis=1).astype(bool)
            y = self.df[feasible][self.obj_names].values
            x = self.df[feasible][self.prm_names].values
            self.meta_model.fit(x, y)

        else:
            raise NotImplementedError

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        assert len(x.shape) == 2
        assert x.shape[1] == len(self.prm_names)
        return self.meta_model.predict(x)

    def create_bounds_from_history(self, history: History, df=None):
        meta_column: str

        if df is None:
            df = history.get_df()

        columns = df.columns

        target_columns = [
            col for col, meta_column in zip(columns, history.meta_columns)
            if meta_column == 'prm_lb' or meta_column == 'prm_ub'
        ]

        bounds_buff = df.iloc[0][target_columns].values  # 2*len(prm_names) array
        bounds = bounds_buff.reshape(-1, 2).astype(float)

        return bounds
