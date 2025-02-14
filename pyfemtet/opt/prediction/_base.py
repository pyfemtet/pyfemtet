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
            self.meta_model.set_bounds_from_history(
                history,
                df,
            )

        self.obj_names = history.obj_names
        self.prm_names = history.prm_names
        self.df = df

    def get_prm_index(self, prm_name):
        return self.prm_names.index(prm_name) if prm_name in self.prm_names else None

    def get_obj_index(self, obj_name):
        return self.obj_names.index(obj_name) if obj_name in self.obj_names else None

    def fit(self) -> None:
        from pyfemtet.opt.prediction.single_task_gp import SingleTaskGPModel
        if isinstance(self.meta_model, SingleTaskGPModel):
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
