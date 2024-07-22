from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from pyfemtet.opt._femopt_core import History


class PredictionModelBase(ABC):

    @abstractmethod
    def fit(self, x: np.ndarray, y: np.ndarray):
        """
        Parameters:
            x (np.ndarray): Input. (Point number) rows and (variable number) columns.
            y (np.ndarray): Output. (Point number) rows and (objective number) columns.
        """

    @abstractmethod
    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Parameters:
            x (np.ndarray): Input. (Point number) rows and (variable number) columns.
        Returns:
            np.ndarray: (Point number) rows and (objective number) columns. Index 0 is mean and index 1 is std.
        """


class PyFemtetPredictionModel:

    def __init__(self, history: History, df: pd.DataFrame, MetaModel: type):
        assert issubclass(MetaModel, PredictionModelBase)
        self.meta_model: PredictionModelBase = MetaModel()
        self.obj_names = history.obj_names
        self.prm_names = history.prm_names
        self.df = df
        self.x: np.ndarray = df[self.prm_names].values
        self.y: np.ndarray = df[self.obj_names].values

    def get_prm_index(self, prm_name):
        return self.prm_names.index(prm_name) if prm_name in self.prm_names else None

    def get_obj_index(self, obj_name):
        return self.obj_names.index(obj_name) if obj_name in self.obj_names else None

    def fit(self) -> None:
        self.meta_model.fit(self.x, self.y)

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        assert len(x.shape) == 2
        assert x.shape[1] == len(self.prm_names)
        return self.meta_model.predict(x)
