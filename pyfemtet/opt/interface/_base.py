from typing import Optional, List

import os
from abc import ABC, abstractmethod

import pandas as pd

from pyfemtet.logger import get_module_logger

logger = get_module_logger('opt.interface', __name__)

here, me = os.path.split(__file__)


class FEMInterface(ABC):
    """Abstract base class for the interface with FEM software.

    Stores information necessary to restore FEMInterface instance in a subprocess.

    The concrete class should call super().__init__() with the desired arguments when restoring.

    Args:
        **kwargs: keyword arguments for FEMInterface (re)constructor.

    """

    kwargs = None

    def __init__(
            self,
            **kwargs
    ):
        # restore のための情報保管
        if self.kwargs is None:
            self.kwargs = kwargs
        else:
            self.kwargs.update(kwargs)

    @abstractmethod
    def update(self, parameters: pd.DataFrame) -> None:
        """Updates the FEM analysis based on the proposed parameters."""
        raise NotImplementedError('update() must be implemented.')

    @property
    def object_passed_to_functions(self):
        return self

    def check_param_value(self, param_name) -> float or None:
        """Checks the value of a parameter in the FEM model (if implemented in concrete class)."""
        pass

    def update_parameter(self, parameters: pd.DataFrame, with_warning=False) -> Optional[List[str]]:
        """Updates only FEM variables (if implemented in concrete class).

        If this method is implemented,
        it is able to get parameter via FEMInterface.

        """
        pass

    def load_parameter(self, opt) -> None:  # opt: AbstractOptimizer
        pass

    def load_objective(self, opt) -> None:  # opt: AbstractOptimizer
        pass

    def load_constraint(self, opt) -> None:  # opt: AbstractOptimizer
        pass

    def _setup_before_parallel(self, client) -> None:
        """Preprocessing before launching a dask worker (if implemented in concrete class).

        Args:
            client: dask client.
            i.e. you can update associated files by
            `client.upload_file(file_path)`
            The file will be saved to dask-scratch-space directory
            without any directory structure.

        """
        pass

    def _setup_after_parallel(self, *args, **kwargs):
        """Preprocessing after launching a dask worker and before run optimization (if implemented in concrete class)."""
        pass

    def _postprocess_func(self, trial: int, *args, dask_scheduler=None, **kwargs):
        pass

    def _create_postprocess_args(self):
        pass

    def quit(self):
        pass


class NoFEM(FEMInterface):
    """Dummy interface without FEM. Intended for debugging purposes."""

    def update(self, parameters: pd.DataFrame) -> None:
        pass
