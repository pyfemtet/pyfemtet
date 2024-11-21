# typing
from abc import ABC, abstractmethod
from typing import Optional

# built-in
import traceback
from time import sleep

# 3rd-party
import numpy as np

# pyfemtet relative
from pyfemtet.opt.interface import FEMInterface
from pyfemtet.opt._femopt_core import OptimizationStatus, Objective, Constraint
from pyfemtet._message import Msg
from pyfemtet.opt.optimizer.parameter import ExpressionEvaluator, Parameter

# logger
import logging
from pyfemtet.logger import get_logger
logger = get_logger('opt')
logger.setLevel(logging.INFO)


class OptimizationMethodChecker:
    """Check implementation of PyFemtet functions."""

    def __init__(self, opt):
        self.opt = opt

    def check_parallel(self, raise_error=True):
        function = 'parallel-processing'
        method = str(type(self.opt))
        message = (Msg.ERR_NOT_IMPLEMENTED
                   + f'method:{method}, function:{function}')
        if raise_error:
            raise NotImplementedError(message)
        else:
            logger.warning(message)

    def check_timeout(self, raise_error=True):
        function = 'timeout'
        method = str(type(self.opt))
        message = (Msg.ERR_NOT_IMPLEMENTED
                   + f'method:{method}, function:{function}')
        if raise_error:
            raise NotImplementedError(message)
        else:
            logger.warning(message)

    def check_multi_objective(self, raise_error=True):
        function = 'multi-objective'
        method = str(type(self.opt))
        message = (Msg.ERR_NOT_IMPLEMENTED
                   + f'method:{method}, function:{function}')
        if raise_error:
            raise NotImplementedError(message)
        else:
            logger.warning(message)

    def check_strict_constraint(self, raise_error=True):
        function = 'strict-constraint'
        method = str(type(self.opt))
        message = (Msg.ERR_NOT_IMPLEMENTED
                   + f'method:{method}, function:{function}')
        if raise_error:
            raise NotImplementedError(message)
        else:
            logger.warning(message)

    def check_constraint(self, raise_error=True):
        function = 'constraint'
        method = str(type(self.opt))
        message = (Msg.ERR_NOT_IMPLEMENTED
                   + f'method:{method}, function:{function}')
        if raise_error:
            raise NotImplementedError(message)
        else:
            logger.warning(message)

    def check_skip(self, raise_error=True):
        function = 'skip'
        method = str(type(self.opt))
        message = (Msg.ERR_NOT_IMPLEMENTED
                   + f'method:{method}, function:{function}')
        if raise_error:
            raise NotImplementedError(message)
        else:
            logger.warning(message)

    def check_seed(self, raise_error=True):
        function = 'random seed setting'
        method = str(type(self.opt))
        message = (Msg.ERR_NOT_IMPLEMENTED
                   + f'method:{method}, function:{function}')
        if raise_error:
            raise NotImplementedError(message)
        else:
            logger.warning(message)

    def check_incomplete_bounds(self, raise_error=True):
        function = 'optimize with no or incomplete bounds'
        method = str(type(self.opt))
        message = (Msg.ERR_NOT_IMPLEMENTED
                   + f'method:{method}, function:{function}')
        if raise_error:
            raise NotImplementedError(message)
        else:
            logger.warning(message)


class AbstractOptimizer(ABC):
    """Abstract base class for an interface of optimization library.

    Attributes:
        fem (FEMInterface): The finite element method object.
        fem_class (type): The class of the finite element method object.
        fem_kwargs (dict): The keyword arguments used to instantiate the finite element method object.
        variables (ExpressionEvaluator): The variables using optimization process including parameters.
        objectives (dict[str, Objective]): A dictionary containing the objective functions used in the optimization.
        constraints (dict[str, Constraint]): A dictionary containing the constraint functions used in the optimization.
        history (History): An actor object that records the history of each iteration in the optimization process.
        seed (int or None): The random seed used for random number generation during the optimization process.

    """

    def __init__(self):
        self.fem = None
        self.fem_class = None
        self.fem_kwargs = dict()
        self.variables: ExpressionEvaluator = ExpressionEvaluator()
        self.objectives: dict[str, Objective] = dict()
        self.constraints: dict[str, Constraint] = dict()
        self.entire_status = None  # actor
        self.history = None  # actor
        self.worker_status = None  # actor
        self.message = ''
        self.seed = None
        self.timeout = None
        self.n_trials = None
        self.is_cluster = False
        self.subprocess_idx = None
        self._exception = None
        self.method_checker: OptimizationMethodChecker = OptimizationMethodChecker(self)
        self._retry_counter = 0

    # ===== algorithm specific methods =====
    @abstractmethod
    def run(self) -> None:
        """Start optimization."""
        pass

    # ----- FEMOpt interfaces -----
    @abstractmethod
    def _setup_before_parallel(self, *args, **kwargs):
        """Setup before parallel processes are launched."""
        pass

    # ===== calc =====
    def f(self, x: np.ndarray) -> list[np.ndarray]:
        """Calculate objectives and constraints.

        Args:
            x (np.ndarray): Optimization parameters.

        Returns:
            list[np.ndarray]:
                The list of internal objective values,
                un-normalized objective values and
                constraint values.

        """


        if isinstance(x, np.float64):
            x = np.array([x])

        # Optimizer の x の更新
        self.set_parameter_values(x)
        logger.info(f'input: {x}')

        # FEM の更新
        try:
            logger.info(f'Solving FEM...')
            df_to_fem = self.variables.get_variables(
                format='df',
                filter_pass_to_fem=True
            )
            self.fem.update(df_to_fem)

        except Exception as e:
            logger.info(f'{type(e).__name__} : {e}')
            logger.info(Msg.INFO_EXCEPTION_DURING_FEM_ANALYSIS)
            logger.info(x)
            raise e  # may be just a ModelError, etc. Handling them in Concrete classes.

        # y, _y, c の更新
        y = [obj.calc(self.fem) for obj in self.objectives.values()]

        _y = [obj.convert(value) for obj, value in zip(self.objectives.values(), y)]

        c = [cns.calc(self.fem) for cns in self.constraints.values()]

        # register to history
        df_to_opt = self.variables.get_variables(
            format='df',
            filter_parameter=True,
        )
        self.history.record(
            df_to_opt,
            self.objectives,
            self.constraints,
            y,
            c,
            self.message,
            postprocess_func=self.fem._postprocess_func,
            postprocess_args=self.fem._create_postprocess_args(),
        )

        logger.info(f'output: {y}')

        return np.array(y), np.array(_y), np.array(c)

    # ===== parameter processing =====
    def get_parameter(self, format='dict'):
        """Returns the parameters in the specified format.

        Args:
            format (str, optional):
                The desired format of the parameters.
                Can be 'df' (DataFrame),
                'values' (np.ndarray),
                'dict' or
                'raw' (list of Variable object).
                Defaults to 'dict'.

        Returns:
            The parameters in the specified format.

        Raises:
            ValueError: If an invalid format is provided.

        """
        return self.variables.get_variables(format=format, filter_parameter=True)

    def set_parameter(self, params: dict[str, float]) -> None:
        """Update parameter.

        Args:
            params (dict):
                Key is the name of parameter and
                the value is the value of it.
                The partial set is available.

        """
        for name, value in params.items():
            self.variables.variables[name].value = value
        self.variables.evaluate()

    def set_parameter_values(self, values: np.ndarray) -> None:
        """Update parameter with values.

        Args:
            values (np.ndarray): Values of all parameters.

        """
        prm_names = self.variables.get_parameter_names()
        assert len(values) == len(prm_names)
        self.set_parameter({k: v for k, v in zip(prm_names, values)})

    # ===== FEMOpt interfaces =====
    def _reconstruct_fem(self, skip_reconstruct=False):
        """Reconstruct FEMInterface in a subprocess."""
        # reconstruct fem
        if not skip_reconstruct:
            self.fem = self.fem_class(**self.fem_kwargs)

        # COM 定数の restore
        for obj in self.objectives.values():
            obj._restore_constants()
        for cns in self.constraints.values():
            cns._restore_constants()

    def _check_interruption(self):
        """"""
        if self.entire_status.get() == OptimizationStatus.INTERRUPTING:
            self.worker_status.set(OptimizationStatus.INTERRUPTING)
            self._finalize()
            return True
        else:
            return False

    def _finalize(self):
        """Destruct fem and set worker status."""
        del self.fem
        if not self.worker_status.get() == OptimizationStatus.CRASHED:
            self.worker_status.set(OptimizationStatus.TERMINATED)

    # run via FEMOpt (considering parallel processing)
    def _run(
            self,
            subprocess_idx,
            worker_status_list,
            wait_setup,
            skip_set_fem=False,
    ) -> Optional[Exception]:

        # 自分の worker_status の取得
        self.subprocess_idx = subprocess_idx
        self.worker_status = worker_status_list[subprocess_idx]
        self.worker_status.set(OptimizationStatus.LAUNCHING_FEM)

        if self._check_interruption():
            return None

        # set_fem をはじめ、終了したらそれを示す
        if not skip_set_fem:  # なくても動く？？
            self._reconstruct_fem()
        self.fem._setup_after_parallel()
        self.worker_status.set(OptimizationStatus.WAIT_OTHER_WORKERS)

        # wait_setup or not
        if wait_setup:
            while True:
                if self._check_interruption():
                    return None
                # 他のすべての worker_status が wait 以上になったら break
                if all([ws.get() >= OptimizationStatus.WAIT_OTHER_WORKERS for ws in worker_status_list]):
                    break
                sleep(1)
        else:
            if self._check_interruption():
                return None

        # set status running
        if self.entire_status.get() < OptimizationStatus.RUNNING:
            self.entire_status.set(OptimizationStatus.RUNNING)
        self.worker_status.set(OptimizationStatus.RUNNING)

        # run and finalize
        try:
            self.run()
        except Exception as e:
            logger.error("=================================")
            logger.error("An unexpected error has occurred!")
            logger.error("=================================")
            logger.error(f'{type(e).__name__}: {e}')
            traceback.print_exc()
            self._exception = e
            self.worker_status.set(OptimizationStatus.CRASHED)
        finally:
            self._finalize()

        return self._exception


if __name__ == '__main__':
    class Optimizer(AbstractOptimizer):
        def run(self): pass
        def _setup_before_parallel(self, *args, **kwargs): pass

    opt = Optimizer()
    opt.set_parameter(
        dict(
            prm1=0.,
            prm2=1.,
        )
    )
