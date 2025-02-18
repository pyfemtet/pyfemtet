# typing
import datetime
from abc import ABC
from typing import Optional, TYPE_CHECKING, Callable

# built-in
import traceback
from time import sleep

# 3rd-party
import numpy as np

# pyfemtet relative
from pyfemtet.opt.interface import FEMInterface
from pyfemtet.opt._femopt_core import OptimizationStatus, Objective, Constraint
from pyfemtet._message import Msg
from pyfemtet.opt.optimizer.parameter import ExpressionEvaluator

# logger
from pyfemtet.logger import get_module_logger

if TYPE_CHECKING:
    from pyfemtet.opt._femopt import SubFidelityModels
    from pyfemtet.opt._femopt_core import History


logger = get_module_logger('opt.optimizer', __name__)


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
        self.sub_fidelity_models: 'SubFidelityModels' = None
        self._should_solve: Callable[['History'], bool] = lambda *args, **kwargs: True
        self.current_time_start = None

    def should_calc(self, *args) -> bool:
        return self._should_solve(*args)

    def set_solve_condition(self, fun: Callable[['np.ndarray', 'History'], bool]):
        self._should_solve = fun

    # ===== algorithm specific methods =====
    def run(self) -> None:
        """Start optimization."""
        raise NotImplementedError

    # ----- FEMOpt interfaces -----
    def _setup_before_parallel(self, *args, **kwargs):
        """Setup before parallel processes are launched."""
        pass

    # ===== calc =====
    def f(self, x: np.ndarray) -> tuple | None:
        """Calculate objectives and constraints.

        Args:
            x (np.ndarray): Optimization parameters.

        Returns:
            list[np.ndarray]:
                The list of internal objective values,
                un-normalized objective values and
                constraint values.

        """

        if isinstance(x, np.float64) or isinstance(x, float) or isinstance(x, int):
            x = np.array([x], dtype=float)

        # Optimizer の x の更新
        self.set_parameter_values(x)
        logger.info(f'input: {x}')

        # values to df
        df_to_fem = self.variables.get_variables(
            format='df',
            filter_pass_to_fem=True
        )

        # default values
        y, c = self.history.generate_hidden_infeasible_result()
        _y = y

        # shouldn't calc であっても record_infeasible で
        # この値を使うので if の前に時間計測開始
        self.current_time_start = datetime.datetime.now()

        # main FEM の更新
        state = self.history.OptTrialState.skipped.value
        if self.should_calc(x, self.history):
            state = self.history.OptTrialState.succeeded.value

            logger.info(f'Solve FEM...')
            try:
                self.fem.update(df_to_fem)
                y = [obj.calc(self.fem) for obj in self.objectives.values()]
                _y = [obj.convert(value) for obj, value in zip(self.objectives.values(), y)]
                c = [cns.calc(self.fem) for cns in self.constraints.values()]

            except Exception as e:
                logger.warning(f'{type(e).__name__}: {" ".join(e.args)}')
                logger.warning(Msg.INFO_EXCEPTION_DURING_FEM_ANALYSIS)
                logger.warning(x)
                raise e  # may be just a ModelError, etc. Handling them in Concrete classes.

            else:
                pass

        # sub-fidelity FEM の更新
        try:
            self.sub_fidelity_models.update(df_to_fem, x, self.history)
            sub_fid_y: dict[str, tuple['Fidelity', list[float]]] = self.sub_fidelity_models.calc_objectives(x, self.history)
            sub_fid_y_internal: dict[str, tuple['Fidelity', list[float]]] = self.sub_fidelity_models.get_internal_objectives(sub_fid_y)

        except Exception as e:
            logger.warning(f'{type(e).__name__}: {" ".join(e.args)}')
            logger.warning(Msg.INFO_EXCEPTION_DURING_FEM_ANALYSIS)
            logger.warning(x)
            # raise e  # may be just a ModelError, etc. Handling them in Concrete classes.
            sub_fid_y = {}
            sub_fid_y_internal = {}

        # register to history
        df_to_opt = self.variables.get_variables(
            format='df',
            filter_parameter=True,
        )
        self.history._record(
            df_to_opt,
            self.objectives,
            self.constraints,
            y,
            c,
            sub_fid_y,
            self.message,
            state=state,
            time_start=self.current_time_start,
            time_end=datetime.datetime.now(),
            postprocess_func=self.fem._postprocess_func,
            postprocess_args=self.fem._create_postprocess_args(),
        )

        return (
            np.array(y) if y is not None else y,
            np.array(_y) if y is not None else _y,
            np.array(c) if y is not None else c,
            sub_fid_y,
            sub_fid_y_internal,
        )

    def record_infeasible(self, x, state):
        y, c = self.history.generate_hidden_infeasible_result()
        # register to history
        self.set_parameter_values(x)
        df_to_opt = self.variables.get_variables(
            format='df',
            filter_parameter=True,
        )
        self.history._record(
            df_to_opt,
            self.objectives,
            self.constraints,
            y,
            c,
            {},
            self.message,
            state,
            self.current_time_start,
            datetime.datetime.now(),
            postprocess_func=self.fem._postprocess_func,
            postprocess_args=self.fem._create_postprocess_args(),
        )

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
        self.fem.quit()
        if not self.worker_status.get() == OptimizationStatus.CRASHED:
            self.worker_status.set(OptimizationStatus.TERMINATED)

    # run via FEMOpt (considering parallel processing)
    def _run(
            self,
            subprocess_idx,  # 自身が何番目の並列プロセスであるかを示す連番
            worker_status_list,  # 他の worker の status オブジェクト
            wait_setup,  # 他の worker の status が ready になるまで待つか
            skip_reconstruct=False,  # reconstruct fem を行うかどうか
            sub_fidelity_reconstructor: callable = None,
            space_dir=None,  # 特定の space_dir を使うかどうか
    ) -> Optional[Exception]:

        # 自分の worker_status の取得
        self.subprocess_idx = subprocess_idx
        self.worker_status = worker_status_list[subprocess_idx]
        self.worker_status.set(OptimizationStatus.LAUNCHING_FEM)

        if self._check_interruption():
            return None

        # set_fem をはじめ、終了したらそれを示す
        self._reconstruct_fem(skip_reconstruct)
        self.fem._setup_after_parallel(opt=self, space_dir=space_dir)
        if sub_fidelity_reconstructor is not None:
            self.sub_fidelity_models = sub_fidelity_reconstructor()
        self.sub_fidelity_models._reconstruct_fem(skip_reconstruct)
        self.sub_fidelity_models.setup_after_parallel(opt=self, space_dir=space_dir)
        self.worker_status.set(OptimizationStatus.WAIT_OTHER_WORKERS)

        # wait_setup or not
        if wait_setup:
            while True:
                if self._check_interruption():
                    return None
                # 他のすべての worker_status が wait 以上になったら break
                if all([ws.get() >= OptimizationStatus.WAIT_OTHER_WORKERS for ws in worker_status_list]):
                    # リソースの競合等を避けるため
                    # break する前に index 秒待つ
                    sleep(int(subprocess_idx))
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

    __opt = Optimizer()
    __opt.set_parameter(
        dict(
            prm1=0.,
            prm2=1.,
        )
    )
