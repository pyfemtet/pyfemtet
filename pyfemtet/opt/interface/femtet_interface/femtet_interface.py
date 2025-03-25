from __future__ import annotations

from typing import TYPE_CHECKING

import os
import sys
import warnings
from time import sleep

# noinspection PyUnresolvedReferences
from pywintypes import com_error, error
# noinspection PyUnresolvedReferences
from pythoncom import CoInitialize, CoUninitialize
# noinspection PyUnresolvedReferences
from win32com.client import constants
import win32con
import win32gui

from pyfemtet.logger import get_module_logger

from pyfemtet._i18n import Msg
from pyfemtet._util.dask_util import *
from pyfemtet._util.femtet_exit import *
from pyfemtet._util.process_util import *
from pyfemtet._util.femtet_version import *
from pyfemtet._util.femtet_autosave import *
from pyfemtet._util.femtet_access_inspection import *

from pyfemtet.dispatch_extensions import *
from pyfemtet.opt.interface.interface import COMInterface
from pyfemtet.opt.exceptions import *
from pyfemtet.opt.variable_manager import SupportedVariableTypes

from ._femtet_parametric import *

if TYPE_CHECKING:
    from pyfemtet.opt.optimizer import AbstractOptimizer

logger = get_module_logger('opt.interface', False)


def _post_activate_message(hwnd):
    win32gui.PostMessage(hwnd, win32con.WM_ACTIVATE, win32con.WA_ACTIVE, 0)


class FailedToPostProcess(Exception):
    pass


class FemtetInterface(COMInterface):
    """Control Femtet from optimizer.

    Args:
        femprj_path (str, optional):
            The path to the project file.
            If not specified, get it from connected Femtet.

        model_name (str, optional):
            The name of the model.
            If not specified, get it from connected Femtet or the
            first model when Femtet open the project file.

        connect_method (str, optional):
            The connection method.
            Default is 'auto'. Other valid values are 'new' or 'existing'.

        save_pdt (str, optional):
            The type to save result file.
            Can specify 'all' or None. Default is 'all'.

        strictly_pid_specify (bool, optional):
            Whether to strictly specify the PID in Femtet connection.
            Default is True.

        allow_without_project (bool, optional):
            Whether to allow without a project. Default is False.

        open_result_with_gui (bool, optional):
            Whether to open the result with GUI. Default is True.

        parametric_output_indexes_use_as_objective (dict[int, str or float], optional):
            A list of parametric output indexes and its direction
            to use as the objective function. If not specified,
            it will be None and no parametric outputs are used
            as objectives.


            Note:
                Indexes start at 0, but the parametric analysis
                output settings in the Femtet dialog box indicate
                setting numbers starting at 1.


            Warning:
                **Setting this argument deletes the parametric
                analysis swept table set in the femprj file.**
                If you do not want to delete the swept table,
                make a copy of the original file.

        **kwargs: Additional arguments from inherited classes.

    Warning:
        Even if you specify ``strictly_pid_specify=True`` on the constructor,
        **the connection behavior is like** ``strictly_pid_specify=False`` **in parallel processing**
        because of its large overhead.
        So you should close all Femtet processes before running FEMOpt.optimize()
        if ``n_parallel`` >= 2.

    Tip:
        If you search for information about the method to
        connect python and Femtet, see :func:`connect_femtet`.

    """

    com_members = {'Femtet': 'FemtetMacro.Femtet'}

    def __init__(
        self,
        femprj_path: str = None,
        model_name: str = None,
        connect_method: str = "auto",  # dask worker では __init__ の中で 'new' にするので super() の引数にしない。（しても意味がない）
        save_pdt: str = "all",  # 'all' or None
        strictly_pid_specify: bool = True,  # dask worker では True にしたいので super() の引数にしない。
        allow_without_project: bool = False,  # main でのみ True を許容したいので super() の引数にしない。
        open_result_with_gui: bool = True,
        parametric_output_indexes_use_as_objective: dict[int, str or float] = None,  # TODO: Remove this
    ):
        # warning
        if parametric_output_indexes_use_as_objective is not None:
            warnings.warn(
                "解析モデルに設定された既存のスイープテーブルは削除されます。"
            )

        # 引数の処理
        if femprj_path is None:
            self.femprj_path = None
        else:
            self.femprj_path = os.path.abspath(femprj_path)
        self.model_name = model_name
        self.connect_method = connect_method
        self.allow_without_project = allow_without_project
        self.original_femprj_path = self.femprj_path
        self.open_result_with_gui = open_result_with_gui
        self.save_pdt = save_pdt

        # その他のメンバーの宣言や初期化
        self.Femtet = None
        self.femtet_pid = 0
        self.quit_when_destruct = False
        self.connected_method = "unconnected"
        self.max_api_retry = 3
        self.strictly_pid_specify = strictly_pid_specify
        self.parametric_output_indexes_use_as_objective = parametric_output_indexes_use_as_objective
        self._load_problem_from_fem = self.parametric_output_indexes_use_as_objective is not None
        self._original_autosave_enabled = _get_autosave_enabled()
        _set_autosave_enabled(False)

        # connect to Femtet
        self._connect_and_open_femtet()

        # 接続した Femtet の種類に応じて del 時に quit するかどうか決める
        self.quit_when_destruct = self.connected_method == "new"

    # ===== system =====

    @property
    def _object_pass_to_fun(self):
        return self.Femtet

    def _setup_before_parallel(self):
        self._distribute_files([self.femprj_path])

    def _setup_after_parallel(self, _: AbstractOptimizer = None):
        # worker space の femprj を開く

        worker = get_worker()

        if worker is not None:

            CoInitialize()

            self.femprj_path = os.path.join(
                worker.local_directory,
                os.path.basename(self.original_femprj_path)
            )
            self.connect_femtet(connect_method='new')
            self.open(self.femprj_path, self.model_name)
            self.quit_when_destruct = True

    def close(self, timeout=1, force=True):
        """Force to terminate connected Femtet."""

        _set_autosave_enabled(self._original_autosave_enabled)

        if self.quit_when_destruct:
            _exit_or_force_terminate(timeout=timeout, Femtet=self.Femtet, force=True)

    def use_parametric_output_as_objective(
            self, number: int, direction: str | float = "minimize"
    ) -> None:
        """Use output setting of Femtet parametric analysis as an objective function.

        Args:
            number (int): The index of output settings tab in parametric analysis dialog of Femtet. Starts at 1.
            direction (str | float): Objective direction.
                Valid input is one of 'minimize', 'maximize' or a specific value.
                Defaults to 'minimize'.

        Returns:
            None

        """
        # check
        if isinstance(direction, str):
            if direction not in ("minimize", "maximize"):
                raise ValueError(
                    f'direction must be one of "minimize", "maximize" or a specific value. Passed value is {direction}'
                )
        else:
            try:
                direction = float(direction)
            except (TypeError, ValueError):
                raise ValueError(
                    f'direction must be one of "minimize", "maximize" or a specific value. Passed value is {direction}'
                )

        index = {number - 1: direction}

        if self.parametric_output_indexes_use_as_objective is None:
            self.parametric_output_indexes_use_as_objective = index

        else:
            self.parametric_output_indexes_use_as_objective.update(index)

        self._load_problem_from_fem = True

    def load_objectives(self, opt: AbstractOptimizer):
        indexes = list(self.parametric_output_indexes_use_as_objective.keys())
        directions = list(self.parametric_output_indexes_use_as_objective.values())
        add_parametric_results_as_objectives(
            opt, self.Femtet, indexes, directions
        )

    def _check_using_fem(self, fun: callable):
        return _is_access_femtet(fun)

    # ===== connect_femtet =====

    def _connect_new_femtet(self):
        logger.info("└ Try to launch and connect new Femtet process.")

        self.Femtet, self.femtet_pid = launch_and_dispatch_femtet(
            strictly_pid_specify=self.strictly_pid_specify
        )

        self.connected_method = "new"

    def _connect_existing_femtet(self, pid: int or None = None):
        logger.info("└ Try to connect existing Femtet process.")
        # 既存の Femtet を探して Dispatch する。
        if pid is None:
            self.Femtet, self.femtet_pid = dispatch_femtet(timeout=5)
        else:
            self.Femtet, self.femtet_pid = dispatch_specific_femtet(pid, timeout=5)
        self.connected_method = "existing"

    def connect_femtet(self, connect_method: str = "auto", pid: int or None = None):
        """Connects to a Femtet process.

        Args:
            connect_method (str, optional): The connection method.
                Can be 'new', 'existing', or 'auto'. Defaults to 'auto'.
            pid (int or None, optional): The process ID of an existing Femtet process and wanted to connect.

        Note:
            When connect_method is 'new', starts a new Femtet process and connects to it.
            `pid` will be ignored.

        Note:
            When 'existing', connect to an existing Femtet process.
            However, if there are no Femtets to which it can connect
            (i.e. already connected to another Python or Excel process),
            it throws an exception.

        Note:
            When set to 'auto', first tries 'existing', and if that fails, connects with 'new'.
            If `pid` is specified and failed to connect,
            it will not try existing another Femtet process.

        """

        if connect_method == "new":
            self._connect_new_femtet()

        elif connect_method == "existing":
            self._connect_existing_femtet(pid)

        elif connect_method == "auto":
            try:
                self._connect_existing_femtet(pid)
            except DispatchExtensionException:
                self._connect_new_femtet()

        else:
            raise Exception(f"{connect_method} は定義されていない接続方法です")

        # ensure makepy
        if not hasattr(constants, "STATIC_C"):
            cmd = f"{sys.executable} -m win32com.client.makepy FemtetMacro"
            os.system(cmd)
            message = Msg.ERR_NO_MAKEPY
            logger.error("================")
            logger.error(message)
            logger.error("================")
            raise RuntimeError(message)

        if self.Femtet is None:
            raise RuntimeError(Msg.ERR_FEMTET_CONNECTION_FAILED)

    def open(self, femprj_path: str, model_name: str or None = None) -> None:
        """Open specific analysis model with connected Femtet."""

        # 引数の処理
        self.femprj_path = os.path.abspath(femprj_path)
        self.model_name = model_name
        # 開く
        if self.model_name is None:
            result = self.Femtet.LoadProject(self.femprj_path, True)
        else:
            result = self.Femtet.LoadProjectAndAnalysisModel(
                self.femprj_path, self.model_name, True
            )
        if not result:
            self.Femtet.ShowLastError()

    def _connect_and_open_femtet(self):
        """Connects to a Femtet process and open the femprj.

        This function is for establishing a connection with Femtet and opening the specified femprj file.

        At the beginning of the function, we check if femprj_path is specified.

        If femprj_path is specified, first connect to Femtet using the specified connection method. Then, if the project path of the connected Femtet is different from the specified femprj_path, open the project using the open function. Also, if model_name is specified, the project will be opened using the open function even if the Femtet analysis model name is different from the specified model_name.

        On the other hand, if femprj_path is not specified, an error message will be displayed stating that femprj_path must be specified in order to use the "new" connection method. If the connection method is not "new", it will try to connect to an existing Femtet instance. If the connection is successful, the project path and analysis model name of the Femtet instance will be stored as femprj_path and model_name.

        """

        logger.info(f'Try to connect Femtet (method: "{self.connect_method}").')
        logger.info(
            f'│ femprj: {self.femprj_path if self.femprj_path is not None else "not specified."}'
        )
        logger.info(
            f'│ model: {self.model_name if self.femprj_path is not None else "not specified."}'
        )

        # femprj が指定されている
        if self.femprj_path is not None:
            # 指定された方法で接続してみて
            # 接続した Femtet と指定された femprj 及び model が異なれば
            # 接続した Femtet で指定された femprj 及び model を開く
            self.connect_femtet(self.connect_method)

            # プロジェクトの相違をチェック
            if self.Femtet.ProjectPath != self.femprj_path:
                self.open(self.femprj_path, self.model_name)

            # モデルが指定されていればその相違もチェック
            if self.model_name is not None:
                if self.Femtet.AnalysisModelName != self.model_name:
                    self.open(self.femprj_path, self.model_name)

        # femprj が指定されていない
        else:
            # かつ new だと解析すべき femprj がわからないのでエラー
            if (self.connect_method == "new") and (not self.allow_without_project):
                raise RuntimeError(Msg.ERR_NEW_FEMTET_BUT_NO_FEMPRJ)

            # さらに auto の場合は Femtet が存在しなければ new と同じ挙動になるので同様の処理
            if (
                (self.connect_method == "auto")
                and (len(_get_pids(process_name="Femtet.exe")) == 0)
                and (not self.allow_without_project)
            ):
                raise RuntimeError(Msg.ERR_NEW_FEMTET_BUT_NO_FEMPRJ)
            self.connect_femtet(self.connect_method)

        # 最終的に接続した Femtet の femprj_path と model を インスタンスに戻す
        self.femprj_path = self.Femtet.Project
        self.model_name = self.Femtet.AnalysisModelName

    # ===== call femtet API =====

    def _check_gaudi_accessible(self) -> bool:
        try:
            _ = self.Femtet.Gaudi
        except com_error:
            # モデルが開かれていないかFemtetが起動していない
            return False
        return True

    # noinspection PyMethodMayBeStatic
    def _construct_femtet_api(self, string):  # static にしてはいけない
        if isinstance(string, str):
            if string.startswith("self."):
                return eval(string)
            else:
                return eval("self." + string)
        else:
            return string  # Callable

    def _call_femtet_api(
        self,
        fun,
        return_value_if_failed,
        if_error,
        error_message,
        is_Gaudi_method=False,
        ret_for_check_idx=None,
        args=None,
        kwargs=None,
        recourse_depth=0,
        print_indent=0,
    ):
        """Internal method. Call Femtet API with error handling.

        Parameters
        ----------
        fun : Callable or str
            Femtet API
        return_value_if_failed : Any
            API が失敗した時の戻り値
        if_error : Type
            エラーが発生していたときに送出したい Exception
        error_message : str
            上記 Exception に記載したいメッセージ
        is_Gaudi_method : bool
            API 実行前に Femtet.Gaudi.Activate() を行うか
        ret_for_check_idx : int or None
            API の戻り値が配列の時, 失敗したかを判定するために
            チェックすべき値のインデックス.
            デフォルト：None. None の場合, API の戻り値そのものをチェックする.
        args
            API の引数
        kwargs
            API の名前付き引数
        recourse_depth
            そのメソッドを再起動してリトライしている回数
        print_indent
            デバッグ用.
        Returns
        -------

        """

        if args is None:
            args = tuple()
        if kwargs is None:
            kwargs = dict()

        # 処理の失敗には 2 パターンある.
        # 1. 結果に関わらず戻り値が None で API 実行時に com_error を送出する
        # 2. API 実行時に成功失敗を示す戻り値を返し、ShowLastError で例外にアクセスできる状態になる

        # 実行する API をデバッグ出力
        if isinstance(fun, str):
            logger.debug(
                " " * print_indent + f"Femtet API:{fun}, args:{args}, kwargs:{kwargs}"
            )
        else:
            logger.debug(
                " " * print_indent
                + f"Femtet API:{fun.__name__}, args:{args}, kwargs:{kwargs}"
            )

        # Gaudi コマンドなら Gaudi.Activate する
        if (
            is_Gaudi_method
        ):  # Optimizer は Gogh に触らないので全部にこれをつけてもいい気がする
            try:
                # まず Gaudi にアクセスできるか
                gaudi_accessible = self._check_gaudi_accessible()
                if gaudi_accessible:
                    # Gaudi にアクセスできるなら Gaudi を Activate する
                    fun = self._construct_femtet_api(fun)  # (str) -> Callable
                    if fun.__name__ != "Activate":
                        # 再帰ループにならないように
                        self._call_femtet_api(
                            self.Femtet.Gaudi.Activate,
                            False,  # None 以外なら何でもいい
                            Exception,
                            "Gaudi のオープンに失敗しました",
                            print_indent=print_indent + 1,
                        )

                else:
                    # Gaudi にアクセスできないならば次の API 実行でエラーになる
                    pass

            except com_error:
                pass

        # API を実行
        try:
            # gaudi のメソッドかどうかにかかわらず、gaudi へのアクセスでエラーが出るか
            if not self._check_gaudi_accessible():
                raise com_error

            # gaudi_accessible なので関数が何であろうが安全にアクセスはできる
            if isinstance(fun, str):
                fun = self._construct_femtet_api(fun)  # (str) -> Callable

            # 解析結果を開いた状態で Gaudi.Activate して ReExecute する場合、ReExecute の前後にアクティブ化イベントが必要
            # さらに、プロジェクトツリーが開いていないとアクティブ化イベントも意味がないらしい。
            if fun.__name__ == "ReExecute":
                if (
                    self.open_result_with_gui
                    or self.parametric_output_indexes_use_as_objective
                ):
                    _post_activate_message(self.Femtet.hWnd)
                # API を実行
                returns = fun(*args, **kwargs)  # can raise pywintypes.error
                if (
                    self.open_result_with_gui
                    or self.parametric_output_indexes_use_as_objective
                ):
                    _post_activate_message(self.Femtet.hWnd)
            else:
                returns = fun(*args, **kwargs)

        # API の実行に失敗
        except (com_error, error):
            # 後続の処理でエラー判定されるように returns を作る
            # com_error ではなく error の場合はおそらく Femtet が落ちている
            if ret_for_check_idx is None:
                returns = return_value_if_failed
            else:
                returns = [return_value_if_failed] * (ret_for_check_idx + 1)
        logger.debug(" " * print_indent + f"Femtet API result:{returns}")

        # チェックすべき値の抽出
        if ret_for_check_idx is None:
            ret_for_check = returns
        else:
            ret_for_check = returns[ret_for_check_idx]

        # エラーのない場合は戻り値を return する
        if ret_for_check != return_value_if_failed:
            return returns

        # エラーがある場合は Femtet の生死をチェックし,
        # 死んでいるなら再起動してコマンドを実行させる.
        else:
            # そもそもまだ生きているかチェックする
            if self.femtet_is_alive():
                # 生きていてもここにきているなら
                # 指定された Exception を送出する
                logger.debug(" " * print_indent + error_message)
                raise if_error(error_message)

            # 死んでいるなら再起動
            else:
                # 再起動試行回数の上限に達していたら諦める
                logger.debug(
                    " " * print_indent + f"現在の Femtet 再起動回数: {recourse_depth}"
                )
                if recourse_depth >= self.max_api_retry:
                    raise Exception(Msg.ERR_FEMTET_CRASHED_AND_RESTART_FAILED)

                # 再起動
                logger.warning(
                    " " * print_indent + Msg.WARN_FEMTET_CRASHED_AND_TRY_RESTART
                )
                CoInitialize()
                self.connect_femtet(connect_method="new")
                self.open(self.femprj_path, self.model_name)

                # 状態を復元するために一度変数を渡して解析を行う（fun.__name__がSolveなら2度手間だが）
                logger.info(" " * print_indent + Msg.INFO_FEMTET_CRASHED_AND_RESTARTED)

                self.update_parameter(self.current_prm_values)
                self.update()

                # 与えられた API の再帰的再試行
                return self._call_femtet_api(
                    fun,
                    return_value_if_failed,
                    if_error,
                    error_message,
                    is_Gaudi_method,
                    ret_for_check_idx,
                    args,
                    kwargs,
                    recourse_depth + 1,
                    print_indent + 1,
                )

    def femtet_is_alive(self) -> bool:
        """Returns connected femtet process is existing or not."""

        try:
            hwnd = self.Femtet.hWnd
        except (com_error, AttributeError):
            return False

        if hwnd == 0:
            return False

        pid = _get_pid(hwnd)

        return pid > 0

    # ===== model check and solve =====

    def _check_param_and_raise(self, param_name) -> None:
        """Check param_name is set in femprj file or not.

        Note:
            This function works with Femtet version 2023.1.1 and above.
            Otherwise, no check is performed.

        """
        major, minor, bugfix = 2023, 1, 1
        if self._version() >= _version(major, minor, bugfix):
            try:
                variable_names = self.Femtet.GetVariableNames_py()
            except AttributeError as e:
                logger.error("================")
                logger.error(Msg.ERR_CANNOT_ACCESS_API + "GetVariableNames_py")
                logger.error(Msg.CERTIFY_MACRO_VERSION)
                logger.error("================")
                raise e

            if variable_names is not None:
                if param_name in variable_names:
                    return self.Femtet.GetVariableValue(param_name)
            message = Msg.ERR_NO_SUCH_PARAMETER_IN_FEMTET
            logger.error("================")
            logger.error(message)
            logger.error(f"`{param_name}` not in {variable_names}")
            logger.error("================")
            raise RuntimeError(message)
        else:
            return None

    def update_parameter(self, x: dict[str, SupportedVariableTypes], with_warning=False) -> None | list[str]:
        """Update parameter of femprj."""
        COMInterface.update_parameter(self, x)

        # Gaudi.Activate()
        sleep(0.1)  # Gaudi がおかしくなる時がある対策
        self._call_femtet_api(
            "self.Femtet.Gaudi.Activate",
            True,  # 戻り値を持たないのでここは無意味で None 以外なら何でもいい
            Exception,  # 生きてるのに開けない場合
            error_message=Msg.NO_ANALYSIS_MODEL_IS_OPEN,
        )

        # Version check
        major, minor, bugfix = 2023, 1, 1

        # 変数一覧を取得する関数がある
        if self._version() >= _version(major, minor, bugfix):

            # Femtet で定義されている変数の取得
            # （2023.1.1 以降でマクロだけ古い場合は
            # check_param_value で引っかかっている
            # はずなのでここは AttributeError を
            # チェックしない）
            existing_variable_names = self._call_femtet_api(
                fun=self.Femtet.GetVariableNames_py,
                return_value_if_failed=False,  # 意味がない
                if_error=ModelError,  # 生きてるのに失敗した場合
                error_message=f"GetVariableNames_py failed.",
                is_Gaudi_method=True,
            )

            # 変数を含まないプロジェクトである場合
            if existing_variable_names is None:
                if with_warning:
                    return [Msg.FEMTET_ANALYSIS_MODEL_WITH_NO_PARAMETER]
                else:
                    return None

            # 変数を更新
            warning_messages = []
            for name, value in self.current_prm_values.items():

                # 渡された変数がちゃんと Femtet に定義されている
                if name in existing_variable_names:

                    # Femtet.UpdateVariable
                    self._call_femtet_api(
                        fun=self.Femtet.UpdateVariable,
                        return_value_if_failed=False,
                        if_error=ModelError,  # 生きてるのに失敗した場合
                        error_message=Msg.ERR_FAILED_TO_UPDATE_VARIABLE
                        + f"{value} -> {name}",
                        is_Gaudi_method=True,
                        args=(name, value),
                    )

                # 渡された変数が Femtet で定義されていない
                else:
                    msg = (
                        f"{name} not in {self.model_name}: "
                        + Msg.WARN_IGNORE_PARAMETER_NOT_CONTAINED
                    )
                    warning_messages.append(msg)
                    logger.warning(msg)

        # 変数一覧を取得する関数がない（チェックしない）
        else:

            # update without parameter check
            warning_messages = []
            for name, value in self.current_prm_values.items():
                self._call_femtet_api(
                    fun=self.Femtet.UpdateVariable,
                    return_value_if_failed=False,
                    if_error=ModelError,  # 生きてるのに失敗した場合
                    error_message=Msg.ERR_FAILED_TO_UPDATE_VARIABLE
                    + f"{value} -> {name}",
                    is_Gaudi_method=True,
                    args=(name, value),
                )

        # ここでは ReExecute しない
        if with_warning:
            return warning_messages
        else:
            return None

    def reconstruct_model(self) -> None:
        """Updates the analysis model only."""

        # 設計変数に従ってモデルを再構築
        self._call_femtet_api(
            "self.Femtet.Gaudi.ReExecute",
            False,
            ModelError,  # 生きてるのに失敗した場合
            error_message=Msg.ERR_RE_EXECUTE_MODEL_FAILED,
            is_Gaudi_method=True,
        )

        # 処理を確定
        self._call_femtet_api(
            self.Femtet.Redraw,
            False,  # 戻り値は常に None なのでこの変数に意味はなく None 以外なら何でもいい
            ModelError,  # 生きてるのに失敗した場合
            error_message=Msg.ERR_MODEL_REDRAW_FAILED,
            is_Gaudi_method=True,
        )

    def solve(self) -> None:
        """Execute FEM analysis."""

        # メッシュを切る
        self._call_femtet_api(
            "self.Femtet.Gaudi.Mesh",
            0,
            MeshError,
            Msg.ERR_MODEL_MESH_FAILED,
            is_Gaudi_method=True,
        )

        if self.parametric_output_indexes_use_as_objective is not None:

            # PyFemtet で保存させる pdt パスを決定する
            pdt_path = self.Femtet.ResultFilePath + ".pdt"

            # 前のものが残っているとややこしいので消しておく
            if os.path.exists(pdt_path):
                os.remove(pdt_path)

            # parametric analysis 経由で解析
            self._call_femtet_api(
                fun=solve_via_parametric_dll,
                return_value_if_failed=False,
                if_error=SolveError,
                error_message=Msg.ERR_PARAMETRIC_SOLVE_FAILED,
                is_Gaudi_method=True,
                args=(self.Femtet,),
            )

            # parametric analysis の場合
            # ダイアログで「解析結果を保存する」に
            # チェックがついていないと次にすべき
            # OpenCurrentResult に失敗するので
            # parametric の場合も pdt を保存する
            self._call_femtet_api(
                fun=self.Femtet.SavePDT,
                args=(pdt_path, True),
                return_value_if_failed=False,
                if_error=SolveError,
                error_message=Msg.ERR_FAILED_TO_SAVE_PDT,
                is_Gaudi_method=False,
            )

        else:
            # ソルブする
            self._call_femtet_api(
                self.Femtet.Solve,
                False,
                SolveError,
                Msg.ERR_SOLVE_FAILED,
                is_Gaudi_method=True,
            )

        # 次に呼ばれるはずのユーザー定義コスト関数の
        # 記述を簡単にするため先に解析結果を開いておく
        self._call_femtet_api(
            self.Femtet.OpenCurrentResult,
            False,
            SolveError,  # 生きてるのに開けない場合
            error_message=Msg.ERR_OPEN_RESULT_FAILED,
            is_Gaudi_method=True,
            args=(self.open_result_with_gui,),
        )

    def preprocess(self, Femtet):
        """A method called just before :func:`solve`.

        This method is called just before solve.
        By inheriting from the Interface class
        and overriding this method, it is possible
        to perform any desired preprocessing after
        the model update and before solving.
        """
        pass

    def postprocess(self, Femtet):
        """A method called just after :func:`solve`.

        This method is called just after solve.
        By inheriting from the Interface class
        and overriding this method, it is possible
        to perform any desired postprocessing after
        the solve and before evaluating objectives.
        """
        pass

    def update(self) -> None:
        """See :func:`FEMInterface.update`"""
        self.reconstruct_model()
        self.preprocess(self.Femtet)
        self.solve()
        self.postprocess(self.Femtet)

    # ===== postprocess after recording =====

    def _create_postprocess_args(self):
        try:
            file_content = self._create_result_file_content()
        except FailedToPostProcess:
            file_content = None

        try:
            jpg_content = self._create_jpg_content()
        except FailedToPostProcess:
            jpg_content = None

        out = dict(
            original_femprj_path=self.original_femprj_path,
            model_name=self.model_name,
            pdt_file_content=file_content,
            jpg_file_content=jpg_content,
        )
        return out

    @staticmethod
    def _create_path(femprj_path, model_name, trial_name, ext):
        result_dir = femprj_path.replace(".femprj", ".Results")
        pdt_path = os.path.join(result_dir, model_name + f"_{trial_name}.{ext}")
        return pdt_path

    # noinspection PyMethodOverriding
    @staticmethod
    def _postprocess_after_recording(
        dask_scheduler,  # must for run_on_scheduler
        trial_name: str,
        *,
        original_femprj_path: str,
        model_name: str,
        pdt_file_content=None,
        jpg_file_content=None,
    ):

        if pdt_file_content is not None:
            pdt_path = FemtetInterface._create_path(
                original_femprj_path, model_name, trial_name, ext='pdt')
            with open(pdt_path, "wb") as f:
                f.write(pdt_file_content)

        if jpg_file_content is not None:
            jpg_path = FemtetInterface._create_path(
                original_femprj_path, model_name, trial_name, ext='jpg')
            with open(jpg_path, "wb") as f:
                f.write(jpg_file_content)

    def _create_result_file_content(self):
        """Called after solve"""
        if self.save_pdt == "all":
            # save to worker space
            result_dir = self.femprj_path.replace(".femprj", ".Results")
            pdt_path = os.path.join(result_dir, self.model_name + ".pdt")

            self._call_femtet_api(
                fun=self.Femtet.SavePDT,
                args=(pdt_path, True),
                return_value_if_failed=False,
                if_error=FailedToPostProcess,
                error_message=Msg.ERR_FAILED_TO_SAVE_PDT,
                is_Gaudi_method=False,
            )

            # convert .pdt to ByteIO and return it
            with open(pdt_path, "rb") as f:
                content = f.read()
            return content

        else:
            return None

    def _create_jpg_content(self):
        result_dir = self.femprj_path.replace(".femprj", ".Results")
        jpg_path = os.path.join(result_dir, self.model_name + ".jpg")

        # モデル表示画面の設定
        self.Femtet.SetWindowSize(600, 600)
        self.Femtet.Fit()

        # ---モデルの画面を保存---
        self.Femtet.Redraw()  # 再描画
        succeed = self.Femtet.SavePicture(jpg_path, 600, 600, 80)

        self.Femtet.RedrawMode = True  # 逐一の描画をオン

        if not succeed:
            raise FailedToPostProcess(Msg.ERR_FAILED_TO_SAVE_JPG)

        if not os.path.exists(jpg_path):
            raise FailedToPostProcess(Msg.ERR_JPG_NOT_FOUND)

        with open(jpg_path, "rb") as f:
            content = f.read()

        return content

    def _version(self):
        return _version(Femtet=self.Femtet)
