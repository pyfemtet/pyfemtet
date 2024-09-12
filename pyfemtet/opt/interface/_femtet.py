from typing import Optional, List, Final

import os
import sys
from time import sleep, time
import winreg

import pandas as pd
import psutil
from dask.distributed import get_worker

# noinspection PyUnresolvedReferences
from pywintypes import com_error, error
# noinspection PyUnresolvedReferences
from pythoncom import CoInitialize, CoUninitialize
# noinspection PyUnresolvedReferences
from win32com.client import constants
import win32con
import win32gui
from femtetutils import util

from pyfemtet.core import (
    ModelError,
    MeshError,
    SolveError,
    _version,
)
from pyfemtet.dispatch_extensions import (
    dispatch_femtet,
    dispatch_specific_femtet,
    launch_and_dispatch_femtet,
    _get_pid,
    _get_pids,
    DispatchExtensionException,
)
from pyfemtet.opt.interface import FEMInterface, logger
from pyfemtet.message import Msg


def post_activate_message(hwnd):
    win32gui.PostMessage(hwnd, win32con.WM_ACTIVATE, win32con.WA_ACTIVE, 0)


class FemtetInterface(FEMInterface):
    """Concrete class for the interface with Femtet.

        Args:
            femprj_path (str or None, optional): The path to the .femprj file. Defaults to None.
            model_name (str or None, optional): The name of the analysis model. Defaults to None.
            connect_method (str, optional): The connection method to use. Can be 'new', 'existing', or 'auto'. Defaults to 'auto'.
            strictly_pid_specify (bool, optional): If True and connect_method=='new', search launched Femtet process strictly based on its process id.
            allow_without_project (bool, optional): Allow to launch Femtet with no project file. Default to False.
            open_result_with_gui (bool, optional): Open analysis result with Femtet GUI. Default to True.
            parametric_output_indexes_use_as_objective (dict, optional): Parametric output indexes and their directions which will be used as objective functions. Parametric output should be set on Femtet parametric analysis dialog. Note that output 'No.' in dialog is starts with 1, but this 'index' is starts with 0. The key is the 'index', the value is direction ('maximize', 'minimize' or float). Default to None.

        Warning:
            Even if you specify ``strictly_pid_specify=True`` on the constructor,
            **the connection behavior is like** ``strictly_pid_specify=False`` **in parallel processing**
            because of its large overhead.
            So you should close all Femtet processes before running FEMOpt.optimize()
            if ``n_parallel`` >= 2.

        Tip:
            If you search for information about the method to connect python and Femtet, see :func:`connect_femtet`.

    """

    def __init__(
            self,
            femprj_path=None,
            model_name=None,
            connect_method='auto',  # dask worker では __init__ の中で 'new' にするので super() の引数にしない。（しても意味がない）
            save_pdt='all',  # 'all' or None
            strictly_pid_specify=True,  # dask worker では True にしたいので super() の引数にしない。
            allow_without_project=False,  # main でのみ True を許容したいので super() の引数にしない。
            open_result_with_gui=True,
            parametric_output_indexes_use_as_objective=None,
            **kwargs  # 継承されたクラスからの引数
    ):

        # win32com の初期化
        CoInitialize()

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
        self.connected_method = 'unconnected'
        self.parameters = None
        self.max_api_retry = 3
        self.strictly_pid_specify = strictly_pid_specify
        self.parametric_output_indexes_use_as_objective = parametric_output_indexes_use_as_objective
        self._original_autosave_enabled = get_autosave_enabled()
        set_autosave_enabled(False)

        # dask サブプロセスのときは femprj を更新し connect_method を new にする
        try:
            worker = get_worker()
            space = worker.local_directory
            # worker なら femprj_path が None でないはず
            self.femprj_path = os.path.join(space, os.path.basename(self.femprj_path))
            self.connect_method = 'new'
            self.strictly_pid_specify = False
        except ValueError:  # get_worker に失敗した場合
            pass

        # femprj_path と model に基づいて Femtet を開き、
        # 開かれたモデルに応じて femprj_path と model を更新する
        self._connect_and_open_femtet()

        # original_fem_prj が None なら必ず
        # dask worker でないプロセスがオリジナルファイルを開いている
        if self.original_femprj_path is None:
            # dask worker でなければ original のはず
            try:
                _ = get_worker()
            except ValueError:
                self.original_femprj_path = self.femprj_path

        # 接続した Femtet の種類に応じて del 時に quit するかどうか決める
        self.quit_when_destruct = self.connected_method == 'new'

        # subprocess で restore するための情報保管
        # パスなどは connect_and_open_femtet での処理結果を反映し
        # メインで開いた解析モデルが確実に開かれるようにする
        super().__init__(
            femprj_path=self.femprj_path,
            model_name=self.model_name,
            open_result_with_gui=self.open_result_with_gui,
            parametric_output_indexes_use_as_objective=self.parametric_output_indexes_use_as_objective,
            save_pdt=self.save_pdt,
            **kwargs
        )

    def __del__(self):
        set_autosave_enabled(self._original_autosave_enabled)
        if self.quit_when_destruct:
            self.quit()
        # CoUninitialize()  # Win32 exception occurred releasing IUnknown at 0x0000022427692748

    def _connect_new_femtet(self):
        logger.info('└ Try to launch and connect new Femtet process.')

        self.Femtet, self.femtet_pid = launch_and_dispatch_femtet(strictly_pid_specify=self.strictly_pid_specify)

        self.connected_method = 'new'

    def _connect_existing_femtet(self, pid: int or None = None):
        logger.info('└ Try to connect existing Femtet process.')
        # 既存の Femtet を探して Dispatch する。
        if pid is None:
            self.Femtet, self.femtet_pid = dispatch_femtet(timeout=5)
        else:
            self.Femtet, self.femtet_pid = dispatch_specific_femtet(pid, timeout=5)
        self.connected_method = 'existing'

    def connect_femtet(self, connect_method: str = 'auto', pid: int or None = None):
        """Connects to a Femtet process.

        Args:
            connect_method (str, optional): The connection method. Can be 'new', 'existing', or 'auto'. Defaults to 'new'.
            pid (int or None, optional): The process ID of an existing Femtet process and wanted to connect.

        Note:
            When connect_method is 'new', starts a new Femtet process and connects to it.
            **`pid` will be ignored.**

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

        if connect_method == 'new':
            self._connect_new_femtet()

        elif connect_method == 'existing':
            self._connect_existing_femtet(pid)

        elif connect_method == 'auto':
            try:
                self._connect_existing_femtet(pid)
            except DispatchExtensionException:
                self._connect_new_femtet()

        else:
            raise Exception(f'{connect_method} は定義されていない接続方法です')

        # ensure makepy
        if not hasattr(constants, 'STATIC_C'):
            cmd = f'{sys.executable} -m win32com.client.makepy FemtetMacro'
            os.system(cmd)
            message = Msg.ERR_NO_MAKEPY
            print('================')
            print(message)
            print('================')
            input('Press enter to finish...')
            raise RuntimeError(message)

        if self.Femtet is None:
            raise RuntimeError(Msg.ERR_FEMTET_CONNECTION_FAILED)

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
            if string.startswith('self.'):
                return eval(string)
            else:
                return eval('self.' + string)
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

        # FIXME: Gaudi へのアクセスなど、self.Femtet.Gaudi.SomeFunc() のような場合、この関数を呼び出す前に Gaudi へのアクセスの時点で com_error が起こる
        # FIXME: => 文字列で渡して eval() すればよい。

        if args is None:
            args = tuple()
        if kwargs is None:
            kwargs = dict()

        # 処理の失敗には 2 パターンある.
        # 1. 結果に関わらず戻り値が None で API 実行時に com_error を送出する
        # 2. API 実行時に成功失敗を示す戻り値を返し、ShowLastError で例外にアクセスできる状態になる

        # 実行する API をデバッグ出力
        if isinstance(fun, str):
            logger.debug(' ' * print_indent + f'Femtet API:{fun}, args:{args}, kwargs:{kwargs}')
        else:
            logger.debug(' ' * print_indent + f'Femtet API:{fun.__name__}, args:{args}, kwargs:{kwargs}')

        # Gaudi コマンドなら Gaudi.Activate する
        if is_Gaudi_method:  # Optimizer は Gogh に触らないので全部にこれをつけてもいい気がする
            try:
                # まず Gaudi にアクセスできるか
                gaudi_accessible = self._check_gaudi_accessible()
                if gaudi_accessible:
                    # Gaudi にアクセスできるなら Gaudi を Activate する
                    fun = self._construct_femtet_api(fun)  # (str) -> Callable
                    if fun.__name__ != 'Activate':
                        # 再帰ループにならないように
                        self._call_femtet_api(
                            self.Femtet.Gaudi.Activate,
                            False,  # None 以外なら何でもいい
                            Exception,
                            'Gaudi のオープンに失敗しました',
                            print_indent=print_indent + 1
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
            if fun.__name__ == 'ReExecute':
                if self.open_result_with_gui or self.parametric_output_indexes_use_as_objective:
                    post_activate_message(self.Femtet.hWnd)
                # API を実行
                returns = fun(*args, **kwargs)  # can raise pywintypes.error
                if self.open_result_with_gui or self.parametric_output_indexes_use_as_objective:
                    post_activate_message(self.Femtet.hWnd)
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
        logger.debug(' ' * print_indent + f'Femtet API result:{returns}')

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
                logger.debug(' ' * print_indent + error_message)
                raise if_error(error_message)

            # 死んでいるなら再起動
            else:
                # 再起動試行回数の上限に達していたら諦める
                logger.debug(' ' * print_indent + f'現在の Femtet 再起動回数: {recourse_depth}')
                if recourse_depth >= self.max_api_retry:
                    raise Exception(Msg.ERR_FEMTET_CRASHED_AND_RESTART_FAILED)

                # 再起動
                logger.warn(' ' * print_indent + Msg.WARN_FEMTET_CRASHED_AND_TRY_RESTART)
                CoInitialize()
                self.connect_femtet(connect_method='new')
                self.open(self.femprj_path, self.model_name)

                # 状態を復元するために一度変数を渡して解析を行う（fun.__name__がSolveなら2度手間だが）
                logger.info(' ' * print_indent + Msg.INFO_FEMTET_CRASHED_AND_RESTARTED)
                self.update(self.parameters)

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
                    print_indent + 1
                )

    def femtet_is_alive(self) -> bool:
        """Returns connected femtet process is existing or not."""
        return _get_pid(self.Femtet.hWnd) > 0  # hWnd の値はすでに Femtet が終了している場合は 0

    def open(self, femprj_path: str, model_name: str or None = None) -> None:
        """Open specific analysis model with connected Femtet."""

        # 引数の処理
        self.femprj_path = os.path.abspath(femprj_path)
        self.model_name = model_name
        # 開く
        if self.model_name is None:
            result = self.Femtet.LoadProject(
                self.femprj_path,
                True
            )
        else:
            result = self.Femtet.LoadProjectAndAnalysisModel(
                self.femprj_path,
                self.model_name,
                True
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
        logger.info(f'│ femprj: {self.femprj_path if self.femprj_path is not None else "not specified."}')
        logger.info(f'│ model: {self.model_name if self.femprj_path is not None else "not specified."}')

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
            if (
                    (self.connect_method == 'new')
                    and (not self.allow_without_project)
            ):
                raise RuntimeError(Msg.ERR_NEW_FEMTET_BUT_NO_FEMPRJ)

            # さらに auto の場合は Femtet が存在しなければ new と同じ挙動になるので同様の処理
            if (
                    (self.connect_method == 'auto')
                    and (len(_get_pids(process_name='Femtet.exe')) == 0)
                    and (not self.allow_without_project)
            ):
                raise RuntimeError(Msg.ERR_NEW_FEMTET_BUT_NO_FEMPRJ)
            self.connect_femtet(self.connect_method)

        # 最終的に接続した Femtet の femprj_path と model を インスタンスに戻す
        self.femprj_path = self.Femtet.Project
        self.model_name = self.Femtet.AnalysisModelName

    def check_param_value(self, param_name):
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
                print('================')
                logger.error(Msg.ERR_CANNOT_ACCESS_API + 'GetVariableNames_py')
                logger.error(Msg.CERTIFY_MACRO_VERSION)
                print('================')
                input(Msg.ENTER_TO_QUIT)
                raise e
                
            if variable_names is not None:
                if param_name in variable_names:
                    return self.Femtet.GetVariableValue(param_name)
            message = Msg.ERR_NO_SUCH_PARAMETER_IN_FEMTET
            print('================')
            logger.error(message)
            logger.error(f'`{param_name}` not in {variable_names}')
            print('================')
            input(Msg.ENTER_TO_QUIT)
            raise RuntimeError(message)
        else:
            return None

    def update_parameter(self, parameters: 'pd.DataFrame', with_warning=False):
        """Update parameter of femprj."""
        self.parameters = parameters.copy()

        # 変数更新のための処理
        sleep(0.1)  # Gaudi がおかしくなる時がある対策
        self._call_femtet_api(
            'self.Femtet.Gaudi.Activate',
            True,  # 戻り値を持たないのでここは無意味で None 以外なら何でもいい
            Exception,  # 生きてるのに開けない場合
            error_message=Msg.NO_ANALYSIS_MODEL_IS_OPEN,
        )

        major, minor, bugfix = 2023, 1, 1
        if self._version() >= _version(major, minor, bugfix):
            # Femtet の設計変数の更新（2023.1.1 以降でマクロだけ古い場合はcheck_param_valueで引っかかっているはずなのでここはAttributeError をチェックしない）
            existing_variable_names = self._call_femtet_api(
                fun=self.Femtet.GetVariableNames_py,
                return_value_if_failed=False,  # 意味がない
                if_error=ModelError,  # 生きてるのに失敗した場合
                error_message=f'GetVariableNames_py failed.',
                is_Gaudi_method=True,
            )

            # 変数を含まないプロジェクトである場合
            if existing_variable_names is None:
                if with_warning:
                    return [Msg.FEMTET_ANALYSIS_MODEL_WITH_NO_PARAMETER]
                else:
                    return None

            # update
            warnings = []
            for i, row in parameters.iterrows():
                name = row['name']
                value = row['value']
                if name in existing_variable_names:
                    self._call_femtet_api(
                        fun=self.Femtet.UpdateVariable,
                        return_value_if_failed=False,
                        if_error=ModelError,  # 生きてるのに失敗した場合
                        error_message=Msg.ERR_FAILED_TO_UPDATE_VARIABLE + f'{value} -> {name}',
                        is_Gaudi_method=True,
                        args=(name, value),
                    )
                else:
                    msg = f'{name} not in {self.model_name}: ' + Msg.WARN_IGNORE_PARAMETER_NOT_CONTAINED
                    warnings.append(msg)
                    logger.warn(msg)

        else:
            # update without parameter check
            warnings = []
            for i, row in parameters.iterrows():
                name = row['name']
                value = row['value']
                self._call_femtet_api(
                    fun=self.Femtet.UpdateVariable,
                    return_value_if_failed=False,
                    if_error=ModelError,  # 生きてるのに失敗した場合
                    error_message=Msg.ERR_FAILED_TO_UPDATE_VARIABLE + f'{value} -> {name}',
                    is_Gaudi_method=True,
                    args=(name, value),
                )

        # ここでは ReExecute しない
        if with_warning:
            return warnings
        else:
            return None

    def update_model(self, parameters: 'pd.DataFrame', with_warning=False) -> Optional[List[str]]:
        """Updates the analysis model only."""

        self.parameters = parameters.copy()

        # 変数の更新
        warnings = self.update_parameter(parameters, with_warning)

        # 設計変数に従ってモデルを再構築
        self._call_femtet_api(
            'self.Femtet.Gaudi.ReExecute',
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

        if with_warning:
            return warnings or []

    def solve(self) -> None:
        """Execute FEM analysis."""
        # # メッシュを切る
        self._call_femtet_api(
            'self.Femtet.Gaudi.Mesh',
            0,
            MeshError,
            Msg.ERR_MODEL_MESH_FAILED,
            is_Gaudi_method=True,
        )

        if self.parametric_output_indexes_use_as_objective is not None:
            from pyfemtet.opt.interface._femtet_parametric import solve_via_parametric_dll
            self._call_femtet_api(
                fun=solve_via_parametric_dll,
                return_value_if_failed=False,
                if_error=SolveError,
                error_message=Msg.ERR_PARAMETRIC_SOLVE_FAILED,
                is_Gaudi_method=True,
                args=(self.Femtet,),
            )
        else:
            # # ソルブする
            self._call_femtet_api(
                self.Femtet.Solve,
                False,
                SolveError,
                Msg.ERR_SOLVE_FAILED,
                is_Gaudi_method=True,
            )

        # 次に呼ばれるはずのユーザー定義コスト関数の記述を簡単にするため先に解析結果を開いておく
        self._call_femtet_api(
            self.Femtet.OpenCurrentResult,
            False,
            SolveError,  # 生きてるのに開けない場合
            error_message=Msg.ERR_OPEN_RESULT_FAILED,
            is_Gaudi_method=True,
            args=(self.open_result_with_gui,),
        )

    def update(self, parameters: 'pd.DataFrame') -> None:
        """See :func:`FEMInterface.update`"""
        self.parameters = parameters.copy()
        self.update_model(parameters)
        self.preprocess(self.Femtet)
        self.solve()
        self.postprocess(self.Femtet)

    def preprocess(self, Femtet):
        pass

    def postprocess(self, Femtet):
        pass

    def quit(self, timeout=1, force=True):
        """Force to terminate connected Femtet."""
        major, minor, bugfix = 2024, 0, 1

        # すでに終了しているならば何もしない
        if not self.femtet_is_alive():
            return

        if self._version() >= _version(major, minor, bugfix):
            # gracefully termination method without save project available from 2024.0.1
            try:
                self.Femtet.Exit(True)
            except AttributeError as e:
                print('================')
                logger.error(Msg.ERR_CANNOT_ACCESS_API + 'Femtet.Exit()')
                logger.error(Msg.CERTIFY_MACRO_VERSION)
                print('================')
                input(Msg.ENTER_TO_QUIT)
                raise e

        else:
            hwnd = self.Femtet.hWnd

            # terminate
            util.close_femtet(hwnd, timeout, force)

            try:
                pid = _get_pid(hwnd)
                start = time()
                while psutil.pid_exists(pid):
                    if time() - start > 30:  # 30 秒経っても存在するのは何かおかしい
                        logger.error(Msg.ERR_CLOSE_FEMTET_FAILED)
                        break
                    sleep(1)
                sleep(1)

            except (AttributeError, OSError):  # already dead
                pass

    def _setup_before_parallel(self, client):
        client.upload_file(
            self.kwargs['femprj_path'],
            False
        )

    def _version(self):
        return _version(Femtet=self.Femtet)

    def create_postprocess_args(self):
        file_content = self.create_result_file_content()
        jpg_content = self.create_jpg_content()

        out = dict(
            original_femprj_path=self.original_femprj_path,
            model_name=self.model_name,
            pdt_file_content=file_content,
            jpg_file_content=jpg_content,
        )
        return out

    @staticmethod
    def create_pdt_path(femprj_path, model_name, trial):
        result_dir = femprj_path.replace('.femprj', '.Results')
        pdt_path = os.path.join(result_dir, model_name + f'_trial{trial}.pdt')
        return pdt_path

    # noinspection PyMethodOverriding
    @staticmethod
    def postprocess_func(
            trial: int,
            original_femprj_path: str,
            model_name: str,
            pdt_file_content=None,
            jpg_file_content=None,
            dask_scheduler=None
    ):
        result_dir = original_femprj_path.replace('.femprj', '.Results')
        if pdt_file_content is not None:
            pdt_path = FemtetInterface.create_pdt_path(original_femprj_path, model_name, trial)
            with open(pdt_path, 'wb') as f:
                f.write(pdt_file_content)

        if jpg_file_content is not None:
            jpg_path = os.path.join(result_dir, model_name + f'_trial{trial}.jpg')
            with open(jpg_path, 'wb') as f:
                f.write(jpg_file_content)

    def create_result_file_content(self):
        """Called after solve"""
        if self.save_pdt == 'all':
            # save to worker space
            result_dir = self.femprj_path.replace('.femprj', '.Results')
            pdt_path = os.path.join(result_dir, self.model_name + '.pdt')

            self._call_femtet_api(
                fun=self.Femtet.SavePDT,
                args=(pdt_path, True),
                return_value_if_failed=False,
                if_error=SolveError,
                error_message=Msg.ERR_FAILED_TO_SAVE_PDT,
                is_Gaudi_method=False,
            )

            # convert .pdt to ByteIO and return it
            with open(pdt_path, 'rb') as f:
                content = f.read()
            return content

        else:
            return None

    def create_jpg_content(self):
        result_dir = self.femprj_path.replace('.femprj', '.Results')
        jpg_path = os.path.join(result_dir, self.model_name + '.jpg')

        # モデル表示画面の設定
        self.Femtet.SetWindowSize(600, 600)
        self.Femtet.Fit()

        # ---モデルの画面を保存---
        self.Femtet.Redraw()  # 再描画
        succeed = self.Femtet.SavePicture(jpg_path, 600, 600, 80)

        self.Femtet.RedrawMode = True  # 逐一の描画をオン

        if not succeed:
            raise Exception(Msg.ERR_FAILED_TO_SAVE_JPG)

        if not os.path.exists(jpg_path):
            raise Exception(Msg.ERR_JPG_NOT_FOUND)

        with open(jpg_path, 'rb') as f:
            content = f.read()

        return content


from win32com.client import Dispatch, constants


class _UnPicklableNoFEM(FemtetInterface):


    original_femprj_path = 'dummy'
    model_name = 'dummy'
    parametric_output_indexes_use_as_objective = None
    kwargs = dict()
    Femtet = None
    quit_when_destruct = False

    # noinspection PyMissingConstructor
    def __init__(self):
        CoInitialize()
        self.unpicklable_member = Dispatch('FemtetMacro.Femtet')
        self.cns = constants

    def _setup_before_parallel(self, *args, **kwargs):
        pass

    def check_param_value(self, *args, **kwargs):
        pass

    def update_parameter(self, *args, **kwargs):
        pass

    def update(self, *args, **kwargs):
        pass

    def create_result_file_content(self):
        """Called after solve"""

        # save to worker space
        with open(__file__, 'rb') as f:
            content = f.read()

        return content

    def create_file_path(self, trial: int):
        # return path of scheduler environment
        here = os.path.dirname(__file__)
        pdt_path = os.path.join(here, f'trial{trial}.pdt')
        return pdt_path


# レジストリのパスと値の名前
REGISTRY_PATH: Final[str] = r"SOFTWARE\Murata Software\Femtet2014\Femtet"
VALUE_NAME: Final[str] = "AutoSave"

def get_autosave_enabled() -> bool:
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, REGISTRY_PATH) as key:
        value, regtype = winreg.QueryValueEx(key, VALUE_NAME)
        if regtype == winreg.REG_DWORD:
            return bool(value)
        else:
            raise ValueError("Unexpected registry value type.")

def set_autosave_enabled(enable: bool) -> None:
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, REGISTRY_PATH, 0, winreg.KEY_SET_VALUE) as key:
        winreg.SetValueEx(key, VALUE_NAME, 0, winreg.REG_DWORD, int(enable))


def _test_autosave_setting():

    # 使用例
    current_setting = get_autosave_enabled()
    print(f"Current AutoSave setting is {'enabled' if current_setting else 'disabled'}.")

    # 設定を変更する例
    new_setting = not current_setting
    set_autosave_enabled(new_setting)
    print(f"AutoSave setting has been {'enabled' if new_setting else 'disabled'}.")

    # 再度設定を確認する
    after_setting = get_autosave_enabled()
    print(f"Current AutoSave setting is {'enabled' if after_setting else 'disabled'}.")

    assert new_setting == after_setting, "レジストリ編集に失敗しました。" 
