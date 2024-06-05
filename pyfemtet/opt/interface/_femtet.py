from typing import Optional, List

import os
import sys
from time import sleep, time
import signal

import pandas as pd
import psutil
from dask.distributed import get_worker

from pywintypes import com_error
from pythoncom import CoInitialize, CoUninitialize
from win32com.client import constants
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


class FemtetInterface(FEMInterface):
    """Concrete class for the interface with Femtet.

        Args:
            femprj_path (str or None, optional): The path to the .femprj file. Defaults to None.
            model_name (str or None, optional): The name of the analysis model. Defaults to None.
            connect_method (str, optional): The connection method to use. Can be 'new', 'existing', or 'auto'. Defaults to 'auto'.
            strictly_pid_specify (bool, optional): If True and connect_method=='new', search launched Femtet process strictly based on its process id.
            allow_without_project (bool, optional): Allow to launch Femtet with no project file. Default to False.
            open_result_with_gui (bool, optional): Open analysis result with Femtet GUI. Default to True.
            parametric_output_indexes_use_as_objective (list of int, optional): Parametric output indexes which will be used as objective functions. Parametric output should be set on Femtet parametric analysis dialog. Note that output 'No.' in dialog is starts with 1, but this 'index' is starts with 0. Default to None.

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
            connect_method='auto',
            strictly_pid_specify=True,
            allow_without_project=False,
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

        # その他のメンバーの宣言や初期化
        self.Femtet = None
        self.femtet_pid = 0
        self.quit_when_destruct = False
        self.connected_method = 'unconnected'
        self.parameters = None
        self.max_api_retry = 3
        self.strictly_pid_specify = strictly_pid_specify
        if parametric_output_indexes_use_as_objective is None:
            self.parametric_output_indexes_use_as_objective = []
        else:
            self.parametric_output_indexes_use_as_objective = parametric_output_indexes_use_as_objective

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
            **kwargs
        )

    def __del__(self):
        if self.quit_when_destruct:
            try:
                # 強制終了 TODO: 自動保存を回避する
                hwnd = self.Femtet.hWnd
                pid = _get_pid(hwnd)
                util.close_femtet(hwnd, 1, True)
                start = time()
                while psutil.pid_exists(pid):
                    if time() - start > 30:  # 30 秒経っても存在するのは何かおかしい
                        os.kill(pid, signal.SIGKILL)
                        break
                    sleep(1)
                sleep(1)

            except (AttributeError, OSError):  # already dead
                pass
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
            message = 'Femtet python マクロ定数の設定が完了してないことを検出しました.'
            message += '次のコマンドにより、設定は自動で行われました（python -m win32com.client.makepy FemtetMacro）.'
            message += 'インタープリタを再起動してください.'
            raise RuntimeError(message)

        if self.Femtet is None:
            raise RuntimeError('Femtet との接続に失敗しました.')

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
        fun : Callable
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

        # FIXME: Gaudi へのアクセスなど、self.Femtet.Gaudi.somefunc() のような場合、この関数を呼び出す前に Gaudi へのアクセスの時点で com_error が起こる
        # FIXME: => 文字列で渡して eval() すればよい。

        if args is None:
            args = tuple()
        if kwargs is None:
            kwargs = dict()

        # 処理の失敗には 2 パターンある.
        # 1. 結果に関わらず戻り値が None で API 実行時に com_error を送出する
        # 2. API 実行時に成功失敗を示す戻り値を返し、ShowLastError で例外にアクセスできる状態になる

        # Gaudi コマンドなら Gaudi.Activate する
        logger.debug(' ' * print_indent + f'Femtet API:{fun.__name__}, args:{args}, kwargs:{kwargs}')
        if is_Gaudi_method:  # Optimizer は Gogh に触らないので全部にこれをつけてもいい気がする
            try:
                self._call_femtet_api(
                    self.Femtet.Gaudi.Activate,
                    False,  # None 以外なら何でもいい
                    Exception,
                    '解析モデルのオープンに失敗しました',
                    print_indent=print_indent + 1
                )
            except com_error:
                # Gaudi へのアクセスだけで com_error が生じうる
                # そういう場合は次の API 実行で間違いなくエラーになるので放っておく
                pass

        # API を実行
        try:
            returns = fun(*args, **kwargs)
        except com_error:
            # パターン 2 エラーが生じたことは確定なのでエラーが起こるよう returns を作る
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
                    raise Exception('Femtet のプロセスが異常終了し、正常に再起動できませんでした.')

                # 再起動
                logger.warn('Femtet プロセスの異常終了が検知されました. 回復を試みます.')
                CoInitialize()
                self.connect_femtet(connect_method='new')
                self.open(self.femprj_path, self.model_name)

                # 状態を復元するために一度変数を渡して解析を行う（fun.__name__がSolveなら2度手間だが）
                logger.info(' ' * print_indent + f'Femtet が再起動されました。解析を行い、状態回復を試みます。')
                self.update(self.parameters)

                # 与えられた API の再帰的再試行
                logger.info(' ' * print_indent + f'Femtet が回復されました。コマンド {fun.__name__} を再試行します。')
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
        return _get_pid(self.Femtet.hWnd) > 0

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
                raise RuntimeError(
                    'femprj_path を指定せず Femtet の connect_method に "new" を指定する場合、"allow_without_project" 引数を True に設定してください。')
            # さらに auto の場合は Femtet が存在しなければ new と同じ挙動になるので同様の処理
            if (
                    (self.connect_method == 'auto')
                    and (len(_get_pids(process_name='Femtet.exe')) == 0)
                    and (not self.allow_without_project)
            ):
                raise RuntimeError(
                    'femprj_path を指定せず Femtet の connect_method を指定しない（又は "auto" に指定する）場合、Femtet を起動して処理したい .femprj ファイルを開いた状態にしてください。')
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
        if self._version() >= _version(2023, 1, 1):
            variable_names = self.Femtet.GetVariableNames_py()
            if variable_names is not None:
                if param_name in variable_names:
                    return self.Femtet.GetVariableValue(param_name)
            message = f'Femtet 解析モデルに変数 {param_name} がありません.'
            message += f'現在のモデルに設定されている変数は {variable_names} です.'
            message += '大文字・小文字の区別に注意してください.'
            raise RuntimeError(message)
        else:
            return None

    def update_parameter(self, parameters: 'pd.DataFrame', with_warning=False):
        """Update parameter of femprj."""
        self.parameters = parameters.copy()

        # 変数更新のための処理
        sleep(0.1)  # Gaudi がおかしくなる時がある対策
        self._call_femtet_api(
            self.Femtet.Gaudi.Activate,
            True,  # 戻り値を持たないのでここは無意味で None 以外なら何でもいい
            Exception,  # 生きてるのに開けない場合
            error_message='解析モデルが開かれていません',
        )

        if self._version() >= _version(2023, 1, 1):
            # Femtet の設計変数の更新
            existing_variable_names = self._call_femtet_api(
                fun=self.Femtet.GetVariableNames_py,
                return_value_if_failed=False,  # 意味がない
                if_error=ModelError,  # 生きてるのに失敗した場合
                error_message=f'GetVariableNames_py に失敗しました。',
                is_Gaudi_method=True,
            )

            # 変数を含まないプロジェクトである場合
            if existing_variable_names is None:
                if with_warning:
                    return ['解析モデルに変数が含まれていません。']
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
                        error_message=f'変数の更新に失敗しました：変数{name}, 値{value}',
                        is_Gaudi_method=True,
                        args=(name, value),
                    )
                else:
                    msg = f'変数 {name} は 解析モデル {self.model_name} に含まれていません。無視されます。'
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
                    error_message=f'変数の更新に失敗しました：変数{name}, 値{value}',
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
            self.Femtet.Gaudi.ReExecute,
            False,
            ModelError,  # 生きてるのに失敗した場合
            error_message=f'モデル再構築に失敗しました.',
            is_Gaudi_method=True,
        )

        # 処理を確定
        self._call_femtet_api(
            self.Femtet.Redraw,
            False,  # 戻り値は常に None なのでこの変数に意味はなく None 以外なら何でもいい
            ModelError,  # 生きてるのに失敗した場合
            error_message=f'モデル再構築に失敗しました.',
            is_Gaudi_method=True,
        )

        if with_warning:
            return warnings or []

    def solve(self) -> None:
        """Execute FEM analysis."""
        # # メッシュを切る
        self._call_femtet_api(
            self.Femtet.Gaudi.Mesh,
            0,
            MeshError,
            'メッシュ生成に失敗しました',
            is_Gaudi_method=True,
        )

        if self.parametric_output_indexes_use_as_objective is not None:
            from pyfemtet.opt.interface._femtet_parametric import solve_via_parametric_dll
            self._call_femtet_api(
                fun=solve_via_parametric_dll,
                return_value_if_failed=False,
                if_error=SolveError,
                error_message='パラメトリック解析を用いたソルブに失敗しました',
                is_Gaudi_method=True,
                args=(self.Femtet,),
            )
        else:
            # # ソルブする
            self._call_femtet_api(
                self.Femtet.Solve,
                False,
                SolveError,
                'ソルブに失敗しました',
                is_Gaudi_method=True,
            )

        # 次に呼ばれるはずのユーザー定義コスト関数の記述を簡単にするため先に解析結果を開いておく
        self._call_femtet_api(
            self.Femtet.OpenCurrentResult,
            False,
            SolveError,  # 生きてるのに開けない場合
            error_message='解析結果のオープンに失敗しました',
            is_Gaudi_method=True,
            args=(self.open_result_with_gui,),
        )

    def update(self, parameters: 'pd.DataFrame') -> None:
        """See :func:`FEMInterface.update`"""
        self.parameters = parameters.copy()
        self.update_model(parameters)
        # TODO: CAD 連携における座標を基にした境界条件の割当直しなどの処理をここに挟めるようにする
        self.solve()

    def quit(self, timeout=1, force=True):
        """Force to terminate connected Femtet."""
        util.close_femtet(self.Femtet.hWnd, timeout, force)

    def _setup_before_parallel(self, client):
        client.upload_file(
            self.kwargs['femprj_path'],
            False
        )

    def _version(self):
        return _version(Femtet=self.Femtet)
