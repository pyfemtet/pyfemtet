import os
import re
import sys
from time import sleep, time
import json
import subprocess
import signal
from abc import ABC, abstractmethod

import pandas as pd
import psutil
from pywintypes import com_error
from pythoncom import CoInitialize, CoUninitialize
from win32com.client import constants, DispatchEx
from dask.distributed import get_worker
from tqdm import trange
from femtetutils import util

from ..core import (
    ModelError,
    MeshError,
    SolveError,
)
from ..dispatch_extensions import (
    dispatch_femtet,
    dispatch_specific_femtet,
    launch_and_dispatch_femtet,
    _get_pid,
    DispatchExtensionException,
)

import logging
from ..logger import get_logger
logger = get_logger('FEM')
logger.setLevel(logging.INFO)

here, me = os.path.split(__file__)


class FEMInterface(ABC):
    """Abstract base class for the interface with FEM software."""

    def __init__(
            self,
            **kwargs
    ):
        """Stores information necessary to restore FEMInterface instance in a subprocess.

        The concrete class should call super().__init__() with the desired arguments when restoring.

        Args:
            **kwargs: keyword arguments for FEMInterface (re)constructor.

        """
        # restore のための情報保管
        self.kwargs = kwargs

    @abstractmethod
    def update(self, parameters: pd.DataFrame) -> None:
        """Updates the FEM analysis based on the proposed parameters."""
        raise NotImplementedError('update() must be implemented.')

    def check_param_value(self, param_name) -> float or None:
        """Checks the value of a parameter in the FEM model (if implemented in concrete class)."""
        if False:
            raise RuntimeError(f"{param_name} doesn't exist on FEM model.")

    def update_parameter(self, parameters: pd.DataFrame) -> None:
        """Updates only FEM variables (if implemented in concrete class)."""
        pass

    def setup_before_parallel(self, client) -> None:
        """Preprocessing before launching a dask worker (if implemented in concrete class).

        Args:
            client: dask client.
            i.e. you can update associated files by
            `client.upload_file(file_path)`
            The file will be saved to dask-scratch-space directory
            without any directory structure.

        """
        pass

    def setup_after_parallel(self):
        """Preprocessing after launching a dask worker and before run optimization (if implemented in concrete class)."""
        pass


class FemtetInterface(FEMInterface):
    """Concrete class for the interface with Femtet software.
    
        Args:
            femprj_path (str or None, optional): The path to the .femprj file. Defaults to None.
            model_name (str or None, optional): The name of the analysis model. Defaults to None.
            connect_method (str, optional): The connection method to use. Can be 'new', 'existing', or 'auto'. Defaults to 'auto'.
            strictly_pid_specify (bool, optional): If True and connect_method=='new', search launched Femtet process strictly based on its process id.

        Warning:
            Even if you specify ``strictly_pid_specify=True`` on the constructor,
            **the connection behavior is like** ``strictly_pid_specify=False`` **in parallel processing**
            because of its large overhead.
            So you should close all Femtet processes before running FEMOpt.main()
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

        # その他のメンバーの宣言や初期化
        self.Femtet = None
        self.quit_when_destruct = False
        self.connected_method = 'unconnected'
        self.parameters = None
        self.max_api_retry = 3
        self.strictly_pid_specify = strictly_pid_specify

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
        self.connect_and_open_femtet()

        # 接続した Femtet の種類に応じて del 時に quit するかどうか決める
        self.quit_when_destruct = self.connected_method == 'new'

        # subprocess で restore するための情報保管
        # パスなどは connect_and_open_femtet での処理結果を反映し
        # メインで開いた解析モデルが確実に開かれるようにする
        super().__init__(
            femprj_path=self.femprj_path,
            model_name=self.model_name,
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

            except AttributeError:  # already dead
                pass
        # CoUninitialize()  # Win32 exception occurred releasing IUnknown at 0x0000022427692748

    def _connect_new_femtet(self):
        logger.info('└ Try to launch and connect new Femtet process.')

        self.Femtet, _ = launch_and_dispatch_femtet(strictly_pid_specify=self.strictly_pid_specify)

        self.connected_method = 'new'


    def _connect_existing_femtet(self, pid: int or None = None):
        logger.info('└ Try to connect existing Femtet process.')
        # 既存の Femtet を探して Dispatch する。
        if pid is None:
            self.Femtet, _ = dispatch_femtet(timeout=5)
        else:
            self.Femtet, _ = dispatch_specific_femtet(pid, timeout=5)
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
            ret_if_failed,
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
        ret_if_failed : Any
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
        logger.debug(' '*print_indent + f'Femtet API:{fun.__name__}, args:{args}, kwargs:{kwargs}')
        if is_Gaudi_method:  # Optimizer は Gogh に触らないので全部にこれをつけてもいい気がする
            try:
                self._call_femtet_api(
                    self.Femtet.Gaudi.Activate,
                    False,  # None 以外なら何でもいい
                    Exception,
                    '解析モデルのオープンに失敗しました',
                    print_indent=print_indent+1
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
                returns = ret_if_failed
            else:
                returns = [ret_if_failed]*(ret_for_check_idx+1)
        logger.debug(' '*print_indent + f'Femtet API result:{returns}')

        # チェックすべき値の抽出
        if ret_for_check_idx is None:
            ret_for_check = returns
        else:
            ret_for_check = returns[ret_for_check_idx]

        # エラーのない場合は戻り値を return する
        if ret_for_check != ret_if_failed:
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
                    ret_if_failed,
                    if_error,
                    error_message,
                    is_Gaudi_method,
                    ret_for_check_idx,
                    args,
                    kwargs,
                    recourse_depth+1,
                    print_indent+1
                )

    def femtet_is_alive(self) -> bool:
        """Returns connected femtet process is exsiting or not."""
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

    def connect_and_open_femtet(self):
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
            if self.connect_method == 'new':
                RuntimeError('Femtet の connect_method に "new" を用いる場合、femprj_path を指定してください。')
            # 開いている Femtet と接続する
            self.connect_femtet('existing')

        # 最終的に接続した Femtet の femprj_path と model を インスタンスに戻す
        self.femprj_path = self.Femtet.Project
        self.model_name = self.Femtet.AnalysisModelName

    def check_param_value(self, param_name):
        """See :func:`FEMInterface.check_param_value`"""
        variable_names = self.Femtet.GetVariableNames()
        if variable_names is not None:
            if param_name in variable_names:
                return self.Femtet.GetVariableValue(param_name)
        message = f'Femtet 解析モデルに変数 {param_name} がありません.'
        message += f'現在のモデルに設定されている変数は {variable_names} です.'
        message += '大文字・小文字の区別に注意してください.'
        raise RuntimeError(message)

    def update_parameter(self, parameters: 'pd.DataFrame'):
        """See :func:`FEMInterface.update_parameter`"""
        self.parameters = parameters.copy()

        # 変数更新のための処理
        sleep(0.1)  # Gaudi がおかしくなる時がある対策
        self._call_femtet_api(
            self.Femtet.Gaudi.Activate,
            True,  # 戻り値を持たないのでここは無意味で None 以外なら何でもいい
            Exception,  # 生きてるのに開けない場合
            error_message='解析モデルが開かれていません',
        )

        # Femtet の設計変数の更新
        existing_variable_names = self._call_femtet_api(
                    fun=self.Femtet.GetVariableNames_py,
                    ret_if_failed=False,  # 意味がない
                    if_error=ModelError,  # 生きてるのに失敗した場合
                    error_message=f'GetVariableNames_py に失敗しました。',
                    is_Gaudi_method=True,
                )

        # 変数を含まないプロジェクトである場合
        if existing_variable_names is None:
            return

        for i, row in parameters.iterrows():
            name = row['name']
            value = row['value']
            if name in existing_variable_names:
                self._call_femtet_api(
                    fun=self.Femtet.UpdateVariable,
                    ret_if_failed=False,
                    if_error=ModelError,  # 生きてるのに失敗した場合
                    error_message=f'変数の更新に失敗しました：変数{name}, 値{value}',
                    is_Gaudi_method=True,
                    args=(name, value),
                )
            else:
                logger.warn(f'変数 {name} は .femprj に含まれていません。無視されます。')

        # ここでは ReExecute しない
        pass

    def update_model(self, parameters: 'pd.DataFrame') -> None:
        """Updates the analysis model only."""

        self.parameters = parameters.copy()

        # 変数の更新
        self.update_parameter(parameters)

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

    def solve(self) -> None:
        """Execute FEM analysis (with updated model)."""
        # # メッシュを切る
        self._call_femtet_api(
            self.Femtet.Gaudi.Mesh,
            0,
            MeshError,
            'メッシュ生成に失敗しました',
            is_Gaudi_method=True,
        )

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
            args=(True,),
        )

    def update(self, parameters: 'pd.DataFrame') -> None:
        """See :func:`FEMInterface.update`"""
        self.parameters = parameters.copy()
        self.update_model(parameters)
        self.solve()

    def quit(self, timeout=1, force=True):
        """Force to terminate connected Femtet."""
        util.close_femtet(self.Femtet.hWnd, timeout, force)

    def setup_before_parallel(self, client):
        client.upload_file(
            self.kwargs['femprj_path'],
            False
        )


class NoFEM(FEMInterface):
    """Interface with no FEM for debug."""
    def update(self, parameters: pd.DataFrame) -> None:
        pass


class FemtetWithNXInterface(FemtetInterface):
    """Femtet with NX interface class.

    Args:
        prt_path: The path to the prt file.
    
    For details of The other arguments, see ``FemtetInterface``.

    """

    _JOURNAL_PATH = os.path.abspath(os.path.join(here, '_FemtetWithNX/update_model.py'))

    def __init__(
            self,
            prt_path,
            femprj_path=None,
            model_name=None,
            connect_method='auto',
            strictly_pid_specify=True,
    ):

        # check NX installation
        self.run_journal_path = os.path.join(os.environ.get('UGII_BASE_DIR'), 'NXBIN', 'run_journal.exe')
        if not os.path.isfile(self.run_journal_path):
            raise FileNotFoundError(r'"%UGII_BASE_DIR%\NXBIN\run_journal.exe" が見つかりませんでした。環境変数 UGII_BASE_DIR 又は NX のインストール状態を確認してください。')

        # 引数の処理
        self.prt_path = os.path.abspath(prt_path)

        # dask サブプロセスのときは prt_path を worker space から取るようにする
        try:
            worker = get_worker()
            space = worker.local_directory
            # worker なら femprj_path が None でないはず
            self.prt_path = os.path.join(space, os.path.basename(self.prt_path))
        except ValueError:  # get_worker に失敗した場合
            pass

        # FemtetInterface の設定 (femprj_path, model_name の更新など)
        # + restore 情報の上書き
        super().__init__(
            femprj_path=femprj_path,
            model_name=model_name,
            connect_method=connect_method,
            strictly_pid_specify=strictly_pid_specify,
            prt_path=self.prt_path,
        )


    def check_param_value(self, name):
        """Override FemtetInterface.check_param_value().
        
        Do nothing because the parameter can be registered
        to not only .femprj but also .prt.
        
        """
        pass

    def setup_before_parallel(self, client):
        client.upload_file(
            self.kwargs['prt_path'],
            False
        )
        super().setup_before_parallel(client)

    def update_model(self, parameters: 'pd.DataFrame') -> None:
        """Update .x_t"""

        self.parameters = parameters.copy()

        # Femtet が参照している x_t パスを取得する
        x_t_path = self.Femtet.Gaudi.LastXTPath

        # 前のが存在するならば消しておく
        if os.path.isfile(x_t_path):
            os.remove(x_t_path)

        # 変数の json 文字列を作る
        tmp_dict = {}
        for i, row in parameters.iterrows():
            tmp_dict[row['name']] = row['value']
        str_json = json.dumps(tmp_dict)

        # NX journal を使ってモデルを編集する
        env = os.environ.copy()
        subprocess.run(
            [self.run_journal_path, self._JOURNAL_PATH, '-args', self.prt_path, str_json, x_t_path],
            env=env,
            shell=True,
            cwd=os.path.dirname(self.prt_path)
        )

        # この時点で x_t ファイルがなければ NX がモデル更新に失敗しているはず
        if not os.path.isfile(x_t_path):
            raise ModelError

        # モデルの再インポート
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

        # femprj モデルの変数も更新
        super().update_model(parameters)


class FemtetWithSolidworksInterface(FemtetInterface):

    # 定数の宣言
    swThisConfiguration = 1  # https://help.solidworks.com/2023/english/api/swconst/SOLIDWORKS.Interop.swconst~SOLIDWORKS.Interop.swconst.swInConfigurationOpts_e.html
    swAllConfiguration = 2
    swSpecifyConfiguration = 3  # use with ConfigName argument
    swSaveAsCurrentVersion = 0
    swSaveAsOptions_Copy = 2  #
    swSaveAsOptions_Silent = 1  # https://help.solidworks.com/2021/english/api/swconst/solidworks.interop.swconst~solidworks.interop.swconst.swsaveasoptions_e.html
    swSaveWithReferencesOptions_None = 0  # https://help-solidworks-com.translate.goog/2023/english/api/swconst/SolidWorks.Interop.swconst~SolidWorks.Interop.swconst.swSaveWithReferencesOptions_e.html?_x_tr_sl=auto&_x_tr_tl=ja&_x_tr_hl=ja&_x_tr_pto=wapp
    swDocPART = 1  # https://help.solidworks.com/2023/english/api/swconst/SOLIDWORKS.Interop.swconst~SOLIDWORKS.Interop.swconst.swDocumentTypes_e.html

    def __init__(
            self,
            sldprt_path,
            femprj_path=None,
            model_name=None,
            connect_method='auto',
            strictly_pid_specify=True,
    ):
        # 引数の処理
        self.sldprt_path = os.path.abspath(sldprt_path)

        # dask サブプロセスのときは prt_path を worker space から取るようにする
        try:
            worker = get_worker()
            space = worker.local_directory
            # worker なら femprj_path が None でないはず
            self.prt_path = os.path.join(space, os.path.basename(self.prt_path))
        except ValueError:  # get_worker に失敗した場合
            pass

        # FemtetInterface の設定 (femprj_path, model_name の更新など)
        # + restore 情報の上書き
        super().__init__(
            femprj_path=femprj_path,
            model_name=model_name,
            connect_method=connect_method,
            strictly_pid_specify=strictly_pid_specify,
            sldprt_path=self.sldprt_path,
        )

    def initialize_sldworks_connection(self):
        # SolidWorks を捕まえ、ファイルを開く
        self.swApp = DispatchEx('SLDWORKS.Application')
        self.swApp.Visible = True

        # open model
        self.swApp.OpenDoc(self.sldprt_path, self.swDocPART)
        self.swModel = self.swApp.ActiveDoc
        self.swEqnMgr = self.swModel.GetEquationMgr
        self.nEquation = self.swEqnMgr.GetCount

    def check_param_value(self, param_name):
        """Override FemtetInterface.check_param_value().
        
        Do nothing because the parameter can be registered
        to not only .femprj but also .SLDPRT.
        
        """
        pass

    def setup_before_parallel(self, client):
        client.upload_file(
            self.kwargs['sldprt_path'],
            False
        )
        super().setup_before_parallel(client)

    def setup_after_parallel(self):
        CoInitialize()
        self.initialize_sldworks_connection()

    def update_model(self, parameters: pd.DataFrame):
        """Update .x_t"""

        self.parameters = parameters.copy()

        # Femtet が参照している x_t パスを取得する
        x_t_path = self.Femtet.Gaudi.LastXTPath

        # 前のが存在するならば消しておく
        if os.path.isfile(x_t_path):
            os.remove(x_t_path)

        # solidworks のモデルの更新
        self.update_sw_model(parameters)

        # export as x_t
        self.swModel.SaveAs(x_t_path)

        # 30 秒待っても x_t ができてなければエラー(COM なので)
        timeout = 30
        start = time()
        while True:
            if os.path.isfile(x_t_path):
                break
            if time()-start > timeout:
                raise ModelError('モデル再構築に失敗しました')
            sleep(1)

        # モデルの再インポート
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

        # femprj モデルの変数も更新
        super().update_model(parameters)

    def update_sw_model(self, parameters: pd.DataFrame):
        """Update .sldprt"""
        # df を dict に変換
        user_param_dict = {}
        for i, row in parameters.iterrows():
            user_param_dict[row['name']] = row['value']

        # プロパティを退避
        buffer_aso = self.swEqnMgr.AutomaticSolveOrder
        buffer_ar = self.swEqnMgr.AutomaticRebuild
        self.swEqnMgr.AutomaticSolveOrder = False
        self.swEqnMgr.AutomaticRebuild = False

        for i in range(self.nEquation):
            # name, equation の取得
            current_equation = self.swEqnMgr.Equation(i)
            current_name = self._get_name_from_equation(current_equation)
            # 対象なら処理
            if current_name in list(user_param_dict.keys()):
                new_equation = f'"{current_name}" = {user_param_dict[current_name]}'
                self.swEqnMgr.Equation(i, new_equation)

        # 式の計算
        # noinspection PyStatementEffect
        self.swEqnMgr.EvaluateAll  # always returns -1

        # プロパティをもとに戻す
        self.swEqnMgr.AutomaticSolveOrder = buffer_aso
        self.swEqnMgr.AutomaticRebuild = buffer_ar

        # 更新する（ここで失敗はしうる）
        result = self.swModel.EditRebuild3  # モデル再構築
        if not result:
            raise ModelError('モデル再構築に失敗しました')

    def _get_name_from_equation(self, equation:str):
        pattern = r'^\s*"(.+?)"\s*$'
        matched = re.match(pattern, equation.split('=')[0])
        if matched:
            return matched.group(1)
        else:
            return None
