from abc import ABC, abstractmethod

import os
import sys
from time import sleep
import tempfile
import json
import subprocess

import shutil
from pywintypes import com_error
from win32com.client import constants
from femtetutils import util

from .core import (
    FemtetAutomationError,
    ModelError,
    MeshError,
    SolveError,
    InterprocessVariables,
)
from pyfemtet.tools.DispatchUtils import (
    Dispatch_Femtet,
    Dispatch_Femtet_with_specific_pid,
    Dispatch_Femtet_with_new_process,
    _get_pid
)

here, me = os.path.split(__file__)


class FEMInterface(ABC):
    """Abstract base class for the interface with FEM software."""

    def __init__(
            self,
            subprocess_idx: int or None = None,
            subprocess_settings: 'Any' or None = None,  # 具象クラスで引数に取ることが必須
            **kwargs
    ):
        """Stores information necessary to restore FEMInterface instance in a subprocess.

        The concrete class should call super().__init__() with the desired arguments when restoring.

        Args:
            subprocess_idx (int or None, optional): The index of the subprocess. Defaults to None.
            subprocess_settings ('Any' or None, optional): Additional settings for the subprocess. Defaults to None.
            **kwargs: keyword arguments for FEMInterface constructor.

        """
        # restore のための情報保管
        self.kwargs = kwargs
        # base record 時にが呼ぶので attr があったほうがいい
        if not hasattr(self, 'subprocess_idx'):
            self.subprocess_idx = subprocess_idx

    def check_param_value(self, param_name) -> float:
        """Checks the value of a parameter in the FEM model.

        Args:
            param_name (str): The name of the parameter.

        Raises:
            Exception: If param_name does not exist in the FEM model.

        Returns:
            float: The value of the parameter.

        """
        pass

    @abstractmethod
    def update(self, parameters: 'pd.DataFrame') -> None:
        """Updates the FEM analysis based on the proposed parameters.

        Args:
            parameters (pd.DataFrame): The parameters to be updated. Must have at least 'name' and 'value'.

         Raises:
            ModelError, MeshError, SolveError: If there are errors in updating or analyzing the model.

         Returns:
            None

         """
        pass

    def update_parameter(self, parameters: 'pd.DataFrame'):
        """Updates only FEM variables that can be obtained with Femtet.GetVariableValue and similar functions.

        If users define a function to be passed to add_objective etc. to take OptimizerBase as an argument,
        and access the variable by calling OptimizerBase.get_parameter() within the function,
        there is no need to implement this method in the concrete class of FEMInterface.

        Args:
         parameters (pd.DataFrame): The parameters to be updated.

        """
        pass

    def quit(self, *args, **kwargs):
        """Destructor."""
        pass

    def settings_before_parallel(self, femopt) -> 'Any':
        """Preprocessing before launching a subprocess."""
        pass  # return subprocess_settings

    def parallel_setup(self):
        """Constructor called in a subprocess."""
        pass

    def parallel_terminate(self):
        """Destructor called in a subprocess."""
        pass


class FemtetInterface(FEMInterface):
    """Concrete class for the interface with Femtet software."""    

    def __init__(
            self,
            femprj_path=None,
            model_name=None,
            connect_method='auto',
            subprocess_idx=None,
            subprocess_settings=None,
    ):
        """Initializes the FemtetInterface.

        Args:
            femprj_path (str or None, optional): The path to the .femprj file. Defaults to None.
            model_name (str or None, optional): The name of the analysis model. Defaults to None.
            connect_method (str, optional): The connection method to use. Can be 'new', 'existing', or 'auto'. Defaults to 'auto'.
            subprocess_idx (int or None, optional): The index of the subprocess. Defaults to None.
            subprocess_settings ('Any' or None, optional): Additional settings for the subprocess. Defaults to None.

        Tip:
            If you search for information about the method to connect python and Femtet, see :func:`connect_femtet`.

        """

        # 引数の処理
        if femprj_path is None:
            self.femprj_path = None
        else:
            self.femprj_path = os.path.abspath(femprj_path)
        self.model_name = model_name
        self.connect_method = connect_method
        # 引数の処理（サブプロセス時）
        self.subprocess_idx = subprocess_idx
        self.ipv = None if subprocess_settings is None else subprocess_settings['ipv']  # 排他処理に使う
        self.target_pid = None if subprocess_settings is None else subprocess_settings['pids'][subprocess_idx]  # before_parallel で立てた Femtet への接続に使う
        self.femprj_path = self.femprj_path if subprocess_settings is None else subprocess_settings['new_femprj_path'][subprocess_idx]

        # その他の初期化
        self.Femtet: 'IPyDispatch' = None
        self.connected_method = 'unconnected'
        self.parameters = None
        self.max_api_retry = 3
        self.pp = False

        # サブプロセスでなければルールに従って Femtet と接続する
        if subprocess_idx is None:
            self.connect_and_open_femtet()

        # サブプロセスから呼ばれているならば
        # settings_before_parallel で起動されているはずの Femtet と接続する.
        # connect_and_open_femtet は排他処理で行う.
        else:
            print('Start to connect femtet. This process is exclusive.')
            while True:
                print(f'My subprocess_idx is {self.subprocess_idx}')
                print(f'Connection allowed idx is {self.ipv.get_allowed_idx()}')
                if self.subprocess_idx == self.ipv.get_allowed_idx():
                    break
                print(f'Wait to be allowed.')
                sleep(1)
            print(f'Allowed.')
            self.connect_femtet('existing', self.target_pid)
            self.ipv.set_allowed_idx(self.subprocess_idx + 1)  # 次の idx に許可を出す
            self.open(self.femprj_path, self.model_name)

        # restore するための情報保管
        # パスなどは connect_and_open_femtet での処理結果を反映し
        # メインで開いた解析モデルが確実に開かれるようにする
        super().__init__(
            femprj_path=self.femprj_path,
            model_name=self.model_name,
        )

    def _connect_new_femtet(self):
        self.Femtet, _ = Dispatch_Femtet_with_new_process()
        self.connected_method = 'new'

    def _connect_existing_femtet(self, pid: int or None = None):
        # 既存の Femtet を探して Dispatch する。
        if pid is None:
            self.Femtet, my_pid = Dispatch_Femtet(timeout=5)
        else:
            self.Femtet, my_pid = Dispatch_Femtet_with_specific_pid(pid)
        if my_pid == 0:
            raise FemtetAutomationError('接続できる Femtet のプロセスが見つかりませんでした。')
        self.connected_method = 'existing'

    def connect_femtet(self, connect_method: str = 'new', pid: int or None = None):
        """Connects to a Femtet process.
        Args:
            connect_method (str, optional): The connection method. Can be 'new', 'existing', or 'auto'. Defaults to 'new'.
            pid (int or None, optional): The process ID of an existing Femtet process. Only used when `connect_method` is set as `'existing'`. Defaults to `None`.

        When connect_method is 'new', starts a new Femtet process and connects to it.

        When 'existing', connect to an existing Femtet process.
        However, if there are no Femtets to which it can connect, it will throw an error.
        In this case, you cannot control which of the existing Femtet processes is connected.
        
        When 'existing', you can specify the pid if it is known.
        Also, if it is already connected to another Python or Excel process.
        Connections cannot be made with Femtet.

        When set to 'auto', first tries 'existing', and if that fails, connects with 'new'.        

        """

        if connect_method == 'new':
            self._connect_new_femtet()

        elif connect_method == 'existing':
            self._connect_existing_femtet(pid)

        elif connect_method == 'auto':
            try:
                self._connect_existing_femtet(pid)
            except FemtetAutomationError:
                self._connect_new_femtet()

        else:
            raise Exception(f'{connect_method} は定義されていない接続方法です')

        # ensure makepy
        if not hasattr(constants, 'STATIC_C'):
            cmd = f'{sys.executable} -m win32com.client.makepy FemtetMacro'
            os.system(cmd)
            message = 'Femtet python マクロ定数の設定が完了してないことを検出しました.'
            message += '次のコマンドにより、設定は自動で行われました（python  -m win32com.client.makepy FemtetMacro）.'
            message += 'インタープリタを再起動してください.'
            raise Exception(message)

        if self.Femtet is None:
            raise Exception('Femtet との接続に失敗しました.')

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
        if args is None:
            args = tuple()
        if kwargs is None:
            kwargs = dict()

        # 処理の失敗には 2 パターンある.
        # 1. 結果に関わらず戻り値が None で API 実行時に com_error を送出する
        # 2. API 実行時に成功失敗を示す戻り値を返し、ShowLastError で例外にアクセスできる状態になる

        # Gaudi コマンドなら Gaudi.Activate する
        if self.pp: print(' '*print_indent, 'コマンド', fun.__name__, args, kwargs)
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
        if self.pp: print(' '*print_indent, 'コマンド結果', returns)

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
                if self.pp: print(' '*print_indent, error_message)
                raise if_error(error_message)

            # 死んでいるなら再起動
            else:
                # 再起動試行回数の上限に達していたら諦める
                if self.pp: print(' '*print_indent, '現在の Femtet 再起動回数：', recourse_depth)
                if recourse_depth >= self.max_api_retry:
                    raise Exception('Femtet のプロセスが異常終了し、正常に再起動できませんでした.')

                # 再起動
                if self.pp: print(' '*print_indent, 'Femtet プロセスの異常終了が検知されました. 再起動を試みます.')
                print('Femtet プロセスの異常終了が検知されました. 回復を試みます.')
                self.connect_femtet(connect_method='new')
                self.open(self.femprj_path, self.model_name)

                # 状態を復元するために一度変数を渡して解析を行う（fun.__name__がSolveなら2度手間だが）
                if self.pp: print(' '*print_indent, f'再起動されました. コマンド{fun.__name__}の再試行のため、解析を行います.')
                self.update(self.parameters)

                # 与えられた API の再帰的再試行
                if self.pp: print(' '*print_indent, f'回復されました. コマンド{fun.__name__}の再試行を行います.')
                print('回復されました.')
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

        print(f'try to open {self.model_name} of {self.femprj_path}')

        # femprj が指定されている
        if self.femprj_path is not None:
            # auto でいいが、その後違ったら open する
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
            # new だと解析すべき femprj がわからない
            if self.connect_method == 'new':
                Exception('Femtet の connect_method に "new" を用いる場合、femprj_path を指定してください。')
            print('femprj_path が指定されていないため、開いている Femtet との接続を試行します。')
            self.connect_femtet('existing', self.target_pid)
            # 接続した Femtet インスタンスのプロジェクト名などを保管する
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
        raise Exception(message)

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
                print(f'変数 {name} は .femprj に含まれていません。無視されます。')

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

    def settings_before_parallel(self, femopt) -> dict:
        """Before starting the subprocess, create the required number of Femtet and copy femprj to a temporary file."""

        pids = []
        new_femprj_pathes = []
        for i in range(femopt.n_parallel - 1):

            # Femtet 起動
            if not util.execute_femtet():
                raise Exception('femtetutils を用いた Femtet の起動に失敗しました')

            # pid 取得
            pid = util.get_last_executed_femtet_process_id()
            if pid == 0:
                raise Exception('起動された Femtet の認識に失敗しました')
            pids.append(pid)

            # 一時フォルダ生成、ファイルをコピー
            td = tempfile.mkdtemp()
            name = 'subprocess'
            new_femprj_path = os.path.join(td, f'{name}.femprj')
            shutil.copy(self.femprj_path, new_femprj_path)
            new_femprj_pathes.append(new_femprj_path)

        subprocess_settings = dict(
            ipv=femopt.ipv,
            pids=pids,
            new_femprj_path=new_femprj_pathes,
        )

        return subprocess_settings

    def parallel_terminate(self):
        """Terminate Femtet instances started by settings_before_parallel."""
        # 何かがあってもあとは終わるだけなので
        # プロセス終了 print が出るようにする。
        try:
            # Femtet プロセスを終了する（自動保存しないよう保存する）
            print(f'try to close Femtet.exe')
            self.Femtet.SaveProjectIgnoreHistory(self.Femtet.Project, True)  # 動いてない？？
            sleep(3)
            self.quit()
            sleep(1)

            # 一時ファイルを削除する
            try:
                shutil.rmtree(self.td)
            except PermissionError:
                pass  # 諦める

        except:
            pass


class NoFEM(FEMInterface):
    """Interface with no FEM for debug."""

    def update(self, parameters):
        pass


class FemtetWithNXInterface(FemtetInterface):

    PATH_JOURNAL = os.path.abspath(os.path.join(here, '_FemtetWithNX/update_model.py'))

    def __init__(
            self,
            prt_path,
            femprj_path=None,
            model_name=None,
            connect_method='auto',
            subprocess_idx=None,
            subprocess_settings=None,
    ):
        """
        Initializes the FemtetWithNXInterface class.

        Args:
            prt_path: The path to the prt file.
            femprj_path: The path to the femprj file (optional).
            model_name: The name of the model (optional).
            connect_method: The connection method (default 'auto').
            subprocess_idx: The subprocess index (optional).
            subprocess_settings: The subprocess settings (optional).

        Returns:
            None
        """

        self.prt_path = os.path.abspath(prt_path)
        super().__init__(
            femprj_path=femprj_path,
            model_name=model_name,
            connect_method=connect_method,
            subprocess_idx=subprocess_idx,
            subprocess_settings=subprocess_settings,
        )

    def check_param_value(self, name):
        # desable FemtetInterface.check_param_value()
        # because the parameter can be registerd to not only .femprj but also .prt.
        return None

    def update_model(self, parameters: 'pd.DataFrame') -> None:
        self.parameters = parameters.copy()

        # 変数更新のための処理
        sleep(0.1)  # Gaudi がおかしくなる時がある対策
        self._call_femtet_api(
            self.Femtet.Gaudi.Activate,
            True,  # 戻り値を持たないのでここは無意味で None 以外なら何でもいい
            Exception,  # 生きてるのに開けない場合
            error_message='解析モデルが開かれていません',
        )

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
        exe = r'%UGII_BASE_DIR%\NXBIN\run_journal.exe'
        env = os.environ.copy()
        subprocess.run(
            [exe, self.PATH_JOURNAL, '-args', self.prt_path, str_json, x_t_path],
            env=env,
            shell=True,
            cwd=os.path.dirname(self.prt_path)
        )

        # この時点で x_t ファイルがなければ NX がモデル更新に失敗しているはず
        if not os.path.isfile(x_t_path):
            raise ModelError

        # 設計変数に従ってモデルを再構築
        self._call_femtet_api(
            self.Femtet.Gaudi.ReExecute,
            False,
            ModelError,  # 生きてるのに失敗した場合
            error_message=f'モデル再構築に失敗しました.',
            is_Gaudi_method=True,
        )

        # femprj モデルの変数も更新
        super().update_model(parameters)

        # 処理を確定
        self._call_femtet_api(
            self.Femtet.Redraw,
            False,  # 戻り値は常に None なのでこの変数に意味はなく None 以外なら何でもいい
            ModelError,  # 生きてるのに失敗した場合
            error_message=f'モデル再構築に失敗しました.',
            is_Gaudi_method=True,
        )


# TODO: SW-Femtet 再実装


