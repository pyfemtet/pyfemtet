from abc import ABC, abstractmethod

import os
import sys
from time import sleep
import tempfile
import shutil

from win32com.client import constants
from femtetutils import util

from .core import (
    FemtetAutomationError,
    ModelError,
    MeshError,
    SolveError,
)
from pyfemtet.tools.DispatchUtils import (
    Dispatch_Femtet,
    Dispatch_Femtet_with_specific_pid,
    Dispatch_Femtet_with_new_process
)


class FEMIF(ABC):

    def __init__(self, **kwargs):
        """サブプロセスで FEM を restore するときに必要な情報.

        ユーザーが指定した args, kwargs を保存する.
        具象クラスでは super().__init__ にすべての引数を
        キーワード付きで保存すること.

        """
        self.kwargs = kwargs

    @abstractmethod
    def check_param_value(self, param_name) -> float:
        """

        Parameters
        ----------
        param_name : str

        Raises
        ----------
        Exception
            param_name が FEM に存在しない

        Returns
        -------
        value : float

        """
        pass

    @abstractmethod
    def update(self, parameters: 'pd.DataFrame') -> None:
        """提案された parameters に基づいて FEM 解析を更新する

        Parameters
        ----------
        parameters : pd.DataFrame
            最低でも name, value を有する.

        Raises
        ----------
        ModelError, MeshError, SolveError
            モデルの更新, 解析に失敗した場合.

        Returns
        -------
        None


        """
        pass

    def quit(self, *args, **kwargs):
        """デストラクタ."""
        pass

    def parallel_setup(self, subprocess_idx, *args, **kwargs):
        """サブプロセスで呼ばれた場合のコンストラクタ."""
        pass

    def parallel_terminate(self, *args, **kwargs):
        """サブプロセスで呼ばれた場合のデストラクタ."""
        pass


class Femtet(FEMIF):

    def __init__(
            self,
            femprj_path=None,
            model_name=None,
            connect_method='auto',
            pid=None
    ):
        self.connected_femtet = 'unconnected'
        if femprj_path is None:
            self.femprj_path = None
        else:
            self.femprj_path = os.path.abspath(femprj_path)
        self.model_name = model_name
        self.Femtet: 'IPyDispatch' = None

        # femprj が指定されている
        if femprj_path is not None:
            # auto でも connect_femtet でもいいが、その後違ったら open する
            self.connect_femtet(connect_method)
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
            if connect_method == 'new':
                Exception('Femtet の connect_method に "new" を用いる場合、femprj_path を指定してください。')
            print('femprj_path が指定されていないため、開いている Femtet との接続を試行します。')
            self.connect_femtet('catch', pid)
            # 接続した Femtet インスタンスのプロジェクト名などを保管する
            self.femprj_path = self.Femtet.ProjectPath
            self.model_name = self.Femtet.AnalysisModelName

        super().__init__(
            femprj_path=femprj_path,
            model_name=model_name,
            connect_method=connect_method,
            pid=pid,
        )

    def _connect_new_femtet(self):
        self.Femtet, _ = Dispatch_Femtet_with_new_process()
        self.connected_femtet = 'new'

    def _connect_existing_femtet(self, pid: int or None = None):
        # 既存の Femtet を探して Dispatch する。
        if pid is None:
            self.Femtet, my_pid = Dispatch_Femtet(timeout=5)
        else:
            self.Femtet, my_pid = Dispatch_Femtet_with_specific_pid(pid)
        if my_pid == 0:
            raise FemtetAutomationError('接続できる Femtet のプロセスが見つかりませんでした。')
        self.connected_femtet = 'existing'

    def connect_femtet(self, connect_method: str = 'new', pid: int or None = None):
        """
        Femtet プロセスとの接続を行い、メンバー変数 Femtet にインスタンスをセットします。

        Parameters
        ----------
        connect_method : str, optional
            'new' or 'catch' or 'auto'. The default is 'new'.
            'new' のとき、新しい Femtet プロセスを起動し、接続します。
            'catch' のとき、既存の Femtet プロセスと接続します。
            ただし、接続できる Femtet が存在しない場合、エラーを送出します。
            この場合、既存 Femtet プロセスのどれに接続されるかは制御できません。
            また、すでに別のPython又はExcelプロセスと接続されている Femtet との接続を行うことはできません。
            'auto' のとき、まず 'catch' を試行し、失敗した際には 'new' で接続します。

        pid : int or None, optional
            接続したい既存の Femtet のプロセス ID. The default is None.
            strategy が catch 以外の時は無視されます。

        Raises
        ------
        Exception
            何らかの理由で Femtet との接続に失敗した際に例外を送出します。

        Returns
        -------
        connected_femtet : str.
            接続された Femtet が既存のものならば 'existing',
            新しいものならば 'new' を返します。

        """
        if connect_method == 'new':
            self._connect_new_femtet()

        elif connect_method == 'catch':
            self._connect_existing_femtet(pid)

        elif connect_method == 'auto':
            try:
                self._connect_existing_femtet(pid)
            except FemtetAutomationError:
                self._connect_new_femtet()

        else:
            raise Exception(f'定義されていない femtet 接続方法です：{connect_method}')

        # ensure makepy
        if not hasattr(constants, 'STATIC_C'):
            cmd = f'{sys.executable} -m win32com.client.makepy FemtetMacro'
            os.system(cmd)
            message = 'Femtet python マクロ定数の設定が完了してないことを検出しました.'
            message += '次のコマンドにより、設定は自動で行われました（python  -m win32com.client.makepy FemtetMacro）.'
            message += 'インタープリタを再起動してください.'
            raise Exception(message)

    def check_param_value(self, param_name):
        variable_names = self.Femtet.GetVariableNames()
        if param_name in variable_names:
            return self.Femtet.GetVariableValue(param_name)
        else:
            message = f'Femtet 解析モデルに変数 {param_name} がありません.'
            message += f'現在のモデルに設定されている変数は {variable_names} です.'
            message += '大文字・小文字の区別に注意してください.'
            raise Exception(message)

    def open(self, femprj_path: str, model_name: str or None = None) -> None:
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

    def update_model(self, parameters: 'pd.DataFrame') -> None:
        # 変数更新のための処理
        sleep(0.1)  # Gaudi がおかしくなる時がある対策
        self.Femtet.Gaudi.Activate()

        # Femtet の設計変数の更新
        for i, row in parameters.iterrows():
            name = row['name']
            value = row['value']
            if not (self.Femtet.UpdateVariable(name, value)):
                raise ModelError('変数の更新に失敗しました：変数{name}, 値{value}')

        # 設計変数に従ってモデルを再構築
        if not self.Femtet.Gaudi.ReExecute():
            raise ModelError('モデル再構築に失敗しました')

        # 処理を確定
        self.Femtet.Redraw()

    def solve(self) -> None:
        # メッシュを切る
        n_mesh = self.Femtet.Gaudi.Mesh()
        if n_mesh == 0:
            raise MeshError('メッシュ生成に失敗しました。')

        # ソルブする
        result = self.Femtet.Solve()
        if not result:
            raise SolveError('ソルブに失敗しました。')

        # 次に呼ばれるはずのユーザー定義コスト関数の記述を簡単にするため先に解析結果を開いておく
        result = self.Femtet.OpenCurrentResult(True)
        if not result:
            raise FemtetAutomationError('解析結果のオープンの失敗しました。')

    def update(self, parameters: 'pd.DataFrame') -> None:
        self.update_model(parameters)
        self.solve()

    def quit(self):
        self.Femtet.SaveProject(self.Femtet.Project, True)
        util.close_femtet(self.Femtet.hWnd)

    def parallel_setup(self, subprocess_idx):
        # .Result の干渉を回避するため、
        # サブプロセスならばプロジェクトを別名保存する
        self.td = tempfile.mkdtemp()
        print(self.td)
        name = f'subprocess{subprocess_idx}'
        femprj_path = os.path.join(self.td, f'{name}.femprj')
        result = self.Femtet.SaveProject(femprj_path, True)
        if not result:
            self.Femtet.ShowLastError()

    def parallel_terminate(self):
        try:
            # Femtet プロセスを終了する（自動保存しないよう保存する）
            print(f'try to close Femtet.exe')
            self.quit()

            # 一時ファイルを削除する
            sleep(1)
            try:
                shutil.rmtree(self.td)
            except PermissionError:
                pass  # 諦める
        except:
            pass  # 何かがあってもプロセス全体が正常終了することを優先


class NoFEM(FEMIF):

    def update(self, parameters):
        pass

    def check_param_value(self, param_name):
        pass

    def quit(self):
        pass
