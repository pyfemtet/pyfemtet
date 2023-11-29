# for hint
from __future__ import annotations
from typing import Any, Callable
from abc import ABC, abstractmethod
from warnings import warn

# built-in modules
import os
import sys
import time
import datetime
import inspect
# from multiprocessing import Pool, Process, Manager
import functools
import math
import ast

# 3rd-party modules
import numpy as np
import pandas as pd
from optuna._hypervolume import WFG

# for Femtet
import win32com.client
from win32com.client import Dispatch, constants, CDispatch
from pywintypes import com_error
from femtetutils import util, constant
from ..tools.DispatchUtils import Dispatch_Femtet_with_pid, Dispatch_Femtet_with_new_process



#### Exception for Femtet error
class ModelError(Exception):
    '''FEM でのモデル更新に失敗した際のエラー'''
    pass

class MeshError(Exception):
    '''FEM でのメッシュ生成に失敗した際のエラー'''
    pass

class SolveError(Exception):
    '''FEM でのソルブに失敗した際のエラー'''
    pass

class PostError(Exception):
    '''FEM 解析前後の目的又は拘束の計算に失敗した際のエラー'''
    pass

class FEMCrash(Exception):
    '''FEM プロセスが異常終了した際のエラー'''
    pass

class UserInterruption(Exception):
    '''ユーザーによる中断'''
    pass


#### FEM クラス

class FEMSystem(ABC):
    '''
    FEM システムのクラス。
    このクラスを継承して具体的な FEM システムのクラスを作成し、
    そのインスタンスを FemtetOptimizationCore の FEM メンバーに割り当てる。
    '''
    @abstractmethod
    def update(self, df)->None:
        '''
        FEM の制御を実装して df に基づいてモデルを更新し
        objectiveValues と constraintValues を更新する
        '''
        pass


class Femtet(FEMSystem):
        
    def open(self, femprj_path:str, model_name:str or None = None)->None:
        '''

        Parameters
        ----------
        femprj_path : str
            .femprj ファイルのパス.
        model_name : str or None, optional
            解析モデルの名前. 指定しない場合、プロジェクトを最後に開いたときのモデルが使われます。
            The default is None.

        Returns
        -------
        None
            DESCRIPTION.

        '''
        
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
    
        
    def _run_with_catch_error(self, function, iferror=False)->str:
        # Solve returns False is an error ocured in Femtet
        # Gaudi.Mesh returns number of elements or 0 if error
        # for example, _withcatcherror(Femtet.Solve, False), _withcatcherror(Femtet.Gaudi.Mesh, 0)
        result = function()
        if result==iferror:
            try:
                # ShowLastError() raises com_error
                # that contains the error message of Femtet
                self.Femtet.ShowLastError() # 必ずエラーが起こる
            except com_error as e:
                info:tuple = e.excepinfo
                error_message:str = info[2] # 例：'E1033 Model : モデルファイルの保存に失敗しました'
                error_code = error_message.split(' ')[0]
                # ただし、エラーの書式が全部" "区切りかは不明
                return error_code
                # 'E1033':モデル保存失敗
                # 'E4019':Femtetが開いてません
                # 例えば、モデルファイルの保存は再試行したらいい。
                # （なんで保存に失敗するかは謎...sleepしたほうがいいのか。）
        else:
            return ''
    
    
    def connect_Femtet(self, strategy:str='new')->None:
        '''
        Femtet プロセスとの接続を行います。

        Parameters
        ----------
        strategy : str, optional
            'new' or 'catch' or 'auto'. The default is 'new'.
            'new' のとき、新しい Femtet プロセスを起動し、接続します。
            'catch' のとき、既存の Femtet プロセスと接続します。
            ただし、接続できる Femtet が存在しない場合、エラーを送出します。
            この場合、既存 Femtet プロセスのどれに接続されるかは制御できません。
            また、すでに別のPython又はExcelプロセスと接続されている Femtet との接続を行うことはできません。
            'auto' のとき、まず 'catch' を試行し、失敗した際には 'new' で接続します。

        Raises
        ------
        Exception
            何らかの理由で Femtet との接続に失敗した際に例外を送出します。

        Returns
        -------
        caught_femtet : str.
            接続された Femtet が既存のものならば 'existing',
            新しいものならば 'new' を返します。

        '''
        
        caught_femtet = None
                
        if strategy=='new':
            # 新しい Femtet プロセスを立てて繋ぐ
            self.Femtet, _ = Dispatch_Femtet_with_new_process()
            caught_femtet = 'new'
        
        elif strategy=='catch':
            # 既存の Femtet を探して Dispatch する。
            self.Femtet, _ = Dispatch_Femtet_with_pid()
            caught_femtet = 'existing'

        elif strategy=='auto':
            try:
                # 既存の Femtet を探して Dispatch する。
                self.Femtet, pid = Dispatch_Femtet_with_pid()
                caught_femtet = 'existing'
                if pid==0:
                    # 新しい Femtet プロセスを立てて繋ぐ
                    self.Femtet, _ = Dispatch_Femtet_with_new_process()
                    caught_femtet = 'new'
            except:
                # 新しい Femtet プロセスを立てて繋ぐ
                self.Femtet, _ = Dispatch_Femtet_with_new_process()
                caught_femtet = 'new'
        
        else:
            raise Exception(f'不明な接続方法です：{strategy}')
        
        # makepy しているかどうかの確認
        if not hasattr(constants, 'STATIC_C'):
            cmd = f'{sys.executable} -m win32com.client.makepy FemtetMacro'
            os.system(cmd)
            message = 'Femtet の定数を使う設定が行われていなかったので、設定を自動で実行しました。設定を反映するため、Pythonコンソールを再起動してください。'
            raise Exception(message)
        
        return caught_femtet


    def update_model(self, df:pd.DataFrame)->None:
        '''
        Femtet の解析モデルを df に従って再構築します。

        Parameters
        ----------
        df : pd.DataFrame
            name, value のカラムを持つこと.

        Raises
        ------
        ModelError
            変数の更新またはモデル再構築に失敗した場合に送出されます。

        Returns
        -------
        None

        '''

        # 変数更新のための処理
        time.sleep(0.1) # Gaudi がおかしくなる時がある対策
        Gaudi = self.Femtet.Gaudi
        Gaudi.Activate()

        # Femtet の設計変数の更新
        for i, row in df.iterrows():
            name = row['name']
            value = row['value']
            if not (self.Femtet.UpdateVariable(name, value) ):
                raise ModelError('変数の更新に失敗しました：変数{name}, 値{value}')
            
        # 設計変数に従ってモデルを再構築
        # 失敗した場合、変数の更新が反映されない
        if not Gaudi.ReExecute():
            raise ModelError('モデル再構築に失敗しました')

        # インポートを含むモデルに対し ReExecute すると
        # 結果画面のボディツリーでボディが増える対策
        self.Femtet.Redraw()

        
    def run(self)->None:
        '''
        Femtet の解析を実行します。

        Raises
        ------
        MeshError
        SolveError

        Returns
        -------
        None.

        '''

        # メッシュを切る
        error_code = self._run_with_catch_error(self.Femtet.Gaudi.Mesh)
        if error_code!='':
            raise MeshError('メッシュ生成に失敗しました。')
        
        # ソルブする
        error_code = self._run_with_catch_error(self.Femtet.Solve)
        if error_code!='':
            raise SolveError('ソルブに失敗しました。')

        # 次に呼ばれるはずのユーザー定義コスト関数の記述を簡単にするため先に解析結果を開いておく
        if not(self.Femtet.OpenCurrentResult(True)):
            self.Femtet.ShowLastError

    
    def update(self, df:pd.DataFrame)->None:
        '''
        update_model と run を実行します。

        Parameters
        ----------
        df : pd.DataFrame
            name, value のカラムを持つこと.

        '''
        self.update_model(df)
        self.run()


class NoFEM(FEMSystem):
    '''
    デバッグ用クラス。
    '''
    
    def __init__(self):
        self.error_regions = []

    def set_error_region(self, f):
        '''
        モデルエラーを起こすための関数 f[[np.ndarray], float] をセットする。
        '''
        self.error_regions.append(f)
    
    def update(self, df):
        '''
        自身にセットされたerror_regionに抵触する場合、ModelErrorを送出する。
        '''
        for f in self.error_regions:
            if f(df['value'].values)<0:
                raise ModelError


#### 周辺クラス、関数

def _is_access_Gaudi(f):
    '''
    ユーザー定義関数が Gaudi にアクセスしているかどうかを簡易的に判断する。
    solve前に評価するstrict_constraintでsolve結果にアクセスする違反を
    検出するためのfool-proof策。
    '''
    # 関数fのソースコードを取得
    source = inspect.getsource(f)
    
    # ソースコードを抽象構文木（AST）に変換
    tree = ast.parse(source)
    
    # 関数定義を見つける
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            # 関数の第一引数の名前を取得
            first_arg_name = node.args.args[0].arg
            
            # 関数内の全ての属性アクセスをチェック
            for sub_node in ast.walk(node):
                if isinstance(sub_node, ast.Attribute):
                    # 第一引数に対して 'Gogh' へのアクセスがあるかチェック
                    if (
                            isinstance(sub_node.value, ast.Name)
                            and sub_node.value.id == first_arg_name
                            and sub_node.attr == 'Gogh'
                            ):
                        return True
            # ここまできてもなければアクセスしてない
            return False


def symlog(x):
    '''
    定義域を負領域に拡張したlog関数です。
    多目的最適化における目的関数同士のスケール差により
    意図しない傾向が生ずることのの軽減策として
    内部でsymlog処理を行います。
    '''
    if isinstance(x, np.ndarray):
        ret = np.zeros(x.shape)
        idx = np.where(x>=0)
        ret[idx] = np.log10(x[idx] + 1)
        idx = np.where(x<0)
        ret[idx] = -np.log10(1 - x[idx])
    else:
        if x>=0:
            ret = np.log10(x + 1)
        else:
            ret = -np.log10(1 - x)
        
    return ret
    

class Objective:
    prefixForDefault = 'objective'

    def __init__(self, f, args, kwargs, name, direction, FEMOpt):
        
        self._checkDirection(direction)
        
        self.f = f
        self.args = args
        self.kwargs = kwargs

        self.name = name
        self.direction = direction
        
        self.FEMOpt = FEMOpt


    def _checkDirection(self, direction):
        if type(direction)==float or type(direction)==int:
            pass
        elif direction=='minimize':
            pass
        elif direction=='maximize':
            pass
        else:
            raise Exception(f'invalid direction "{direction}" for Objective')
        

    def _convert(self, value:float):
        '''目的関数の値を direction を加味した内部値に変換します。'''
        ret = value
        if type(self.direction)==float or type(self.direction)==int:
            ret = abs(value - self.direction)
        elif self.direction=='minimize':
            ret = value
        elif self.direction=='maximize':
            ret = -value

        ret = symlog(ret)

        return ret         
                

    def calc(self):
        # Femtet を継承したクラスの場合第一引数は Femtet であるとする
        if issubclass(self.FEMOpt.FEMClass, Femtet):
            return self.f(*(self.FEMOpt.FEM.Femtet, *self.args), **self.kwargs)
        else:
            return self.f(*self.args, **self.kwargs)
            

        
class Constraint:
    prefixForDefault = 'constraint'

    def __init__(self, f, args, kwargs, name, lb, ub, strict, FEMOpt):
        self.f = f
        self.args = args
        self.kwargs = kwargs

        self.name = name
        self.lb = lb
        self.ub = ub
        self.strict = strict
        
        self.FEMOpt = FEMOpt


    def calc(self):
        # Femtet を継承したクラスの場合第一引数は Femtet であるとする
        if issubclass(self.FEMOpt.FEMClass, Femtet):
            return self.f(*(self.FEMOpt.FEM.Femtet, *self.args), **self.kwargs)
        else:
            return self.f(*self.args, **self.kwargs)
        

#### 主要抽象クラス

class FemtetOptimizationCore(ABC):
    
    def __init__(
            self,
            femprj_path:str or None,
            model_name:str or None = None,
            result_path:str=None,
            FEMClass:FEMSystem = Femtet,
            ):
        
        # 引数の処理
        self.femprj_path = os.path.abspath(femprj_path)
        self.model_name = model_name
        self.history_path = result_path
        self.FEMClass = FEMClass

        #### Femtet に接続
        if self.FEMClass==Femtet:
            self.FEM = Femtet()
            caught_femtet = self.FEM.connect_Femtet('auto') # 既存のものがあれば新しいのを開かない
            if caught_femtet=='new':
                # 新しい Femtet を開いたのに femprj がない場合
                if femprj_path is None:
                    print('-----')
                    print('.femprj ファイルのパスを指定してください。')
                    femprj_path = input('>>> ')
                    self.FEM.open(femprj_path)
                    print('-----')
                    print('Femtet で目的の解析モデルを開き、この画面で Enter を押してください。')
                    femprj_path = input('Enter to continue...')
                # 新しい Femtet で指定された femprj を開く
                else:
                    self.FEM.open(femprj_path, model_name)
            else: # existing の場合
                # femprj などが指定されていない場合、existing を基にする
                if self.femprj_path is None:
                    self.femprj_path = os.path.abspath(self.FEM.Femtet.Project)
                    self.model_name = self.FEM.Femtet.AnalysisModelName
                else:
                    # 開いている femtet の prj が指定された情報と一致しなかった場合、
                    # プログラムを優先する...と言いたいところだが、開いている Femtet の作業内容が失われる可能性があるので
                    # Exception を送出する
                    if self.femprj_path!=self.FEM.Femtet.Project:
                        raise Exception('開いている Femtet に接続しましたが、プログラムで指定されているプロジェクトとは違うプロジェクトが開かれています。正しいプロジェクトを開き、Pythonプロセスを再起動してください。PythonからはどのFemtetプロセスに接続するかを制御できないため、目的のFemtetに確実に接続するためには、目的のプロジェクトを開いている以外のFemtetプロセスをすべて終了し、そのプロセスが他のマクロプロセスに接続されていない状態にしてください。')
                    else:
                        if self.model_name is not None:
                            raise Exception('開いている Femtet に接続しましたが、プログラムで指定されている解析モデルとは違うモデルが開かれています。正しい解析モデルを開き、Pythonプロセスを再起動してください。')
                
        else:
            self.FEM = self.FEMClass()
        
        
        #### メンバーの宣言
        # ヒストリパスの設定
        if self.history_path is None:
            name = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')
            self.history_path = os.path.join(os.getcwd(), f'{name}.csv')
        # member
        self.objectives = []
        self.constraints = []
        # abstract member
        self._objectives = []
        self._constraints = []
        # per main()
        self.last_execution_time = -1
        self.process_monitor = None
        self.interruption = False
        self.history = None # parameter などが入ってからでないと初期化もできないので
        # temporal variables per solve
        self._objective_values = None
        self._constraint_values = None
        self._optimization_message = ''
        self._parameter = pd.DataFrame(
            columns=[
                'name',
                'value',
                'lbound',
                'ubound',
                'memo'
                ],
            dtype=object
            )

    #### マルチプロセスで pickle するときに COM を持っていても仕方がないので消す
    def __getstate__(self):
        state = self.__dict__.copy()
        del state['FEM']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    # 子プロセスで FEM インスタンスを作成することを想定
    # import 元が干渉するため NX_Femtet と SW_Femtet はマルチプロセスにできない
    # この問題さえクリアすれば、即ち femprj と x_t を分ければマルチプロセスは可能
    # ただし SW は複数プロセスの COM 制御が不可能なのでウィンドウ単位で制御するとか工夫が必要
    def set_FEM(self):
        self.FEM = self.FEMClass()
        if self.FEMClass==Femtet:
            self.FEM.connect_Femtet('new')
            self.FEM.open(self.femprj_path, self.model_name)
    

    #### pre-processing methods

    def add_objective(
            self,
            fun:Callable[[CDispatch, Any], float],
            name:str or None = None,
            direction:str or float = 'minimize',
            args:tuple or None = None,
            kwargs:dict or None = None
            ):
        # 引数の処理
        if args is None:
            args = tuple()
        elif type(args)!=tuple:
            args = (args,)
        if kwargs is None:
            kwargs = dict()
        
        # name が指定されなかった場合に自動生成
        if name is None:
            name = self._getUniqueDefaultName(Objective.prefixForDefault, self.objectives)

        # objective オブジェクトを生成
        obj = Objective(fun, args, kwargs, name, direction, self) # サブプロセス内で引っかからないか？

        # 被っているかどうかを判定し上書き or 追加
        isExisting = False
        for i, obj in enumerate(self.objectives):
            if obj.name==name:
                isExisting = True
                warn('すでに登録されている名前が追加されました。上書きされます。')
                self.objectives[i] = obj
        if not isExisting:
            self.objectives.append(obj)


    def add_constraint(
            self,
            fun:Callable[[CDispatch, Any], float],
            name:str or None = None,
            lower_bound:float or None = None,
            upper_bound:float or None = None,
            strict:bool = True,
            args:tuple or None = None,
            kwargs:dict or None = None,
            ):
        # 引数の処理
        if args is None:
            args = tuple()
        elif type(args)!=tuple:
            args = (args,)
        if kwargs is None:
            kwargs = dict()

        # strict constraint の場合、solve 前に評価したいので Gaudi へのアクセスを禁ずる
        if strict:
            if _is_access_Gaudi(fun):
                raise Exception(f'関数{fun.__name__}に Gaudi へのアクセスがあります。拘束の計算において解析結果にアクセスするには、strict = False を指定してください。')
                
        # name が指定されなかった場合に自動生成
        if name is None:
            name = self._getUniqueDefaultName(Constraint.prefixForDefault, self.constraints)
        
        # Constraint オブジェクトの生成
        cns = Constraint(fun, args, kwargs, name, lower_bound, upper_bound, strict, self)

        # 被っているかどうかを判定し上書き or 追加
        isExisting = False
        for i, cns in enumerate(self.constraints):
            if cns.name==name:
                isExisting = True
                warn('すでに登録されている名前が追加されました。上書きされます。')
                self.constraints[i] = cns
        if not isExisting:
            self.constraints.append(cns)


    def add_parameter(
            self,
            name:str,
            initial_value:float or None = None,
            lower_bound:float or None = None,
            upper_bound:float or None = None,
            memo:str = ''
            ):
  
        # 変数が Femtet にあるかどうかのチェック
        if type(self.FEM)==Femtet: # 継承した NX_Femtet とかでは Femtet には変数がない
            variable_names = self.FEM.Femtet.GetVariableNames()
            if name in variable_names:
                femtetValue = self.FEM.Femtet.GetVariableValue(name)
            else:
                raise Exception(f'Femtet プロジェクトに変数{name}が設定されていません。現在の Femtet で登録されいてる変数は {variable_names} です。大文字・小文字の区別の注意してください。')
        
        if initial_value is None:
            initial_value = femtetValue

        d = {
            'name':name,
            'value':initial_value,
            'lbound':lower_bound,
            'ubound':upper_bound,
            'memo':memo,
                }
        df = pd.DataFrame(d, index=[0], dtype=object)
        
        if len(self._parameter)==0:
            newdf = df
        else:
            newdf = pd.concat([self._parameter, df], ignore_index=True)
        
        if self._isDfValid(newdf):
            self._parameter = newdf
        else:
            raise Exception('パラメータの設定が不正です。')


    def set_parameter(self, df:pd.DataFrame):
        
        if self._isDfValid(df):
            self._parameter = df
        else:
            raise Exception('パラメータの設定が不正です。')
    

    def get_current_parameter(self, format:str='dict'):
        '''
        現在の変数セットを返します。

        Parameters
        ----------
        format : str, optional
            parameter format.  'df', 'values' or 'dict'.
            The default is 'dict'.

        Raises
        ------
        Exception
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        '''
        if format=='df':
            return self._parameter
        elif format=='values' or format=='value':
            return self._parameter.value.values
        elif format=='dict':
            ret = {}
            for i, row in self._parameter.iterrows():
                ret[row['name']] = row.value
            return ret
        else:
            fname = getCurrentMethod()
            raise Exception('get_current_parameter() got invalid format: {format}')


    def _set_process_monitor(self):
        # main の直前、parameter 等のセッティングが終わった後に呼ぶ
        if self.history is None:
            self._init_history()
        from .visualization._dash import DashProcessMonitor
        self.process_monitor = DashProcessMonitor(self)
        self.process_monitor.start()
        

    def _isDfValid(self, df):
        # column が指定の属性をすべて有しなければ invalid
        if not all([name in df.columns for name in ['name', 'value', 'ubound', 'lbound']]):
            return False
        
        # value が float でなければ invalid
        for value in df.value.values:
            try:
                float(value)
            except:
                return False
        
        # ubound または lbound が float 又は None でなければ invalid
        for value in df.ubound.values:
            if value is None:
                continue
            try:
                float(value)
            except:
                return False
            
        for value in df.lbound.values:
            if value is None:
                continue
            try:
                float(value)
            except:
                return False

        # value, ubound と lbound の上下関係がおかしければ invalid
        for i, row in df.iterrows():
            v = row['value']
            l = row['lbound']
            u = row['ubound']
            
            if l is None:
                l = v - 1
            
            if u is None:
                u = v + 1

            if not(l <= v <= u):
                return False
        
        return True


    def _getUniqueDefaultName(self, prefix, objects):
        names = [obj.name for obj in objects]
        i = 0
        while True:
            name = f'{prefix}_{i}'
            if not(name in names):
                break
            i += 1
        return name
                

    #### processing methods

    def f(self, x:np.ndarray, *args, **kwargs)->[float]:
        '''
        現在のパラメータセットを np.ndarray として受け取り、
        FEM 計算を実行して目的関数の内部値のリストを返します。
        この関数は具象クラスの _main、即ち最適化ライブラリの最適化関数に渡されることを想定しています。
        内部値は、与えられた目的関数が最小化問題となるよう変換された値のsymlogです。

        Parameters
        ----------
        x : np.ndarray
            最適化アルゴリズムによって提案された "次の" パラメータセットです。
            get_current_parameter('values') の値とは一致しません。

        Raises
        ------
        UserInterruption
            ユーザーによる中断指令があった際に送出されます。
            その時点で実行中の解析がすべて終了した時点で最適化が終了します。

        Returns
        -------
        [float]
            変換された目的関数の内部値です。

        '''
        
        # 中断指令があったら Exception を起こす
        if self.interruption:
            raise UserInterruption('ユーザーによって最適化が取り消されました。')
        
        # 渡された x がすでに計算されたものであれば
        # objective も constraint も更新する必要がなく
        # そのまま converted objective を返せばいい
        if not self._isCalculated(x):
            # メッセージの更新
            if 'message' in list(kwargs.keys()):
                self._optimization_message = kwargs['message']
            else:
                self._optimization_message = ''
            
            # update parameters
            self._parameter['value'] = x

            # solve(Exception は当面 Optuna で処理する)
            self.FEM.update(self._parameter)
            self._objective_values = [obj.calc() for obj in self.objectives]
            self._constraint_values = [cns.calc() for cns in self.constraints]
            self._record()
                
        return [obj._convert(v) for obj, v in zip(self.objectives, self._objective_values)]


    def _isCalculated(self, x):
        # 提案された x が最後に計算したものと一致していれば True, ただし 1 回目の計算なら False
        #    ひとつでも違う  1回目の計算  期待
        #    True            True         False
        #    False           True         False
        #    True            False        False
        #    False           False        True
        
        # ひとつでも違う
        condition1 = False
        for _x, _p in zip(x, self._parameter['value'].values):
            condition1 = condition1 or (float(_x)!=float(_p))
        # 1回目の計算
        condition2 = len(self.history)==0
        return not(condition1) and not(condition2)


    @abstractmethod
    def _main(self):
        pass
    

    def main(self, history_path:str or None = None, n_parallel=1):
        '''
        最適化を実行する。

        Parameters
        ----------
        history_path : str or None, optional
            実行結果を保存する csv パス.
        n_parallel : int
            Femtet の並列実行数。1以上の値を指定すること。

        Raises
        ------
        Exception
            add_objective または add_parameter などが呼ばれていない場合に例外を送出します.

        Returns
        -------
        None

        '''
        #### 最適化プロセスの実行前チェック
        # 変数のチェック
        if ((len(self.objectives)==0)
            or (len(self._parameter)==0)):
            raise Exception('パラメータまたは目的関数が設定されていません。')

        # ヒストリの初期化
        self._init_history()

        # 中断指令の初期化
        self.interruption = False
        
        # プロセスモニタの初期化
        # self._set_process_monitor()

        #### 最適化の開始        
        start = time.time()
        self._main()
        end = time.time()
        self.last_execution_time = end - start # 秒


    # post-processing methods    

    def _genarateHistoryColumnNames(self):
        paramNames = [p for p in self._parameter['name'].values]
        objNames = [obj.name for obj in self.objectives]
        _consNames = [obj.name for obj in self.constraints]
        consNames = []
        for n in _consNames:
            consNames.extend([f'{n}_lower_bound', n, f'{n}_upper_bound', f'{n}_fit'])
        return paramNames, objNames, consNames


    def get_history_columns(self, kind:str='objective')->[str]:
        '''
        get history column names. 

        Parameters
        ----------
        kind : str, optional
            valid kind is: "objective", "parameter", "constraint". The default is 'objective'.

        Returns
        -------
        [str]
            list of column names.

        '''
        p, o, c = self._genarateHistoryColumnNames()
        if kind == 'parameter':
            return p
        elif kind == 'objective':
            return o
        elif kind == 'constraint':
            if len(c)==0:
                return []
            else:
                return c[1::4]
        else:
            raise Exception(f'invalid kind: {kind}; kind must be "parameter", "objective" or "constraint"')
            
    def _init_history(self):
        columns = []
        columns.append('n_trial')
        pNames, oNames, cNames = self._genarateHistoryColumnNames()
        columns.extend(pNames)
        columns.extend(oNames)
        columns.extend(cNames)
        columns.append('non_domi')
        columns.append('fit')
        columns.append('hypervolume')
        columns.append('optimization_message')
        columns.append('time')
        self.history = pd.DataFrame(columns=columns)
    
    def _record(self):
        # record の計算
        row = []
        paramValues = list(self._parameter['value'].copy().values)
        objValues = self._objective_values
        consValues = self._constraint_values
        consLbList = [constraint.lb for constraint in self.constraints]
        consUbList = [constraint.ub for constraint in self.constraints]
        consData = []
        for v, lb, ub in zip(consValues, consLbList, consUbList):
            consData.extend([lb, v, ub])
            lb = -np.inf if lb is None else lb
            ub = np.inf if ub is None else ub
            fit = (lb <= v <= ub)
            consData.extend([fit])
        row.append(len(self.history)+1)
        row.extend(paramValues)
        row.extend(objValues)
        row.extend(consData)
        row.append(False) # non_domi, temporal data
        row.append(False) # fit, temporal data
        row.append(np.nan) # hypervolume, temporal data
        row.append(self._optimization_message)
        row.append(datetime.datetime.now()) # time

        # list を series に変換
        pdf = pd.Series(row, index=self.history.columns)

        # 拘束を満たすかどうか
        fit = self._calcConstraintFit(pdf)
        pdf['fit'] = fit
        # series を history に追加
        self.history.loc[len(self.history)] = pdf

        # 履歴に依存する項目の計算
        # update non-domi(after addition to history)
        self._calcNonDomi()
        # calc hypervolume(after addition to history)
        self._calc_hypervolume()

        # 保存
        try:
            self.history.to_csv(self.history_path, index=None, encoding='shift-jis')
        except PermissionError:
            # excel で開くなど permission error が出る可能性がある。
            # その場合は単にスキップすれば、次の iteration ですべての履歴が保存される
            pass

            
    def _calcNonDomi(self):
        '''non-dominant 解を計算する'''

        # 目的関数の履歴を取り出してくる
        _, oNames, _ = self._genarateHistoryColumnNames()
        _objectiveValues = self.history[oNames].copy()

        # 最小化問題の座標空間に変換する
        for objective in self.objectives:
            name = objective.name
            _objectiveValues[name] = _objectiveValues[name].map(objective._convert)        

        # 非劣解の計算
        non_domi = []
        for i, row in _objectiveValues.iterrows():
            non_domi.append((row > _objectiveValues).product(axis=1).sum(axis=0) == 0)
            
        # 非劣解の登録
        self.history['non_domi'] = non_domi

        del _objectiveValues
        

    def _calcConstraintFit(self, pdf):
        '''constraint fit 解を計算する'''

        # 各拘束に対し fit しているかどうかの値を取り出してくる
        _, _, cNames = self._genarateHistoryColumnNames()
        fit = np.all(pdf[[cName for cName in cNames if '_fit' in cName]])
        
        return fit


    def _calc_hypervolume(self):
        '''
        hypervolume 履歴を更新する
        ※ reference point が変わるたびに hypervolume を計算しなおす必要がある
        [1]Hisao Ishibuchi et al. "Reference Point Specification in Hypercolume Calculation for Fair Comparison and Efficient Search"
        '''
        #### 前準備
        # パレート集合の抽出
        idx = self.history['non_domi']
        pdf = self.history[idx]
        parate_set = pdf[self.get_history_columns('objective')].values
        n = len(parate_set) # 集合の要素数
        m = len(parate_set.T) # 目的変数数
        # 長さが 2 以上でないと計算できない
        if n<=1:
            return np.nan
        # 最小化問題に convert
        for i, objective in enumerate(self.objectives):
            for j in range(n):
                parate_set[j,i] = objective._convert(parate_set[j,i])        
        #### reference point の計算[1]
        # 逆正規化のための範囲計算
        maximum = parate_set.max(axis=0)
        minimum = parate_set.min(axis=0)
        # (H+m-1)C(m-1) <= n <= (m-1)C(H+m) になるような H を探す
        H = 0
        while True:
            left = math.comb(H+m-1, m-1)
            right = math.comb(H+m, m-1)
            if left <= n <= right:
                break
            else:
                H += 1
        # H==0 なら r は最大の値
        if H==0:
            r = 2
        else:
            # r を計算
            r = 1 + 1./H
        r = 1.01
        # r を逆正規化
        reference_point = r * (maximum - minimum) + minimum
        
        #### hv 履歴の計算
        wfg = WFG()
        hvs = []
        for i in range(n):
            hv = wfg.compute(parate_set[:i], reference_point)
            hvs.append(hv)
        
        # 計算結果を履歴の一部に割り当て
        df = self.history
        df.loc[idx, 'hypervolume'] = np.array(hvs)

        # dominated の行に対して、上に見ていって
        # 最初に見つけた non-domi 行の hypervolume の値を割り当てます
        for i in range(len(df)):
            if df.loc[i, 'non_domi'] == False:
                try:
                    df.loc[i, 'hypervolume'] = df.loc[:i][df.loc[:i]['non_domi']].iloc[-1]['hypervolume']
                except IndexError:
                    pass # nan のままにする
        
    