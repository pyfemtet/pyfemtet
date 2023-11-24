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
def getCurrentMethod():
    return inspect.currentframe().f_code.co_name
import threading
import functools
import math
import ast

# 3rd-party modules
import numpy as np
import pandas as pd
from optuna._hypervolume import WFG

# for Femtet
import win32com.client
from win32com.client import Dispatch, constants
from pywintypes import com_error
from femtetutils import util, constant



#### Exception for Femtet error
class ModelError(Exception):
    _message = 'モデル作成に失敗しました。目的関数と拘束関数の値は意味を持ちません。'

class MeshError(Exception):
    _message = 'メッシュ作成に失敗しました。目的関数と拘束関数の値は意味を持ちません。'

class SolveError(Exception):
    _message = 'ソルブに失敗しました。目的関数と拘束関数の値は意味を持ちません。'

class UserInterruption(Exception):
    _message = 'ユーザー操作によって中断されました'


#### FEM クラス

class FEMSystem(ABC):
    '''FEM システムのクラス。
    このクラスを継承して具体的な FEM システムのクラスを作成し、
    そのインスタンスを FemtetOptimizationCore の FEM メンバーに割り当てる。
    '''
    @abstractmethod
    def run(self, df)->None:
        '''
        FEM の制御を実装して df に基づいてモデルを更新し
        objectiveValues と constraintValues を更新する
        '''
        pass


class Femtet(FEMSystem):
    
    def _catchError(self):
        raise NotImplementedError()

        # Solve returns False is an error ocured in Femtet
        # Gaudi.Mesh returns number of elements or 0 if error
        if not Femtet.Solve():
            try:
                # ShowLastError() raises com_error
                # that contains the error message of Femtet
                self.Femtet.ShowLastError()
            except com_error as e:
                info:tuple = e.excepinfo
                error_message:str = info[2] # 例：'E1033 Model : モデルファイルの保存に失敗しました'
                error_code = error_message.split(':')[0].split(' ')[0]
                # 例えば、モデルファイルの保存は再試行したらいい。
                # （なんで保存に失敗するかは謎...sleepしたほうがいいのか。）
                # ただし、エラーの書式が全部":"と" "区切りかは不明
                if error_code=='E1033': # モデル保存失敗
                    raise ModelError
    
    def setFemtet(self, strategy='auto'):
        if strategy=='catch':
            # 既存の Femtet を探す
            self.Femtet = Dispatch('FemtetMacro.Femtet')
        elif strategy=='new':
            # femtetUtil を使って新しい Femtet を立てる
            succeed = util.execute_femtet()
            if succeed:
                self.Femtet = Dispatch('FemtetMacro.Femtet')
        elif strategy=='auto':
            succeed = util.auto_execute_femtet()
            if succeed:
                self.Femtet = Dispatch('FemtetMacro.Femtet')
        
        if not hasattr(constants, 'STATIC_C'):
            cmd = f'{sys.executable} -m win32com.client.makepy FemtetMacro'
            os.system(cmd)
            message = 'Femtet の定数を使う設定が行われていなかったので、設定を自動で実行しました。設定を反映するため、Pythonコンソールを再起動してください。'
            raise Exception(message)

    def update_model(self, df):
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
        
    def run(self, df):
        '''run : 渡された df に基づいて Femtet に計算を実行させる関数。

        '''
        self.update_model(df)
        
        Gaudi = self.Femtet.Gaudi

        # メッシュを切る
        try:
            Gaudi.Mesh()
        except win32com.client.pythoncom.com_error:
            raise MeshError('error was occured at meshing updated model')
        
        # ソルブする
        try:
            self.Femtet.Solve()
        except win32com.client.pythoncom.com_error:
            raise SolveError('error was occured at solver')

        # 次に呼ばれるはずのユーザー定義コスト関数の記述を簡単にするため先に解析結果を開いておく
        if not(self.Femtet.OpenCurrentResult(True)):
            self.Femtet.ShowLastError


class NoFEM(FEMSystem):
    def __init__(self):
        self.error_regions = []

    def set_error_region(self, f):
        self.error_regions.append(f)
    
    def run(self, df):
        for f in self.error_regions:
            if f(df['value'].values)<0:
                raise ModelError


#### 周辺クラス、関数

def is_access_Gaudi(f):
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
    def __init__(self, fun, direction, name, opt, ptrFunc, args, kwargs):
        
        self._checkDirection(direction)
        
        self.fun = fun
        self.direction = direction
        self.name = name
        
        self.ptrFunc = ptrFunc
        self.args = args
        self.kwargs = kwargs
        

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
        ret = value
        if type(self.direction)==float or type(self.direction)==int:
            ret = abs(value - self.direction)
        elif self.direction=='minimize':
            ret = value
        elif self.direction=='maximize':
            ret = -value

        ret = symlog(ret)

        return ret         
                
        
class Constraint:
    prefixForDefault = 'constraint'
    def __init__(self, fun, lb, ub, name, strict, opt, ptrFunc, args, kwargs):
        self.fun = fun # partial object
        self.lb = lb
        self.ub = ub
        self.name = name
        self.strict = strict

        self.ptrFunc = ptrFunc # original function
        self.args = args
        self.kwargs = kwargs
        

#### core クラス    
class FemtetOptimizationCore(ABC):
    
    def __init__(self, setFemtetStrategy:str or FEMSystem = 'auto'):
        

        # 初期化
        self.FEM = None
        self.objectives = []
        self.constraints = []
        self._objectives = [] # abstract member
        self._constraints = [] # abstract member
        self.objectiveValues = None
        self.constraintValues = None
        self.history = None
        self.historyPath = None
        self.processMonitor = None
        self.continuation = False # 現在の結果に上書き
        self.interruption = False
        self.algorithm_message = ''
        self.error_message = ''
        
        # 初期値
        if type(setFemtetStrategy) is str:
            self.FEM = Femtet()
            self.FEM.setFemtet(setFemtetStrategy)
        elif isinstance(setFemtetStrategy, FEMSystem):
            self.FEM = setFemtetStrategy


        # 初期値
        self.parameters = pd.DataFrame(
            columns=[
                'name',
                'value',
                'lbound',
                'ubound',
                'memo'
                ],
            dtype=object
            )

    # マルチプロセスなどで pickle するときに COM を持っていても仕方がないので消す
    def __getstate__(self):
        state = self.__dict__.copy()
        del state['FEM']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)


    #### pre-processing methods


    def add_objective(self,
                      fun:Callable[[Any], float],
                      name:str or None = None,
                      direction:str or float = 'minimize',
                      args:tuple or None = None,
                      kwargs:dict or None = None):

        f, name, args, kwargs = self._setupFunction(fun, args, kwargs, name, Objective.prefixForDefault, self.objectives)
        
        obj = Objective(f, direction, name, self, fun, args, kwargs)

        isExisting = False
        for i, objective in enumerate(self.objectives):
            if objective.name==name:
                isExisting = True
                # warn('すでに登録されている名前が追加されました。上書きされます。')
                self.objectives[i] = obj
        
        if not isExisting:
            self.objectives.append(obj)


    

    
    def add_constraint(self,
                      fun:Callable[[Any], float],
                      name:str or None = None,
                      lower_bound:float or None = None,
                      upper_bound:float or None = None,
                      strict:bool = True,
                      args:tuple or None = None,
                      kwargs:dict or None = None,
                      ):

        if strict:
            if is_access_Gaudi(fun):
                raise Exception(f'関数{fun.__name__}に Gaudi へのアクセスがあります。拘束の計算において解析結果にアクセスするには、strict = False を指定してください。')
                

        f, name, args, kwargs = self._setupFunction(fun, args, kwargs, name, Constraint.prefixForDefault, self.constraints)

        obj = Constraint(f, lower_bound, upper_bound, name, strict, self, fun, args, kwargs)

        isExisting = False
        for i, constraint in enumerate(self.constraints):
            if constraint.name==name:
                isExisting = True
                # warn('すでに登録されている名前が追加されました。上書きされます。')
                self.constraints[i] = obj
        
        if not isExisting:
            self.constraints.append(obj)

    def add_parameter(self,
                      name:str,
                      initial_value:float or None = None,
                      lower_bound:float or None = None,
                      upper_bound:float or None = None,
                      memo:str = ''
                      ):
        
        # 変数が Femtet にあるかどうかのチェック
        if type(self.FEM)==Femtet:
            try:
                femtetValue = self.FEM.Femtet.GetVariableValue(name)
            except:
                raise Exception(f'Femtet から変数 {name} を取得できませんでした。Femtet で開いている解析モデルが変数 {name} を含んでいるか確認してください。')
        
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
        
        if len(self.parameters)==0:
            newdf = df
        else:
            newdf = pd.concat([self.parameters, df], ignore_index=True)
        
        if self._isDfValid(newdf):
            self.parameters = newdf
        else:
            raise Exception('パラメータの設定が不正です。')

    def set_parameter(self, df:pd.DataFrame):
        
        if self._isDfValid(df):
            self.parameters = df
        else:
            raise Exception('パラメータの設定が不正です。')
    
    def get_parameter(self, format:str='dict'):
        '''
        

        Parameters
        ----------
        format : str, optional
            parameter format.  'df', 'value' or 'dict'.
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
            return self.parameters
        elif format=='value':
            return self.parameters.value.values
        elif format=='dict':
            ret = {}
            for i, row in self.parameters.iterrows():
                ret[row['name']] = row.value
            return ret
        else:
            fname = getCurrentMethod()
            raise Exception(f'{fname} got invalid format: {format}')

    def set_process_monitor(self, ProcessMonitorClass=None):
        if self.history is None:
            self._initRecord()
        
        if ProcessMonitorClass is None:
            if len(self.objectives)==1:
                from .visualization import SimpleProcessMonitor
                self.processMonitor = SimpleProcessMonitor(self)
            elif len(self.objectives)>1:
                from .visualization import UpdatableSuperFigure
                from .visualization import HypervolumeMonitor
                from .visualization import MultiobjectivePairPlot
                # self.processMonitor = HypervolumeMonitor(self)
                self.processMonitor = UpdatableSuperFigure(
                    self,
                    HypervolumeMonitor,
                    MultiobjectivePairPlot
                    )
        else:
            self.processMonitor = ProcessMonitorClass(self)
            

    def _setupFunction(self, fun, args, kwargs, name, prefix, objects):
        if args is None:
            args = ()
        elif type(args)!=tuple:
            args = (args,)

        if kwargs is None:
            kwargs = {}

        #### Femtetを使う場合は、第一引数に自動的に Femtet の COM オブジェクトが入っていることとする。
        if isinstance(self.FEM, Femtet):
            f = functools.partial(fun, *(self.FEM.Femtet, *args), **kwargs)
        else:
            f = functools.partial(fun, *args, *kwargs)
            
        
        if name is None:
            name = self._getUniqueDefaultName(prefix, objects)
        
        return f, name, args, kwargs

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

    def f(self, x:np.ndarray, algorithm_message='')->[float]:
        # 具象クラスの _main に呼ばれる関数
        
        # 中断指令があったら Exception を起こす
        if self.interruption:
            raise UserInterruption
        
        # 渡された x がすでに計算されたものであれば
        # objective も constraint も更新する必要がなく
        # そのまま converted objective を返せばいい
        if not self._isCalculated(x):
            # メッセージの初期化
            self.error_message = ''
            self.algorithm_message = algorithm_message
            
            # update parameters
            self.parameters['value'] = x

            # solve
            if type(self.FEM)==NoFEM:
                self.objectiveValues = [obj.fun() for obj in self.objectives]
                self.constraintValues = [obj.fun() for obj in self.constraints]
            else:
                self.FEM.run(self.parameters)
                self.objectiveValues = [obj.fun() for obj in self.objectives]
                self.constraintValues = [obj.fun() for obj in self.constraints]
            self._record()
                
        return [obj._convert(v) for obj, v in zip(self.objectives, self.objectiveValues)]

    def _isCalculated(self, x):
        #    ひとつでも違う  1回目の計算  期待
        #    True            True         False
        #    False           True         False
        #    True            False        False
        #    False           False        True
        
        # ひとつでも違う
        condition1 = False
        for _x, _p in zip(x, self.parameters['value'].values):
            condition1 = condition1 or (float(_x)!=float(_p))
        # 1回目の計算
        condition2 = len(self.history)==0
        return not(condition1) and not(condition2)

    @abstractmethod
    def _main(self):
        # 最適化を実施する
        # f を何度も呼ぶことを想定
        pass

    def main(self, historyPath:str or None = None):
        '''
        最適化を実行する。

        Parameters
        ----------
        historyPath : str or None, optional
            実行結果を保存する csv パス.
            None を指定した場合、以下の挙動となります。
            1. このインスタンスで初めての最適化計算を行う場合->カレントディレクトリに実行時刻に基づくファイルを自動で生成します。
            2. すでにこのインスタンスで計算が行われていて、再計算を行っている場合->既存のファイルを上書きします。

        Raises
        ------
        Exception
            add_objective または add_parameter などが呼ばれていない場合に例外を送出します.

        Returns
        -------
        result : OptimizationResult or Study
            使用したアルゴリズムによって、最適化結果のデータ型が異なります.このクラスのメンバー history はアルゴリズムに依らず一貫したデータ型の結果を格納します。

        '''
        # 一度だけ呼ばれる。何度も呼ぶのは f。
        
        # 実行前チェック
        if ((len(self.objectives)==0)
            or (len(self.parameters)==0)):
            raise Exception('パラメータまたは目的関数が設定されていません。')
        if historyPath is None:
            if self.historyPath is None:
                name = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')
                self.historyPath = os.path.join(os.getcwd(), f'{name}.csv')
            else:
                pass
        else:
            self.historyPath = historyPath

        # 前処理
        if not self.continuation:
            self._initRecord()
        self.interruption = False

        # 時間測定しながら実行
        startTime = time.time()
        try:
            if self.processMonitor is None:
                # multithread する必要がない
                self._main()
            else:
                # GUI と別 thread にする
                thread = threading.Thread(target=self._main)
                thread.start()
                while thread.is_alive():
                    self.processMonitor.update()
 
        except UserInterruption:
            # 中断指令
            pass

        endTime = time.time()
        self.lastExecutionTime = endTime - startTime

        return


    # post-processing methods    

    def _genarateHistoryColumnNames(self):
        paramNames = [p for p in self.parameters['name'].values]
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
            
    def _initRecord(self):
        columns = []
        columns.append('n_trial')
        pNames, oNames, cNames = self._genarateHistoryColumnNames()
        columns.extend(pNames)
        columns.extend(oNames)
        columns.extend(cNames)
        columns.append('non_domi')
        columns.append('fit')
        columns.append('algorithm_message')
        columns.append('error_message')
        columns.append('hypervolume')
        columns.append('time')
        self.history = pd.DataFrame(columns=columns)
    
    def _record(self):
        row = []
        paramValues = list(self.parameters['value'].copy().values)
        objValues = self.objectiveValues
        consValues = self.constraintValues
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
        row.append(self.algorithm_message)
        row.append(self.error_message)
        row.append(np.nan) # hypervolume, temporal data
        row.append(datetime.datetime.now()) # time
        pdf = pd.Series(row, index=self.history.columns)
        # calc fit
        fit = self._calcConstraintFit(pdf)
        pdf['fit'] = fit
        # add to history
        self.history.loc[len(self.history)] = pdf
        # update non-domi(after addition to history)
        self._calcNonDomi()
        # calc hypervolume(after addition to history)
        self._calc_hypervolume()
        # 保存
        try:
            self.history.to_csv(self.historyPath, index=None, encoding='shift-jis')
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
        
    