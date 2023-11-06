from __future__ import annotations
from typing import Any, Callable
from abc import ABC, abstractmethod
from warnings import warn

import os
import time
import datetime
import inspect
def getCurrentMethod():
    return inspect.currentframe().f_code.co_name

import numpy as np
import pandas as pd

import win32com.client
from win32com.client import Dispatch #, constants

from PyFemtet.tools.FemtetClassConst import FemtetClassName as const


class ModelError(Exception):
    pass

class MeshError(Exception):
    pass

class SolveError(Exception):
    pass

class UserInterruption(Exception):
    pass

class FEM:
    '''FEM システムのインスタンス。
    ただし、このクラスは型ヒントのために定義されたダミークラスです。
    このクラスのインスタンスを作成しても Femtet は操作できません。
    '''
    # 型ヒントのためだけのダミークラス
    pass


class Objective:
    prefixForDefault = 'objective'
    def __init__(self, fun, direction, name, opt):
        
        self._checkDirection(direction)
        
        self.fun = fun
        self.direction = direction
        self.name = name
        

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
        return ret         
                
        
class Constraint:
    prefixForDefault = 'constraint'
    def __init__(self, fun, lb, ub, name, opt):
        self.fun = fun
        self.lb = lb
        self.ub = ub
        self.name = name


class Femtet:

    def setFemtet(self, strategy='catch'):
        if strategy=='catch':
            # 既存の Femtet を探す
            self.Femtet = Dispatch(const.CFemtet)
        elif strategy=='new':
            # femtetUtil を使って新しい Femtet を立てる
            raise Exception('femtetUtilを使って新しいFemtetを立てるのはまだ実装していません')

    def run(self, df):
        '''run : 渡された df に基づいて Femtet に計算を実行させる関数。
        

        Raises
        ------
        ModelError
            アップデートされた変数セットでモデルの再構築に失敗した場合.
        MeshError
            アップデートされた変数セットでメッシュ生成に失敗した場合.
        SolveError
            アップデートされた変数セットでソルブに失敗した場合.

        Returns
        -------
        None.

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
                self.Femtet.ShowLastError
            
        # 設計変数に従ってモデルを再構築
        try:
            Gaudi.ReExecute()
        except win32com.client.pythoncom.com_error:
            raise ModelError('error was occured on updating model')

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
    
    

class FemtetOptimizationCore(ABC):
    
    def __init__(self, setFemtetStrategy:str or FEM or None = 'catch'):

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
        
        if setFemtetStrategy is not None:
            self.FEM = Femtet()
            self.FEM.setFemtet(setFemtetStrategy)

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


    #### pre-processing methods


    def add_objective(self,
                      fun:Callable[[Any], float],
                      name:str or None = None,
                      direction:str or float = 'minimize',
                      args:tuple or None = None,
                      kwargs:dict or None = None):

        f, name = self._setupFunction(fun, args, kwargs, name, Objective.prefixForDefault, self.objectives)
        
        obj = Objective(f, direction, name, self)

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
                      args:tuple or None = None,
                      kwargs:dict or None = None):

        f, name = self._setupFunction(fun, args, kwargs, name, Constraint.prefix, self.constraints)

        obj = Constraint(f, lower_bound, upper_bound, name, self)

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
    
    def get_parameter(self, format='dict'):
        if format=='df':
            return self.parameters
        elif format=='value':
            return self.parameters.value.values
        elif format=='dict':
            ret = {}
            for i, row in self.parameters.iterrows():
                ret[row.name] = row.value
            return ret
        else:
            fname = getCurrentMethod()
            raise Exception(f'{fname} got invalid format: {format}')

    def set_process_monitor(self):
        self._initRecord()
        if len(self.objectives)==1:
            from visualization._simpleObjectivePlot import SimpleProcessMonitor
            self.processMonitor = SimpleProcessMonitor(self)
        elif len(self.objectives)>1:
            from visualization._multiobjectivePairPlot import MultiobjectivePairPlot
            self.processMonitor = MultiobjectivePairPlot(self)
            

    def _setupFunction(self, fun, args, kwargs, name, prefix, objects):
        if args is None:
            args = ()
        elif type(args)!=tuple:
            args = (args,)

        if kwargs is None:
            kwargs = {}

        #### Femtetを使う場合は、第一引数に自動的に Femtet の com オブジェクトが入っていることとする。
        if type(self.FEM)==Femtet:
            f = lambda : fun(self.FEM.Femtet, *args, **kwargs)
        else:
            f = lambda : fun(*args, **kwargs)
        
        if name is None:
            name = self._getUniqueDefaultName(prefix, objects)
        
        return f, name

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
        return name
                

    #### processing methods

    def f(self, x:np.ndarray)->[float]:
        # 中断指令があったら Exception を起こす
        if self.interruption:
            raise UserInterruption
        
        
        # 渡された x がすでに計算されたものであれば
        # objective も constraint も更新する必要がない
        if not self._isCalculated(x):
            # update parameters
            self.parameters['value'] = x

            # solve
            if self.FEM is not None:
                try:
                    self.FEM.run(self.parameters)
                except (ModelError, MeshError, SolveError): # TODO:モデル、メッシャ、ソルバのエラーごとに何か処理を入れる
                    pass

            # calc
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

        # 時間測定しながら実行
        startTime = time.time()
        try:
            result = self._main()
        except UserInterruption: # 中断指令があったら
            pass
        endTime = time.time()
        self.lastExecutionTime = endTime - startTime

        return result


    # post-processing methods    

    def _genarateHistoryColumnNames(self):
        paramNames = [p for p in self.parameters['name'].values]
        objNames = [obj.name for obj in self.objectives]
        _consNames = [obj.name for obj in self.constraints]
        consNames = []
        for n in _consNames:
            consNames.extend([f'{n}_lower_bound', n, f'{n}_upper_bound', f'{n}_fit'])
        return paramNames, objNames, consNames

    def get_history_columns(self, kind='objective'):
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
        pNames, oNames, cNames = self._genarateHistoryColumnNames()
        columns.extend(pNames)
        columns.extend(oNames)
        columns.extend(cNames)
        columns.append('non_domi')
        columns.append('fit')
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
        row.extend(paramValues)
        row.extend(objValues)
        row.extend(consData)
        row.append(False) # non-domi, temporal data
        row.append(False) # fit, temporal data
        row.append(datetime.datetime.now()) # time
        pdf = pd.Series(row, index=self.history.columns)
        # calc fit
        fit = self._calcConstraintFit(pdf)
        pdf['fit'] = fit
        # add to history
        self.history.loc[len(self.history)] = pdf
        if self.processMonitor is not None:
            self.processMonitor.update()
        # update non-domi
        self._calcNonDomi()
        # 保存
        self.history.to_csv(self.historyPath, index=None, encoding='shift-jis')
            
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
        
        