import numpy as np
import pandas as pd

from .core import FemtetOptimizationCore

from scipy.optimize import minimize # pip install scipy
from scipy.optimize import NonlinearConstraint


# basinhopping(func, x0[, niter, T, stepsize, ...])
# differential_evolution(func, bounds[, args, ...])
# shgo(func, bounds[, args, constraints, n, ...])
# dual_annealing(func, bounds[, args, ...])
# direct(func, bounds, *[, args, eps, maxfun, ...])



class FemtetScipy(FemtetOptimizationCore):
    """FemtetScipy class

    execute parameter optimization via Femtet and Scipy.optimize.minimize

    Attributes:

    """    
    def __init__(self, setFemtetStorategy=None):
        if setFemtetStorategy is None:
            super().__init__()
        else:
            super().__init__(setFemtetStorategy)
        self._objectives = []
        self._constraints = []
        
    def _main(self):
        """
        run parameter optimization.

        Returns
        -------
        opt : OptimizeResult
            for detail, see Scipy documentation.
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html

        """
        if len(self.objectives)>1:
            raise Exception('単目的最適化用アルゴリズムで複数の目的関数を最適化しようとしています。')
        
        df = self.parameters
        
        method='COBYLA'
        # method='SLSQP'
        x0 = df['value'].values
        bounds = df[['lbound', 'ubound']].replace([np.nan], [None]).values
        self._parseConstraints(method)
        
        opt = minimize(
            fun=lambda x: self.f(x)[0],
            x0=x0,
            bounds=bounds,
            method=method,
            constraints=self._constraints,
            # options={'maxiter':10},
            # callback=self.record,
            )

        return opt

    def _createConstraintFun(self, x, i):
        if not self._isCalculated(x):
            self.f(x)
        return self.constraintValues[i]

    def _parseConstraints(self, method):
        '''与えられた拘束情報を Scipy 形式に変換する'''
        if method=='trust-constr':
            # for trust-constr
            self._constraints = []
            for i, constraint in enumerate(self.constraints):
                fun = lambda x,i=i : self._createConstraintFun(x, i)
                lb = -np.inf if constraint.lb is None else constraint.lb
                ub = np.inf if constraint.ub is None else constraint.ub
                cons = NonlinearConstraint(fun, lb, ub)
                self._constraints.append(cons)
        elif method=='COBYLA' or method=='SLSQP':
            # for COBYLA, SLSQP
            self._constraints = []
            for i, constraint in enumerate(self.constraints):
                lb = constraint.lb
                ub = constraint.ub
                if lb is not None:
                    fun = lambda x,i=i,lb=lb : self._createConstraintFun(x, i) - lb
                    self._constraints.append({'type':'ineq', 'fun':fun})
                if ub is not None:
                    fun = lambda x,i=i,ub=ub : ub - self._createConstraintFun(x, i)
                    self._constraints.append({'type':'ineq', 'fun':fun})
        else:
            # other scipy methods cannot consider constraints
            pass

        
        
if __name__=='__main__':
    import numpy as np
    
    FEMOpt = FemtetScipy(None)
    
    # 変数の設定
    FEMOpt.add_parameter('x', 5, -10, 10)
    FEMOpt.add_parameter('y', 5, -10, 10)
    
    # 目的関数の設定
    def obj(FEMOpt):
        x = FEMOpt.parameters['value'].values
        return (x**2).sum()

    FEMOpt.add_objective(obj, '放物線', args=(FEMOpt,))


    # 拘束関数の設定
    def r(FEMOpt):
        x = FEMOpt.parameters['value'].values
        return np.sqrt((x**2).sum())

    FEMOpt.add_constraint(r, '半径（1以上）', 1, None, args=(FEMOpt,))


    def theta(FEMOpt):
        x = FEMOpt.parameters['value'].values
        return np.arctan2(x[1], x[0]) * 360 / 2 / np.pi

    FEMOpt.add_constraint(theta, '角度（0以上90以下）', 0, 90, args=(FEMOpt,))
    
    # プロセスモニタの設定（experimental）
    FEMOpt.set_process_monitor()

    # 計算の実行
    opt = FEMOpt.main()
    
    # 結果表示
    print(FEMOpt.history)
    print(opt)
    
    # df = FEMOpt.history
    # xydata = df[['x', 'y']].values.T
    # import matplotlib.pyplot as plt
    # plt.plot(*xydata)
    # plt.show()