from time import sleep

from PyFemtet.opt import FemtetOptuna
from PyFemtet.opt.core import NoFEM

import numpy as np

def objective_x(FEMOpt):
    r, theta = FEMOpt.get_current_parameter('values')
    return r * np.cos(theta)

def objective_y(FEMOpt):
    r, theta = FEMOpt.get_current_parameter('values')
    sleep(1)
    return r * np.sin(theta)

def constraint_y(FEMOpt):
    y = objective_y(FEMOpt)
    return y
    
    
if __name__=='__main__':
    FEMOpt = FemtetOptuna(None, FEMClass=NoFEM)
    FEMOpt.add_parameter('r', 1, 0, 1)
    FEMOpt.add_parameter('theta', np.pi/3, 0, 2*np.pi)
    FEMOpt.add_objective(objective_x, 'x', args=FEMOpt)
    FEMOpt.add_objective(objective_y, 'y', args=FEMOpt)
    # FEMOpt.add_constraint(constraint_y, 'y<=0', upper_bound=0, args=FEMOpt)
    # print(FEMOpt.FEM)
    # print(FEMOpt.FEMClass)
    # print(FEMOpt.constraints)
    # print(FEMOpt.objectives)
    # print(FEMOpt.get_current_parameter())
    # print(FEMOpt.history)
    # print(FEMOpt.history_path)
    # print(FEMOpt.last_execution_time)
    # print(FEMOpt._init_history())
    # print(len(FEMOpt.history))
    # print(FEMOpt.f(FEMOpt.get_current_parameter('values')))
    # print(FEMOpt._objective_values)
    # print(FEMOpt._constraint_values)
    

    FEMOpt.main(n_trials=9, n_parallel=3)
    
    print(len(FEMOpt.history)) # n_trials から prune を抜いた数か

    import optuna
    study = optuna.load_study(FEMOpt.study_name, FEMOpt.storage_name)
    df = study.trials_dataframe()
    idx = df['state']=='COMPLETE'
    print(len(df[idx]))
    
    print(len(df[idx])==len(FEMOpt.history)) # 中身も見たが OK
    
    print(len(df))
    
    
    # ['number', 'values_0', 'values_1', 'datetime_start', 
    # 'datetime_complete',
    # 'duration', 'params_r', 'params_theta', 'user_attrs_constraint',
    # 'user_attrs_memo', 'system_attrs_constraints',
    # 'system_attrs_fixed_params', 'state']

    
    
    


