from time import sleep

from PyFemtet.opt import FemtetOptuna
from PyFemtet.opt.core import NoFEM

import numpy as np

def objective_x(FEMOpt):
    sleep(1)
    r, theta, fai = FEMOpt.get_parameter('values')
    return r * np.cos(theta) * np.cos(fai)


def objective_y(FEMOpt):
    r, theta, fai = FEMOpt.get_parameter('values')
    return r * np.cos(theta) * np.sin(fai)

def objective_z(FEMOpt):
    r, theta, fai = FEMOpt.get_parameter('values')
    return r * np.sin(theta)

def constraint_y(FEMOpt):
    y = objective_y(FEMOpt)
    return y
    
def constraint_z(FEMOpt):
    z = objective_z(FEMOpt)
    return z

    
if __name__=='__main__':
    FEMOpt = FemtetOptuna(None, FEM=NoFEM())
    FEMOpt.add_parameter('r', .5, 0, 1)
    FEMOpt.add_parameter('theta', np.pi/3, 0, 2*np.pi)
    FEMOpt.add_parameter('fai', np.pi/3, -np.pi/2, np.pi/2)
    FEMOpt.add_objective(objective_x, 'x', args=FEMOpt)
    FEMOpt.add_objective(objective_y, 'y', args=FEMOpt)
    FEMOpt.add_objective(objective_z, 'z', args=FEMOpt)
    FEMOpt.add_constraint(constraint_y, 'y<=0', upper_bound=0, args=FEMOpt)
    FEMOpt.add_constraint(constraint_z, 'z<=0', upper_bound=0, args=FEMOpt, strict=False)
    # print(FEMOpt.FEM)
    # print(FEMOpt.FEMClass)
    # print(FEMOpt.constraints)
    # print(FEMOpt.objectives)
    # print(FEMOpt.get_parameter())
    # print(FEMOpt.history)
    # print(FEMOpt.history_path)
    # print(FEMOpt.last_execution_time)
    # FEMOpt._init_history()
    # print(len(FEMOpt.history))
    # print(FEMOpt.f(FEMOpt.get_parameter('values')))
    # print(FEMOpt._objective_values)
    # print(FEMOpt._constraint_values)
    # row = FEMOpt._get_current_data()
    # FEMOpt._append_history(row)
    # FEMOpt._calc_hypervolume()
    # FEMOpt.history['hypervolume']
    

    FEMOpt.main(n_trials=1000, n_parallel=3)
    
    print(len(FEMOpt.history)) # n_trials から prune を抜いた数か

    import optuna
    study = optuna.load_study(study_name=FEMOpt.study_name, storage=FEMOpt.storage_name)
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

    
    
    


