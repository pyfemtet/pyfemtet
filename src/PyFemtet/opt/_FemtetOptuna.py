import os
import datetime
import functools

import numpy as np
import pandas as pd

from .core import FemtetOptimizationCore

import optuna
# optuna.logging.disable_default_handler()

from scipy.stats.qmc import LatinHypercube

from multiprocessing import Process, Manager

import gc



def generate_LHS(bounds)->np.ndarray:
    '''
    d 個の変数の bounds を貰い、N 個のサンプル点を得る。
    N は d を超える最小の素数 p に対し N = p**2
    '''
    d = len(bounds)

    sampler = LatinHypercube(
        d,
        scramble=False,
        strength=2,
        # optimization='lloyd',
        optimization='random-cd',
        # seed=1
        )
    
    LIMIT = 100

    def is_prime(p):
        for i in range(2, p):
            if p%i==0:
                return False
        return True
    
    def get_prime(minimum):
        for p in range(minimum, LIMIT):
            if is_prime(p):
                return p
        
    n = get_prime(d+1)**2
    data = sampler.random(n) # [0,1)
    
    for i, (data_range, datum) in enumerate(zip(bounds, data.T)):
        minimum, maximum = data_range
        band = maximum - minimum
        converted_datum = datum * band + minimum
        data[:,i] = converted_datum
    
    return data # data.shape = (N, d)

# https://optuna.readthedocs.io/en/stable/index.html
class FemtetOptuna(FemtetOptimizationCore):

    def main(
            self,
            timeout=None,
            n_trials=None,
            study_name=None,
            use_init_LHS=True, # Latin Hypercube Sampling を初期値にする
            n_parallel=1,
            ):
        # 宣言
        self._constraints = []

        # 引数の処理
        self.use_init_LHS = use_init_LHS
        self.n_parallel = n_parallel
        
        #### study name のセットアップ
        # study_name（ファイル名）の設定
        if study_name is None:
            self.study_name = os.path.splitext(os.path.basename(self.history_path))[0] + ".db"
        # csv 保存パスと同じところに db ファイルを保存する
        self.study_db_path = os.path.abspath(
            os.path.join(
                os.path.dirname(self.history_path),
                self.study_name
                )
            )
        self.storage_name = f"sqlite:///{self.study_db_path}"
        

        #### timeout or n_trials の設定
        self.timeout = timeout
        self.n_trials = n_trials

        # 本番処理に移る
        super().main()


    def _setup_study(self):
        #### sampler の設定
        # sampler = optuna.samplers.NSGAIISampler(constraints_func=_constraint_function)
        # sampler = optuna.samplers.NSGAIIISampler()
        sampler = optuna.samplers.TPESampler(constraints_func=self._constraint_function)

        #### study ファイルの作成
        # 並列計算について https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/004_distributed.html
        # 並列計算時の最大試行回数について https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.MaxTrialsCallback.html#optuna.study.MaxTrialsCallback
        study = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage_name,
            load_if_exists=True,
            directions=['minimize']*len(self.objectives),
            sampler=sampler,
            )

        #### 初期値の設定
        params = self.get_current_parameter('dict')
        study.enqueue_trial(params, user_attrs={"memo": "initial"})
        
        #### LHS を初期値にする
        if self.use_init_LHS:
            df = self.get_current_parameter('df')
            names = []
            bounds = []
            for i, row in df.iterrows():
                names.append(row['name'])
                lb = row['lbound']
                ub = row['ubound']
                bounds.append([lb, ub])
            data = generate_LHS(bounds)        
            for datum in data:
                d = {}
                for name, v in zip(names, datum):
                    d[name] = v
                study.enqueue_trial(d, user_attrs={"memo": "initial Latin Hypercube Sampling"})        
        
        return study


        

    def _main(self):
        study = self._setup_study()
        
        if self.n_parallel>1:
            processes = []
            for i in range(self.n_parallel-1):
                p = Process(
                    target=self._subprocess_main,
                    args=(self,)
                    )
                p.start()
                print('subprocess start')
                processes.append(p)

        # study の実行
        if self.n_trials is None:
            n_trials = None
        else:
            n_trials = self.n_trials//self.n_parallel
        study.optimize(
            self._objective_function,
            timeout=self.timeout,
            n_trials=n_trials,
            )
        
        for p in processes:
            p.join()
            print('subprocess end')

        return study

    #### 目的関数
    def _objective_function(self, trial):
        #### 変数の設定
        x = []
        df = self.get_current_parameter('df')
        for i, row in df.iterrows():
            name = row['name']
            lb = row['lbound']
            ub = row['ubound']
            x.append(trial.suggest_float(name, lb, ub))
        x = np.array(x)

        #### user_attribute の設定
        # memo
        try:
            # あれば
            message = trial.user_attrs["memo"]
        except (AttributeError, KeyError):
            # なければ
            message = ''
        # 拘束のアトリビュート（Prune などの対策、計算できたら後で上書きする）
        dummy_data = tuple([1 for f in self._constraints])
        trial.set_user_attr("constraint", dummy_data)
        
        #### strict 拘束の計算
        # is_calcrated の直後まで変数をアップデートしてはいけないため
        # 一度 buff に現在の変数を控え、処理が終わったら戻す
        # 退避+更新
        buff = self._parameter.copy()
        self._parameter['value'] = x
        # restore の注意 / strict 拘束の処理から抜けうるところには
        # すべてに以下の処理を配置すること
        # self._parameter = buff
        # Femtet 関係の FEM システムであれば変数を更新
        from .core import Femtet, ModelError
        if isinstance(self.FEM, Femtet):
            try:
                # ModelError が起きうる
                self.FEM.update_model(self._parameter)
            except ModelError:
                raise optuna.TrialPruned()
        # strict 拘束の計算
        val_lb_ub_fn_list = [[cns.calc(), cns.lb, cns.ub, cns.f.__name__] for cns in self.constraints if cns.strict==True]
        for val, lb, ub, fn in val_lb_ub_fn_list:
            if lb is not None:
                if not (lb <= val):
                    print(f'拘束 {fn} が満たされませんでした。')
                    print('変数の組み合わせは以下の通りです。')
                    print(self._parameter)
                    self._parameter = buff # Femtet は戻す必要ない、以下同文
                    raise optuna.TrialPruned()
            if ub is not None:
                if not (val <= ub):
                    self._parameter = buff
                    print(f'拘束 {fn} が満たされませんでした。')
                    print('変数の組み合わせは以下の通りです。')
                    print(self._parameter)
                    raise optuna.TrialPruned()
        self._parameter = buff

        #### 解析実行
        # ModelError 等が起こりうる
        from .core import ModelError, MeshError, SolveError
        try:
            # parameters を更新し、解析、目的関数、全ての拘束の計算
            ojectiveValuesToMinimize = self.f(x, message)
        except (ModelError, MeshError, SolveError):
            raise optuna.TrialPruned()
        
        #### 拘束の計算
        _constraint_values = [] # 非正なら OK
        for i, cns in enumerate(self.constraints):
            lb, ub = cns.lb, cns.ub
            if lb is not None: # fun >= lb  <=>  lb - fun <= 0
                _constraint_values.append(lb - cns.calc())
            if ub is not None: # ub >= fun  <=>  fun - ub <= 0
                _constraint_values.append(cns.calc() - ub)
        trial.set_user_attr("constraint", _constraint_values)
        
        # 一応
        gc.collect()

        # 結果
        return tuple(ojectiveValuesToMinimize)


    #### 拘束関数(experimental)
    def _constraint_function(self, trial):
        return trial.user_attrs["constraint"]


    def _subprocess_main(_, FEMOpt): # ここでは self はアクセスしたらダメ
        # サブプロセスから呼ばれることを想定

        # サブプロセス時に破棄されているはずの COM を含みうる FEM を接続
        FEMOpt.set_FEM()

        # メインプロセスで作ったはずの study に接続        
        study = optuna.load_study(
            study_name=FEMOpt.study_name,
            storage=FEMOpt.storage_name
        )
        
        # n_trials を指定されていた場合の処理（もう少しうまいことできる）
        if FEMOpt.n_trials is None:
            n_trials = None
        else:
            n_trials = FEMOpt.n_trials//FEMOpt.n_parallel

        # 実行
        study.optimize(
            FEMOpt._objective_function,
            timeout=FEMOpt.timeout,
            n_trials=n_trials,
            )

        # TODO:この辺で新しい Femtet インスタンスを閉じる(自動で全部閉じるかも）
            


if __name__=='__main__':
    FEMOpt = FemtetOptuna(None)
    
    # 変数の設定
    FEMOpt.add_parameter('r', 5, 0, 10)
    FEMOpt.add_parameter('theta', 0, -np.pi/2, np.pi/2)
    FEMOpt.add_parameter('fai', np.pi, 0, 2*np.pi)
    
    # 目的関数の設定
    def obj1(FEMOpt):
        r, theta, fai = FEMOpt.parameters['value'].values
        return r * np.cos(theta) * np.cos(fai)
    FEMOpt.add_objective(obj1, 'x', direction='maximize', args=(FEMOpt,))

    def obj2(FEMOpt):
        r, theta, fai = FEMOpt.parameters['value'].values
        return r * np.cos(theta) * np.sin(fai)
    FEMOpt.add_objective(obj2, 'y', args=(FEMOpt,)) # defalt direction is minimize

    def obj3(FEMOpt):
        r, theta, fai = FEMOpt.parameters['value'].values
        return r * np.sin(theta)
    FEMOpt.add_objective(obj3, 'z', direction=3, args=(FEMOpt,))


    # プロセスモニタの設定（experimental / 問題設定後実行直前に使うこと）
    FEMOpt.set_process_monitor()
    
    # 計算の実行
    study = FEMOpt.main()
    
    # 結果表示
    # print(FEMOpt.history)
    # print(study)
    
#     stop

#     #### ポスト
#     # https://optuna-readthedocs-io.translate.goog/en/stable/reference/visualization/index.html?_x_tr_sl=auto&_x_tr_tl=ja&_x_tr_hl=ja&_x_tr_pto=wapp

#     # You can use Matplotlib instead of Plotly for visualization by simply replacing `optuna.visualization` with
#     # `optuna.visualization.matplotlib` in the following examples.
#     from optuna.visualization.matplotlib import plot_contour
#     from optuna.visualization.matplotlib import plot_edf
#     from optuna.visualization.matplotlib import plot_intermediate_values
#     from optuna.visualization.matplotlib import plot_optimization_history
#     from optuna.visualization.matplotlib import plot_parallel_coordinate
#     from optuna.visualization.matplotlib import plot_param_importances
#     from optuna.visualization.matplotlib import plot_rank
#     from optuna.visualization.matplotlib import plot_slice
#     from optuna.visualization.matplotlib import plot_timeline
#     from optuna.visualization.matplotlib import plot_pareto_front
#     from optuna.visualization.matplotlib import plot_terminator_improvement
    
#     # 多分よくつかう
#     show_objective_index = [0,1,2]
    
#     ax = plot_pareto_front(
#         study,
#         targets = lambda trial: [trial.values[idx] for idx in show_objective_index]
#         )
    
#     # parate_front の scatter をクリックしたら値を表示するようにする
#     import matplotlib.pyplot as plt
#     from matplotlib.collections import PathCollection
    
#     def on_click(event):
#         ind = event.ind[0]
#         try:
#             x = event.artist._offsets3d[0][ind]
#             y = event.artist._offsets3d[1][ind]
#             z = event.artist._offsets3d[2][ind]
#             print(f"Clicked on point: ({x}, {y}, {z})")
#             costs = [x, y, z]
#         except:
#             x = event.artist._offsets[ind][0]
#             y = event.artist._offsets[ind][1]
#             print(f"Clicked on point: ({x}, {y})")
#             costs = [x, y]

#         for trial in study.trials:
#             paramsDict = trial.params
#             if costs==[trial.values[idx] for idx in show_objective_index]:
#                 print(paramsDict)
#                 try:
#                     print(trial.user_attrs['memo'])
#                 except:
#                     pass

#     def set_artist_picker(ax):
#         for artist in ax.get_children():
#             if isinstance(artist, PathCollection):
#                 artist.set_picker(True)

#     set_artist_picker(ax)
#     fig = ax.get_figure()
#     fig.canvas.mpl_connect('pick_event', on_click)
#     plt.show()





#     plot_optimization_history(study, target=lambda t: t.values[1])
#     # plot_contour(study, target=lambda t: t.values[1])
#     # plot_param_importances(study, target=lambda t: t.values[1])
#     # plot_slice(study, target=lambda t: t.values[1])
    
#     # # 多分あまり使わない
#     # plot_timeline(study)
#     # plot_edf(study, target=lambda t: t.values[0])
#     # plot_intermediate_values(study) # no-pruner study not supported
#     # plot_rank(study, target=lambda t: t.values[1])
#     # plot_terminator_improvement(study) # multiobjective not supported
    
    
    







# #### 並列処理
# if False:
#     from subprocess import Popen
    
#     # 元studyの作成
#     optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
#     study_name = "example-study"  # Unique identifier of the study.
#     storage_name = "sqlite:///{}.db".format(os.path.join(here, study_name))
#     study = optuna.create_study(study_name=study_name, storage=storage_name, load_if_exists=True)
    
    
#     path = os.path.join(here, '_optuna_dev.py')
#     pythonpath = r"C:\Users\mm11592\Documents\myFiles2\working\PyFemtetOpt\venvPyFemtetOpt\Scripts\python.exe c:\\users\\mm11592\\documents\\myfiles2\\working\\pyfemtetopt\\local\\pyfemtetopt\\pyfemtetopt\\core\\_optuna_dev.py example-study sqlite:///c:\\users\\mm11592\\documents\\myfiles2\\working\\pyfemtetopt\\local\\pyfemtetopt\\pyfemtetopt\\core\\example-study.db"
#     # Popen([pythonpath, path, study_name, storage_name])
#     # Popen([pythonpath, path, study_name, storage_name])
    
#     study = optuna.create_study(study_name=study_name, storage=storage_name, load_if_exists=True)
#     df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
#     print(df)
    
#     optuna.delete_study(study_name, storage_name)

# #### 停止・再開
# if False:
#     # Add stream handler of stdout to show the messages
#     # 保存するための呪文
#     optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
#     study_name = "example-study"  # Unique identifier of the study.
#     storage_name = "sqlite:///{}.db".format(os.path.join(here, study_name))
#     study = optuna.create_study(study_name=study_name, storage=storage_name)
    
#     study.optimize(objective, n_trials=3)
    
#     # study.best_params  # E.g. {'x': 2.002108042}
    
#     # 再開
#     study = optuna.create_study(study_name=study_name, storage=storage_name, load_if_exists=True)
#     study.optimize(objective, n_trials=3)
    
    
#     # # sampler の seed をも restore するには以下の呪文を使う。
#     # import pickle
    
#     # # Save the sampler with pickle to be loaded later.
#     # with open("sampler.pkl", "wb") as fout:
#     #     pickle.dump(study.sampler, fout)
    
#     # restored_sampler = pickle.load(open("sampler.pkl", "rb"))
#     # study = optuna.create_study(
#     #     study_name=study_name, storage=storage_name, load_if_exists=True, sampler=restored_sampler
#     # )
#     # study.optimize(objective, n_trials=3)
    
    
#     # history とかの取得
#     study = optuna.create_study(study_name=study_name, storage=storage_name, load_if_exists=True)
#     df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
#     print("Best params: ", study.best_params)
#     print("Best value: ", study.best_value)
#     print("Best Trial: ", study.best_trial)
#     print("Trials: ", study.trials)


