import optuna

def objective(trial):
    x = trial.suggest_float('x', 0, 1)
    y = trial.suggest_float('y', 0, 1)
    return x**2 + y**2

sampler = optuna.samplers.TPESampler(seed=42)

study = optuna.create_study(
    sampler=sampler,
)

study.enqueue_trial(
    dict(x=.5, y=.5),
)

study.optimize(objective, n_trials=10)

for trial in study.trials:
    print(trial.params)

