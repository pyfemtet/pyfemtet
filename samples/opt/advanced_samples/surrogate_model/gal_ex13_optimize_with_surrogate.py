import os

from optuna.samplers import TPESampler

from pyfemtet.opt import FEMOpt, OptunaOptimizer
from pyfemtet.opt.interface import PoFBoTorchInterface


def main(target):

    os.chdir(os.path.dirname(__file__))

    # Instead of connecting with Femtet, create
    # a surrogate model. Read the CSV file created
    # by the training data creation script to build
    # the surrogate model.
    fem = PoFBoTorchInterface(
        history_path='training_data.csv'
    )

    # Set up the optimization object.
    opt = OptunaOptimizer(
        sampler_class=TPESampler,
    )

    # Set up the FEMOpt object.
    femopt = FEMOpt(
        fem=fem,
        opt=opt,
        history_path=f'optimized_result_target_{target}.csv'
    )

    # Set up the design variables.
    # The upper and lower limits can differ from
    # those in the training data creation script,
    # but please note that extrapolation will
    # occur outside the range that has not been
    # trained, which may reduce the prediction
    # accuracy of the surrogate model.
    femopt.add_parameter('length', 0.1, 0.02, 0.2)
    femopt.add_parameter('width', 0.01, 0.001, 0.02)

    # If there are parameters that were set as
    # design variables during training and wanted
    # to fix during optimization, specify only the
    # `initial_value` and set the `fix` argument True.
    # You cannot add design variables that were not
    # set during training for optimization.
    femopt.add_parameter('base_radius', 0.008, fix=True)

    # Specify the objective functions set during
    # training that you want to optimize.
    # You may provide the fun argument, but it will
    # be overwritten during surrogate model creation,
    # so it will be ignored.
    # You cannot use objective functions that were
    # not set during training for optimization.
    obj_name = 'First Resonant Frequency (Hz)'
    femopt.add_objective(
        name=obj_name,
        direction=target,
    )

    # Execute the optimization.
    femopt.set_random_seed(42)
    df = femopt.optimize(
        n_trials=50,
        confirm_before_exit=False
    )

    # Display the optimal solution.
    prm_names = femopt.history.prm_names
    obj_names = femopt.history.obj_names
    prm_values = df[df['non_domi'] == True][prm_names].values[0]
    obj_values = df[df['non_domi'] == True][obj_names].values[0]

    message = f'''
===== Optimization Results =====
Target Value: {target}
Prediction by Surrogate Model:
'''
    for name, value in zip(prm_names, prm_values):
        message += f'  {name}: {value}\n'
    for name, value in zip(obj_names, obj_values):
        message += f'  {name}: {value}\n'

    return message


if __name__ == '__main__':
    # Using the surrogate model created from the training data,
    # we will find a design that results in a resonant frequency of 1000.
    message_1000 = main(target=1000)

    # Next, using the same surrogate model,
    # we will find a design that results in a resonant frequency of 2000.
    message_2000 = main(target=2000)

    print(message_1000)
    print(message_2000)
