import numpy as np

from pyfemtet.opt.prediction.single_task_gp import SingleTaskGPModel


def main(n_parameters, n_objectives, n_data, n_test):

    x = np.random.rand(n_data, n_parameters)
    y = np.random.rand(n_data, n_objectives)
    test_x = np.random.rand(n_test, n_parameters)

    model = SingleTaskGPModel()

    model.fit(x, y)

    mean, std = model.predict(test_x)

    assert mean.shape == (n_test, n_objectives)


def test1():
    n_parameters = 5
    n_objectives = 3
    n_data = 10
    n_test = 1
    main(n_parameters, n_objectives, n_data, n_test)


def test2():
    n_parameters = 2
    n_objectives = 1
    n_data = 10
    n_test = 5
    main(n_parameters, n_objectives, n_data, n_test)


test2()
