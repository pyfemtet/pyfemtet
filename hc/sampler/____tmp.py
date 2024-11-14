import numpy as np
from sklearn.gaussian_process import GaussianProcessClassifier


train_x = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
])

train_y = np.array([
    0, 0, 1, 1
])


from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process.kernels import DotProduct
gpc = GaussianProcessClassifier(
    kernel=1. * RBF(length_scale=1.),
    # kernel=1. * Matern(length_scale=1, nu=0.5),
    # kernel=1. * DotProduct(sigma_0=1.),
    random_state=42,
)

gpc.fit(train_x, train_y)

test_x = np.array([train_x[3]])
test_y = gpc.predict(test_x)
test_y_prob = gpc.predict_proba(test_x)
test_y_ll = gpc.log_marginal_likelihood()
print(test_y_prob)
