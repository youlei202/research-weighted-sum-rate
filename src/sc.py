from toolz import partial
from scipy.optimize import fmin_slsqp
import numpy as np
from sklearn.linear_model import LinearRegression


class SyntheticControl:
    """The code refers to
    https://github.com/matteocourthoud/Blog-Posts/blob/main/notebooks/synthetic_control.ipynb
    """

    def loss(self, W, X, y) -> float:
        return np.sqrt(np.mean((y - X.dot(W)) ** 2))

    def fit(self, X, y):
        w_start = [1 / X.shape[1]] * X.shape[1]
        self.coef_ = fmin_slsqp(
            partial(self.loss, X=X, y=y),
            np.array(w_start),
            f_eqcons=lambda x: np.sum(x) - 1,
            bounds=[(0.0, 1.0)] * len(w_start),
            disp=False,
        )
        self.mse = self.loss(W=self.coef_, X=X, y=y)
        return self

    def predict(self, X):
        return X.dot(self.coef_)


class SyntheticControlUnconstrained:
    def __init__(self):
        self.model = LinearRegression()

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return X.dot(self.model.coef_)
