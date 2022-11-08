from toolz import partial
from scipy.optimize import fmin_slsqp
import numpy as np


class SyntheticControl:

    # Loss function
    def loss(self, W, X, y) -> float:
        return np.sqrt(np.mean((y - X.dot(W)) ** 2))

    # Fit model
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

    # Predict
    def predict(self, X):
        return X.dot(self.coef_)
