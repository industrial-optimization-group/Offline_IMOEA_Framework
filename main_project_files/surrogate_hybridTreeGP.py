import numpy as np
import pandas as pd
import hybrid_tree_gp
from desdeo_problem.surrogatemodels.SurrogateModels import BaseRegressor, ModelError

class HybridTreeGP(BaseRegressor):
    def __init__(self):
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.m = None

    def fit(self, X, y):
        if isinstance(X, (pd.DataFrame, pd.Series)):
            X = X.values
        if isinstance(y, (pd.DataFrame, pd.Series)):
            y = y.values

        # Make a 2-D array if needed
        y = np.atleast_1d(y)
        if y.ndim == 1:
            y = y

        self.m = hybrid_tree_gp.fit(X,y)
        self.X = X
        self.y = y


    def predict(self, X):
        #y_mean = np.asarray(self.m.predict(X)[0]).reshape(1,-1)
        y_mean = np.asarray(self.m.predict(X))
        y_mean = (y_mean.reshape(1,-1))
        y_stdev = None
        return (y_mean, y_stdev)

