import numpy as np
import pandas as pd
import GPy
from desdeo_problem.surrogatemodels.SurrogateModels import BaseRegressor, ModelError

class SparseGPRegressor(BaseRegressor):
    def __init__(self, Z):
        self.Z = Z
        self.m = None
        #self.opt_inducing_inputs = None

    def fit(self, X, y):
        if isinstance(X, (pd.DataFrame, pd.Series)):
            X = X.values
        if isinstance(y, (pd.DataFrame, pd.Series)):
            y = y.values.reshape(-1, 1)

        # Make a 2-D array if needed
        y = np.atleast_1d(y)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        kernel = GPy.kern.Matern52(ARD=True) #+ GPy.kern.White(2)
        self.m = GPy.models.SparseGPRegression(X,y,Z=self.Z,kernel=kernel)
        self.m.inducing_inputs.fix()
        self.m.optimize('bfgs')
        self.X = X
        self.y = y
        #self.opt_inducing_inputs = np.asarray(self.m.inducing_inputs)

    def predict(self, X):
        #y_mean = np.asarray(self.m.predict(X)[0]).reshape(1,-1)
        y_mean, y_stdev = np.asarray(self.m.predict(X))
        y_mean = (y_mean.reshape(1,-1))
        y_stdev = (y_stdev.reshape(1,-1))
        return (y_mean, y_stdev)



