import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from desdeo_problem.surrogatemodels.SurrogateModels import BaseRegressor, ModelError

class PolynomialRegressionsurrogate(BaseRegressor):
    def __init__(self, L: float = None):
        self.L: float = L
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.m = None
        

    def fit(self, X, y):
        if isinstance(X, (pd.DataFrame, pd.Series)):
            X = X.values
        if isinstance(y, (pd.DataFrame, pd.Series)):
            y = y.values.squeeze()

        self.m = make_pipeline(PolynomialFeatures(degree=3), Ridge(alpha=1e-3))
        self.m.fit(X,y)

    def predict(self, X):
        #y_mean, y_stdev = np.asarray(self.m.predict(X)[0]).reshape(1,-1)
        y_mean = np.asarray(self.m.predict(X))
        y_mean = (y_mean.reshape(1,-1))
        return (y_mean, y_mean)

