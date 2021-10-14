import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from desdeo_problem.surrogatemodels.SurrogateModels import BaseRegressor, ModelError

class SVRsurrogate(BaseRegressor):
    def __init__(self, L: float = None):
        self.L: float = L
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.m = None
        self.sc_y=None
        

    def fit(self, X, y):
        if isinstance(X, (pd.DataFrame, pd.Series)):
            X = X.values
        if isinstance(y, (pd.DataFrame, pd.Series)):
            y = y.values.squeeze()

        # Make a 2-D array if needed
        #y = np.atleast_1d(y)
        #if y.ndim == 1:
        #    y = y.reshape(-1, 1)
        self.m = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))
        #sc_X = StandardScaler()
        #self.sc_y = StandardScaler()
        #X = sc_X.fit_transform(X)
        #y = self.sc_y.fit_transform(y)
        #self.m = SVR(kernel = 'rbf')
        self.m.fit(X,y)

    def predict(self, X):
        #y_mean, y_stdev = np.asarray(self.m.predict(X)[0]).reshape(1,-1)
        #y_mean = np.asarray(self.m.predict(X))
        #y_mean = (y_mean.reshape(1,-1))
        y_pred = self.m.predict(X)
        #y_pred = self.sc_y.inverse_transform(y_pred) 
        return (y_pred,y_pred)

