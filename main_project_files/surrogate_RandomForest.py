import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from desdeo_problem.surrogatemodels.SurrogateModels import BaseRegressor, ModelError

class RandomForest(BaseRegressor):
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

        # Make a 2-D array if needed
        #y = np.atleast_1d(y)
        #if y.ndim == 1:
        #    y = y.reshape(1, -1)

        self.m = RandomForestRegressor(n_estimators=500)
        self.m.fit(X,y)

    def predict(self, X):
        #y_mean, y_stdev = np.asarray(self.m.predict(X)[0]).reshape(1,-1)
        y_mean = np.asarray(self.m.predict(X))
        y_mean = (y_mean.reshape(1,-1))
        """
        percentile=95
        err_down = []
        err_up = []
        for x in range(len(X)):
            preds = []
            for pred in self.m.estimators_:
                preds.append(pred.predict(X[x])[0])
            err_down.append(np.percentile(preds, (100 - percentile) / 2. ))
            err_up.append(np.percentile(preds, 100 - (100 - percentile) / 2.))
        return (y_mean,err_down, err_up)
        """
        return (y_mean, y_mean)




