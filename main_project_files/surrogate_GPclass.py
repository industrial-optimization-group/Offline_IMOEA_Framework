import numpy as np
import pandas as pd
import GPy
from desdeo_problem.surrogatemodels.SurrogateModels import BaseRegressor, ModelError

def build_classification_failed():
    main_directory = 'Pump_Test_Tomas_6_177'
    data_folder = '/home/amrzr/Work/Codes/data'
    #data_file = data_folder+'/pump_data/sim_stat2.csv'
    #data_file = data_folder+'/pump_data/03_DOE_180_failed.csv'
    #data_file = data_folder+'/pump_data/04_DOE_table_all.csv'
    #data_file = data_folder+'/pump_data/DOE_01_03_04_all.csv'
    data_file = data_folder+'/pump_data/DataSets_all.csv'
    df = pd.read_csv(data_file)   
    X=df.values[:,0:22]
    y=df.values[:,22]

    labels = list(set(y.flatten()))
    models = {}
    for label in labels:
        ytmp=y.copy()
        ytmp[ytmp!=label]=0
        ytmp[ytmp==label]=1
        
        m=GPy.models.GPClassification(X, ytmp[:, None])
        
        m.optimize_restarts(messages=True, robust=True, 
                            num_restarts=1
                            )
        #    else:
        #        m.optimize(messages=True)
        models[label]=m
    return models[1]



class GPclassification(BaseRegressor):
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


        self.m.fit(X,y)

    def predict(self, X):
        #y_mean, y_stdev = np.asarray(self.m.predict(X)[0]).reshape(1,-1)
        #y_mean = np.asarray(self.m.predict(X))
        #y_mean = (y_mean.reshape(1,-1))
        y_pred = self.m.predict(X)
        #y_pred = self.sc_y.inverse_transform(y_pred) 
        return (y_pred,y_pred)

