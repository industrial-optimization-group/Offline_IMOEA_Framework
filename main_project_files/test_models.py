import sys
sys.path.insert(1, '/home/amrzr/Work/Codes/Offline_IMOEA_Framework/')

from desdeo_problem.Problem import DataProblem
from main_project_files.surrogate_fullGP import FullGPRegressor as fgp
from main_project_files.surrogate_SVR import SVRsurrogate
from main_project_files.surrogate_RandomForest import RandomForest
from main_project_files.surrogate_PolynomialRegression import PolynomialRegressionsurrogate as polyreg
import time
import numpy as np
import pandas as pd

main_directory = 'Pump_Test_Tomas_5_140_all'
data_folder = '/home/amrzr/Work/Codes/data'
#data_file = data_folder+'/pump_data/01_DOE_data.csv'
data_file = data_folder+'/pump_data/02_DOE_140_data.csv'
#data_file = data_folder+'/pump_data/03_DOE_140_all_data.csv'
path = data_folder + '/test_runs/' + main_directory

df = pd.read_csv(data_file)
df[['f1','f2','f3']] = df[['f1','f2','f3']]*-1

x_low = np.ones(22)*0
x_high = np.ones(22)

x_low_new = np.ones(22)*0
x_high_new = np.ones(22)

def scale_data(data):
    x_data = np.asarray(data.loc[:,:'x22'])
    x_data_scaled = (x_data - x_low)/(np.asarray(x_high) - np.asarray(x_low))
    data.loc[:,:'x22'] = x_data_scaled
    return data

def build_surrogates(nobjs, nvars, df): #x_data, y_data):
    x_names = list(df.columns)[0:22]
    y_names = list(df.columns)[22:25]
    row_names = ['lower_bound','upper_bound']
    #data = pd.DataFrame(np.hstack((x_data,y_data)), columns=x_names+y_names)
    #data = df
    #print(data)
    zz = np.asarray(df.loc[:,:'x22'])
    #x_low = np.min(zz,axis=0)
    #x_high = np.max(zz,axis=0)
    #x_low = [20, 0.2, 0.22, 0.25, -5, 85, 355, 450, 15, 15, -10, 16, 0.25, 0.2, 0.25, -5, 85, 450, 15, 15, 27, -15]
    #x_high = [30,0.72, 0.76 , 0.8 , 0 , 90, 380 , 600, 45 , 50,10 , 26, 0.76, 0.7 ,0.76, 0 ,90 ,600 ,60 ,50 ,35 ,5]
    bounds = pd.DataFrame(np.vstack((x_low_new,x_high_new)), columns=x_names, index=row_names)

    problem = DataProblem(data=df, variable_names=x_names, objective_names=y_names,bounds=bounds)
    start = time.time()

    #### Change the surrogate here.
    
    #problem.train(fgp)
    #problem.train(SVRsurrogate)
    problem.train(polyreg)
    #problem.train(RandomForest)
    
    end = time.time()
    time_taken = end - start
    return problem

data_scaled = scale_data(df)
nobjs = 3 
nvars = 22
surrogate_problem = build_surrogates(nobjs, nvars, data_scaled)

# Provide test data here
X_in=df.values[:,0:22]
Y_pred = surrogate_problem.evaluate(X_in,use_surrogate=True)[0]
print(Y_pred)