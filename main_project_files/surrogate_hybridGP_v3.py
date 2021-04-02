import numpy as np
import pandas as pd
import hybrid_tree_gp
from desdeo_problem.surrogatemodels.SurrogateModels import BaseRegressor, ModelError
from sklearn import tree
import GPy
import graphviz
import datetime

######## New htgp 

class HybridTreeGP_v2(BaseRegressor):
    def __init__(self, nd_points=None):
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.regr = None
        self.dict_gps = {}
        self.model_htgp = None
        self.error_leaves = None
        self.total_point = 0
        self.total_point_gps = 0
        self.nd_points = nd_points
        self.num_nd_points = None
        self.unique_nd_leaf_nodes = None
        self.freq_nd_leaf_nodes = None

    def fit(self, X, y):
        if isinstance(X, (pd.DataFrame, pd.Series)):
            X = X.values
        if isinstance(y, (pd.DataFrame, pd.Series)):
            y = y.values

        # Make a 2-D array if needed
        y = np.atleast_1d(y)
        if y.ndim == 1:
            y = y
        self.X = X
        self.y = y
        #self.regr = tree.DecisionTreeRegressor(max_depth=100, min_samples_leaf=20)
        self.regr = tree.DecisionTreeRegressor(max_depth=100, min_samples_leaf=219)
        self.regr = self.regr.fit(X, y)
        #dot_data = tree.export_graphviz(self.regr, out_file='tree'+str(datetime.datetime.now())+'.dot') 
        
        
        n_nodes = self.regr.tree_.node_count
        children_left = self.regr.tree_.children_left
        children_right = self.regr.tree_.children_right
        feature = self.regr.tree_.feature
        threshold = self.regr.tree_.threshold
        rmse = self.regr.tree_.impurity
        rmse_threshold = 1
        samples_leaf_nodes = self.regr.apply(X)
        self.samples_leaf_nodes = samples_leaf_nodes
        # The tree structure can be traversed to compute various properties such
        # as the depth of each node and whether or not it is a leaf.
        node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
        is_leaves = np.zeros(shape=n_nodes, dtype=bool)
        error_leaves = None


        # Cut the clutter here
        stack = [(0, -1)]  # seed is the root node id and its parent depth
       
        while len(stack) > 0:
            node_id, parent_depth = stack.pop()
            node_depth[node_id] = parent_depth + 1

            # If we have a test node
            if (children_left[node_id] != children_right[node_id]):
                stack.append((children_left[node_id], parent_depth + 1))
                stack.append((children_right[node_id], parent_depth + 1))
            else:
                is_leaves[node_id] = True
                if rmse[node_id] > rmse_threshold:
                    if error_leaves is None:
                        error_leaves = [node_id]
                    else:
                        error_leaves = np.append(error_leaves,node_id)
      
        leaf_error_nodes_dict = {}
        if error_leaves is not None:
            for i in error_leaves:
                leaf_error_nodes_dict[str(i)] = np.where(samples_leaf_nodes==i)[0]
            
            total_point=0
            for i in error_leaves:
                loc_leaf = np.where(samples_leaf_nodes==i)[0]
                X_leaf = X[loc_leaf]
                Y_leaf = y[loc_leaf]
                print("Number of data points:",np.shape(X_leaf)[0])
                total_point += np.shape(X_leaf)[0]
                #kernel = GPy.kern.Matern52(10,ARD=True) #+ GPy.kern.White(2)
                m = GPy.models.GPRegression(X_leaf,Y_leaf.reshape(-1, 1))#,kernel=kernel)
                m.optimize('bfgs')
                self.dict_gps[str(i)] = m
            print("Total points:",total_point)
            self.error_leaves = error_leaves
        else:
            self.error_leaves = None
        
        # Find the non-dominated samples present at each leaf
        #print("Non-dom indices:", self.nd_points)
        #print("Sample leaf node indices:", self.samples_leaf_nodes)
        self.unique_nd_leaf_nodes, self.freq_nd_leaf_nodes = np.unique(self.samples_leaf_nodes[self.nd_points],return_counts=True)




    def addGPs(self, X_solutions):

        Y_solution_leaf = self.regr.apply(X_solutions)
        #sizeinit = np.shape(Y_solution_leaf)[0]
        #if self.error_leaves is not None:
        #    Y_solution_leaf = [i for i in Y_solution_leaf if i not in self.error_leaves]
        #    Y_solution_leaf = np.delete(Y_solution_leaf,np.where(Y_solution_leaf == self.error_leaves))  
        #print("Reduction=",(sizeinit - np.shape(Y_solution_leaf)[0]))
        
        unique_solutions, count_solutions = np.unique(Y_solution_leaf, return_counts=True)
        unique_solutions = np.setdiff1d(unique_solutions,self.error_leaves)
        #print("Reduction=",(sizeinit -np.shape(unique_solutions)[0]))
        self.total_point_gps = unique_solutions.size
        if unique_solutions.size > 0:
            
            # Taking max MSE
            mse_solutions = self.regr.tree_.impurity[unique_solutions]
            arg_max_mse = np.argmax(mse_solutions)
            # Taking max points
            #print("Count:",count_solutions)
            #arg_max_mse = np.argmax(count_solutions)
            #print("Max freq:",np.max(count_solutions))
            #if np.max(count_solutions)<=3:
            #    mse_solutions = self.regr.tree_.impurity[unique_solutions]
            #    arg_max_mse = np.argmax(mse_solutions)
            if self.error_leaves is None:
                self.error_leaves = [unique_solutions[arg_max_mse]]
            else:
                self.error_leaves = np.append(self.error_leaves,unique_solutions[arg_max_mse])
            loc_leaf = np.where(self.samples_leaf_nodes==unique_solutions[arg_max_mse])[0]
            X_leaf = self.X[loc_leaf]
            Y_leaf = self.y[loc_leaf]
            #print("Leaf node index : ", self.error_leaves)
            #print("Number of data points : ",np.shape(X_leaf)[0])
            self.total_point += np.shape(X_leaf)[0]
            #print("Total points : ",self.total_point)
            #kernel = GPy.kern.Matern52(10,ARD=True) #+ GPy.kern.White(2)
            m = GPy.models.GPRegression(X_leaf,Y_leaf.reshape(-1, 1))#,kernel=kernel)
            m.optimize('bfgs')
            self.dict_gps[str(unique_solutions[arg_max_mse])] = m


    # For nondom samples
    def addGPs_max_ndsamples(self, X_solutions):

        Y_solution_leaf = self.regr.apply(X_solutions)
        #sizeinit = np.shape(Y_solution_leaf)[0]
        #if self.error_leaves is not None:
        #    Y_solution_leaf = [i for i in Y_solution_leaf if i not in self.error_leaves]
        #    Y_solution_leaf = np.delete(Y_solution_leaf,np.where(Y_solution_leaf == self.error_leaves))  
        #print("Reduction=",(sizeinit - np.shape(Y_solution_leaf)[0]))
        
        unique_solutions, count_solutions = np.unique(Y_solution_leaf, return_counts=True)
        unique_solutions = np.setdiff1d(unique_solutions,self.error_leaves)
        #print("Reduction=",(sizeinit -np.shape(unique_solutions)[0]))
        self.total_point_gps = unique_solutions.size
        if unique_solutions.size > 0:
            
            # Taking max MSE
            #mse_solutions = self.regr.tree_.impurity[unique_solutions]
            #arg_max_mse = np.argmax(mse_solutions)
            # Taking max points
            #print("Count:",count_solutions)
            #arg_max_mse = np.argmax(count_solutions)
            #print("Max freq:",np.max(count_solutions))
            
            
            index_predicted_leafs_nd = np.where(np.in1d(self.unique_nd_leaf_nodes, unique_solutions))[0]
            if index_predicted_leafs_nd.shape[0] > 0:
                index_temp = np.argmax(self.freq_nd_leaf_nodes[index_predicted_leafs_nd])
                max_leaf_node = (self.unique_nd_leaf_nodes[index_predicted_leafs_nd])[index_temp]
                arg_max_mse = np.where(unique_solutions==max_leaf_node)[0][0]
                #print("max nd leaf:", max_leaf_node)
            else:
                mse_solutions = self.regr.tree_.impurity[unique_solutions]
                arg_max_mse = np.argmax(mse_solutions)
            #if np.max(count_solutions)<=3:
            #    mse_solutions = self.regr.tree_.impurity[unique_solutions]
            #    arg_max_mse = np.argmax(mse_solutions)

            if self.error_leaves is None:
                self.error_leaves = [unique_solutions[arg_max_mse]]
            else:
                self.error_leaves = np.append(self.error_leaves,unique_solutions[arg_max_mse])
            loc_leaf = np.where(self.samples_leaf_nodes==unique_solutions[arg_max_mse])[0]
            X_leaf = self.X[loc_leaf]
            Y_leaf = self.y[loc_leaf]
            #print("Leaf node index : ", self.error_leaves)
            #print("Number of data points : ",np.shape(X_leaf)[0])
            self.total_point += np.shape(X_leaf)[0]
            #print("Total points : ",self.total_point)
            #kernel = GPy.kern.Matern52(10,ARD=True) #+ GPy.kern.White(2)
            m = GPy.models.GPRegression(X_leaf,Y_leaf.reshape(-1, 1))#,kernel=kernel)
            m.optimize('bfgs')
            self.dict_gps[str(unique_solutions[arg_max_mse])] = m

    def predict(self, X):
        Y_predict = self.regr.predict(X=X)
        Y_test_leaf = self.regr.apply(X)
        Y_predict_mod = Y_predict
        #print("Leaf node predicted:",Y_test_leaf)
        #print("Leaf node predicted by GP:")
        if self.error_leaves is not None:
            
            for i in range(np.shape(X)[0]):
                if Y_test_leaf[i] in self.error_leaves:
                    #print(Y_test_leaf[i])                    
                    Y_predict_mod[i] = self.dict_gps[str(Y_test_leaf[i])].predict(X[i].reshape(1,-1))[0][0]
            
            #loc_gps = np.in1d(Y_test_leaf,self.error_leaves)
          
        y_mean = Y_predict_mod
        y_stdev = None
        #print('y_mean:',y_mean)  
        return (y_mean, y_stdev)

    def predict_new(self, X):
        Y_predict = self.regr.predict(X=X)
        Y_test_leaf = self.regr.apply(X)
        Y_predict_mod = Y_predict
        if self.error_leaves is not None: 
            Y_intersecting_nodes = np.intersect1d(Y_test_leaf,self.error_leaves)
            #print("GP nodes present:",self.error_leaves)
            #print("Leaf nodes predicted:",Y_test_leaf)
            #print("Intersecting nodes:",Y_intersecting_nodes)                    
            for i in Y_intersecting_nodes:
                loc = np.where(Y_test_leaf==i)[0]
                #print("Element location:",loc)
                X_loc = X[loc]
                #Y_predict_mod[loc] = self.dict_gps[str(i)].predict(X_loc.reshape(1,-1))[0][0]
                Y_predict_mod[loc] = self.dict_gps[str(i)].predict(X_loc)[0].flatten()
        y_mean = Y_predict_mod
        y_stdev = None
        return (y_mean, y_stdev)
