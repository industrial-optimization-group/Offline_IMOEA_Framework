import numpy as np
import pandas as pd
import hybrid_tree_gp
from desdeo_problem.surrogatemodels.SurrogateModels import BaseRegressor, ModelError
from sklearn import tree
import GPy
import graphviz
import datetime

class HybridTreeGP(BaseRegressor):
    def __init__(self):
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.regr = None
        self.dict_gps = {}
        self.model_htgp = None
        self.error_leaves = None

    def fit(self, X, y):
        if isinstance(X, (pd.DataFrame, pd.Series)):
            X = X.values
        if isinstance(y, (pd.DataFrame, pd.Series)):
            y = y.values

        # Make a 2-D array if needed
        y = np.atleast_1d(y)
        if y.ndim == 1:
            y = y

        self.regr = tree.DecisionTreeRegressor(max_depth=5)
        self.regr = self.regr.fit(X, y)
        dot_data = tree.export_graphviz(self.regr, out_file='tree'+str(datetime.datetime.now())+'.dot') 
        n_nodes = self.regr.tree_.node_count
        children_left = self.regr.tree_.children_left
        children_right = self.regr.tree_.children_right
        feature = self.regr.tree_.feature
        threshold = self.regr.tree_.threshold
        rmse = self.regr.tree_.impurity
        rmse_threshold = 0.01
        samples_leaf_nodes = self.regr.apply(X)

        # The tree structure can be traversed to compute various properties such
        # as the depth of each node and whether or not it is a leaf.
        node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
        is_leaves = np.zeros(shape=n_nodes, dtype=bool)
        error_leaves = None

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
                print("Number of data points:")
                print(np.shape(X_leaf)[0])
                total_point += np.shape(X_leaf)[0]
                kernel = GPy.kern.Matern52(2,ARD=True) + GPy.kern.White(2)
                m = GPy.models.GPRegression(X_leaf,Y_leaf.reshape(-1, 1),kernel=kernel)
                m.optimize('bfgs')
                self.dict_gps[str(i)] = m
            print("Total points:")
            print(total_point)
            self.error_leaves = error_leaves
        else:
            self.error_leaves = None


    def predict(self, X):
        Y_predict = self.regr.predict(X=X)
        Y_test_leaf = self.regr.apply(X)
        Y_predict_mod = Y_predict
        print("Leaf node predicted:")
        print(Y_test_leaf)
        print("Leaf node predicted by GP:")
        if self.error_leaves is not None:
            for i in range(np.shape(X)[0]):
                if Y_test_leaf[i] in self.error_leaves: 
                    print(Y_test_leaf[i])
                    Y_predict_mod[i] = self.dict_gps[str(Y_test_leaf[i])].predict(X[i].reshape(1,-1))[0][0]
        y_mean = Y_predict_mod
        y_stdev = None
        return (y_mean, y_stdev)

