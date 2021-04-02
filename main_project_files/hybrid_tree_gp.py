import numpy as np
import pandas as pd
#from pygmo import problem
from pyDOE import lhs
from desdeo_problem.testproblems.TestProblems import test_problem_builder
from sklearn import tree
import matplotlib.pyplot as plt
import GPy

def fit(X,y):
    m = hybrid_tree_gp()
    m.fitx(X,y)
    return m


class hybrid_tree_gp():

    def __init__(self):
        self.regr = None
        self.dict_gps = {}
        self.model_htgp = None
        self.error_leaves = None

    def fitx(self,X,y):
        self.regr = tree.DecisionTreeRegressor(max_depth=2)
        self.regr = self.regr.fit(X, y)
        n_nodes = self.regr.tree_.node_count
        children_left = self.regr.tree_.children_left
        children_right = self.regr.tree_.children_right
        feature = self.regr.tree_.feature
        threshold = self.regr.tree_.threshold
        rmse = self.regr.tree_.impurity
        rmse_threshold = 0.001
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
                        error_leaves = node_id
                    else:
                        error_leaves = np.append(error_leaves,node_id)
        leaf_error_nodes_dict = {}
        for i in error_leaves:
            leaf_error_nodes_dict[str(i)] = np.where(samples_leaf_nodes==i)[0]
        

        for i in error_leaves:
            loc_leaf = np.where(samples_leaf_nodes==i)[0]
            X_leaf = X[loc_leaf]
            Y_leaf = y[loc_leaf]
            kernel = GPy.kern.Matern52(2,ARD=True) + GPy.kern.White(2)
            m = GPy.models.GPRegression(X_leaf,Y_leaf.reshape(-1, 1),kernel=kernel)
            m.optimize('bfgs')
            self.dict_gps[str(i)] = m
        self.error_leaves = error_leaves

        return self

    def predict(self, X_test):
        Y_predict = self.regr.predict(X=X_test)
        Y_test_leaf = self.regr.apply(X_test)
        Y_predict_mod = Y_predict
        for i in range(np.shape(X_test)[0]):
            if Y_test_leaf[i] in self.error_leaves: 
                Y_predict_mod[i] = self.dict_gps[str(Y_test_leaf[i])].predict(X_test[i].reshape(1,-1))[0][0]
        return Y_predict_mod


