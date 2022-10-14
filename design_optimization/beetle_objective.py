import numpy as np
from scipy.special import expit

# Cholesky decomposition of X'W X = Z'Z
#
# X    = design matrix (2d numpy array)
# beta = logistic model parameter vector (1d numpy array)
def get_chol(X, beta):
    mu = expit(np.inner(X, beta))
    weight = np.sqrt(mu*(1 - mu))
    return X * weight[:, None]

# Compute expected costs of the design
#
# x_1  = 1st phase doses (1d numpy array)
# beta = logistic model parameter vector (1d numpy array)
def cost(x_1, beta):
    return np.sum(1 / expit(beta[0] + beta[1]*x_1))

# Compute the determinant of the design matrix and the associated costs
#
# x1   = dose value vector of 1st phase treatments (1d numpy array)
# x2   = dose value vector of 2nd phase treatments (1d numpy array)
# beta = logistic model parameter vector (1d numpy array)
# n    = number of design points
# obj_scale = sign and scale of the objectives (tuple)
def design_objectives(x1, x2, beta, n, obj_scale):
    X = np.empty([n, 4])
    X[:, 0] = np.ones(n)
    X[:, 1] = x2 # Note the order here, 1st column is the dose of the current phase, i.e., x2
    X[:, 2] = x1
    # NOTE, simplified interaction (either 1 or 0), so we can focus on the magnitude of beta[3]
    X[:, 3] = (X[:, 1]*X[:, 2] > 0).astype(float)
    Z = get_chol(X, beta)
    y1 = obj_scale[0] * np.linalg.slogdet(np.matmul(Z.T, Z))[1] # log-scale might be more stable, sign is not needed
    y2 = obj_scale[1] * cost(X[:, 1], beta)
    return np.array([y1, y2])

# Compute design objectives with uncertainty (assuming gaussian priors)
#
# x          = dose value vector of both treatments (1d numpy array, length = 2*n)
# beta_mean  = prior mean of model parameters (1d numpy array)
# beta_sd    = prior sd of model parameters (1d numpy array)
# n_eval     = number of times to evaluate the objective
# obj_scale  = can be used to change the sign and scale of the objectives, if needed.
#              (default values assume optimizer tries to minimize all objectives)
# seed       = RNG seed
def objective_unc(x, beta_mean, beta_sd, n_eval, obj_scale = (-1, 1), seed = 123):
    m = np.shape(x)[0]
    n = m // 2
    objs = np.empty([n_eval, 2])
    rng = np.random.default_rng(seed)
    beta_len = np.shape(beta_mean)[0]
    beta = np.empty([n_eval, beta_len])
    #for j in range(beta_len):
    #    beta[:,j] = rng.normal(beta_mean[j], beta_sd[j], n_eval)
    for j in range(beta_len-1):
        beta[:,j] = np.repeat(beta_mean[j], n_eval)
    beta[:,3] = np.random.rand(n_eval)-0.5
    for i in range(n_eval):
        objs[i,:] = design_objectives(x[0:n], x[n:m], beta[i,:], n, obj_scale)
    return objs
    #return np.array([np.mean(objs[:, 0]), np.std(objs[:, 0]), np.mean(objs[:, 1]), np.std(objs[:, 1])])
