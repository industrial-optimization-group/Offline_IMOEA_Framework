#%%
import GPy
import numpy as np
np.random.seed(101)
N = 50
noise_var = 0.05

X = np.linspace(0,10,50)[:,None]
k = GPy.kern.RBF(1)
y = np.random.multivariate_normal(np.zeros(N),k.K(X)+np.eye(N)*np.sqrt(noise_var)).reshape(-1,1)

print(X)
print(y)


m_full = GPy.models.GPRegression(X,y)
m_full.optimize('bfgs')
m_full.plot()
print(m_full)

Z = np.hstack((np.linspace(2.5,4.,3),np.linspace(7,8.5,3)))[:,None]
m = GPy.models.SparseGPRegression(X,y,Z=Z)
m.likelihood.variance = noise_var
m.plot()
print(m)

m.inducing_inputs.fix()
m.optimize('bfgs')
m.plot()
print(m)

m.Z.unconstrain()
m.optimize('bfgs')
m.plot()


# %%
