#!/usr/bin/env python
# coding: utf-8

# # SVM 가지고 2 dimension BO example

# In[9]:


from sklearn.datasets import make_classification

data, target = make_classification(n_samples=2500,
                                   n_features=45,
                                   n_informative=15,
                                   n_redundant=5)


# In[14]:


from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

def sample_loss(params):
  C = params[0]
  gamma = params[1]

  # Sample C and gamma on the log-uniform scale
  model = SVC(C=10 ** C, gamma=10 ** gamma, random_state=12345)

  # Sample parameters on a log scale
  return cross_val_score(SVC(C=10 ** params[0], gamma=10 ** params[1], random_state=12345),
                          X=data, y=target, scoring='roc_auc', cv=3).mean()


# In[16]:


data, target = make_classification(n_samples=2500,
                                   n_features=45,
                                   n_informative=15,
                                   n_redundant=5)

def sample_loss(params):
    return cross_val_score(SVC(C=10 ** params[0], gamma=10 ** params[1], random_state=12345),
                           X=data, y=target, scoring='roc_auc', cv=3).mean()


# In[33]:


target


# In[17]:


lambdas = np.linspace(1, -4, 25)
gammas = np.linspace(1, -4, 20)

# We need the cartesian combination of these two vectors
param_grid = np.array([[C, gamma] for gamma in gammas for C in lambdas])

real_loss = [sample_loss(params) for params in param_grid]

# The maximum is at:
param_grid[np.array(real_loss).argmax(), :]


# In[22]:



get_ipython().run_line_magic('matplotlib', 'inline')

# Load the Python scripts that contain the Bayesian optimization code
get_ipython().run_line_magic('run', './../python/gp.py')
get_ipython().run_line_magic('run', './../python/plotters.py')


# In[19]:



from matplotlib import rc
rc('text', usetex=True)


# In[28]:


bounds = np.array([[-4, 1], [-4, 1]])

xp, yp = bayesian_optimisation(n_iters=30, 
                               sample_loss=sample_loss, 
                               bounds=bounds,
                               n_pre_samples=3,
                               random_search=100000)


# In[30]:


rc('text', usetex=False)
plot_iteration(lambdas, xp, yp, first_iter=3, second_param_grid=gammas, optimum=[0.58333333, -2.15789474])


# In[ ]:




