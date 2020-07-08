#### Setting ####
%matplotlib inline

import itertools
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn import datasets
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split

from mlxtend.plotting import plot_learning_curves
from mlxtend.plotting import plot_decision_regions

import warnings
warnings.filterwarnings('ignore')

np.random.seed(2019311266)

#### Data ####
iris = datasets.load_iris()
X, y = iris.data[:, 0:2], iris.target

iris_plot = pd.DataFrame(data=np.c_[iris['data'],iris['target']],columns = iris['feature_names']+['target'])
iris_plot['target'] = iris_plot['target'].map({0:"setosa", 1:"versicolor",2:"virginia"})

sns.pairplot(iris_plot,x_vars=["sepal length (cm)"],y_vars=["sepal width (cm)"],hue='target', size = 5)


#### Loss fucntion ####
import plotly.graph_objects as go

def sample_loss(params):
    return cross_val_score(RandomForestClassifier(random_state=0, n_estimators=int(params[0]), max_depth=int(params[1])),
                            X, y, scoring='accuracy', cv=3).mean()

n_estimators = np.linspace(1, 10, 10)
max_depths = np.linspace(1, 10, 10)

# We need the cartesian combination of these two vectors
param_grid = np.array([[n_estimator, max_depth] for max_depth in max_depths for n_estimator in n_estimators])
real_loss = [sample_loss(params) for params in param_grid]

# The maximum is at:
param_grid[np.array(real_loss).argmax(), :]

param = pd.DataFrame(param_grid)
colorscale = [[0, 'grey'], [0.5, 'yellow'], [1, 'red']]
fig = go.Figure(data =
    go.Contour(
        z=real_loss,
        x=param[0],
        y=param[1],
        colorscale=colorscale
    ))
fig.show()
print("Maximum point is :", param_grid[np.array(real_loss).argmax(), :])


import numpy as np
import reggie as rg
from reggie import models

from pybo.bayesopt import IndexPolicy, RecPolicy
from pybo.domains import Grid
from matplotlib import pyplot as ez

__all__ = []

bounds = [[1, 10],[1, 10]]
xopt = [9, 3]
fopt = 0.7933333333333333

def init(self, method='latin', rng=None, **kwargs):
        return continuous.INITS[method](self.bounds, rng=rng, **kwargs)

domain = Grid(bounds, 10)
Xs = list(domain.init())
Y = [sample_loss(params) for params in Xs]
F = list()
F.append(0)

# policy choices: PI, EI, UCB, Thompson, PES, IPES
policy_name = 'PES'

# pybo.policies simple.py 에 계산식 있음.
if policy_name == 'PES':
    policy = IndexPolicy(domain, 'PES', {'opes': False, 'ipes': False})
elif policy_name == 'IPES':
    policy = IndexPolicy(domain, 'PES', {'opes': False, 'ipes': True})
else:
    policy = IndexPolicy(domain, policy_name)

# pybo bayesopt.py 에 계산식 있음.
recommender = RecPolicy(domain, 'observed')


# initialize the model
model = models.make_gp(0.02, 10, [1., 1.], 0)
model
# sn2 : Gaussian sigma^2
# rho : Signal variance sigma^2 in SE-ARD Kernel (overall variance?)
# ell : lengthscale parameters (Calculate distance D after rescaling X by ell)
# bias : Mean function(0=zero mean)

# set the priors and make the model sample over hyperparameters
model.params['like']['sn2'].prior = rg.core.priors.Uniform(0.01, 0.035)
model.params['kern']['rho'].prior = rg.core.priors.LogNormal(0, 100)
model.params['kern']['ell'].prior = rg.core.priors.LogNormal(0, 10)
model.params['mean']['bias'].prior = rg.core.priors.Normal(0, 20)
model

# make a meta-model to sample over models; add data
model = models.MCMC(model, n=10, burn=5, skip=True)
model.add_data(Xs, Y)
# Xs와 Y를 가지고 MCMC 시작


model.get_models()
len(model.get_models())
model.get_hypers()

# get the recommendation and the next query
xbest = recommender(model, Xs, Y)
xnext = policy(model, Xs)
ynext = cross_val_score(RandomForestClassifier(random_state=0, n_estimators=int(xnext[0]), max_depth=int(xnext[1])),X, y, scoring='accuracy', cv=3).mean()

# record our data and update the model
Xs.append(xnext)
Y.append(ynext)
F.append(cross_val_score(RandomForestClassifier(random_state=0, n_estimators=int(xbest[0]), max_depth=int(xbest[1])),X, y, scoring='accuracy', cv=3).mean())
model.add_data(xnext, ynext)

model
model.get_models()
model
model.get_models()
