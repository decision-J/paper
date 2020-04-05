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

import plotly.graph_objects as go

iris = datasets.load_iris()
X, y = iris.data[:, 0:2], iris.target

iris_plot = pd.DataFrame(data=np.c_[iris['data'],iris['target']],columns = iris['feature_names']+['target'])
iris_plot['target'] = iris_plot['target'].map({0:"setosa", 1:"versicolor",2:"virginia"})

sns.pairplot(iris_plot,x_vars=["sepal length (cm)"],y_vars=["sepal width (cm)"],hue='target', size = 5)


######## 0. Parameters setting
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


######## 1. Baseline
clf_base = RandomForestClassifier(random_state=0, n_estimators=2, max_depth=5)

label = ['Base']
clf_list = [clf_base]

fig = plt.figure(figsize=(10, 8))
gs = gridspec.GridSpec(1, 3)
grid = itertools.product([0,1,2],repeat=2)

for clf, label, grd in zip(clf_list, label, grid):
    scores = cross_val_score(clf, X, y, cv=3, scoring='accuracy')
    print ("Accuracy: %.2f (+/- %.2f) [%s]" %(scores.mean(), scores.std(), label))

    clf.fit(X, y)
    ax = plt.subplot(gs[grd[0], grd[1]])
    fig = plot_decision_regions(X=X, y=y, clf=clf, legend=2)
    plt.title(label)

plt.show()


######## 2. Randomized Search
from scipy.stats import randint
from sklearn.model_selection import RandomizedSearchCV

def hypertuning_rscv(est, p_distr, nbr_iter,X,y):
    rdmsearch = RandomizedSearchCV(est, param_distributions=p_distr,
                                  n_jobs=-1, n_iter=nbr_iter, cv=3)
    rdmsearch.fit(X,y)
    ht_params = rdmsearch.best_params_
    ht_score = rdmsearch.best_score_
    return ht_params, ht_score

clf = RandomForestClassifier(n_jobs=-1)

rf_p_dist={'n_estimators':[0,1,3,5,7,9,10,15,20],
            'max_features':[1,3,5,7,9,10] }

rf_parameters = hypertuning_rscv(clf, rf_p_dist, 5, X, y)
print(rf_parameters)

## fitting
clf_RS = RandomForestClassifier(random_state=0, n_estimators=5, max_features=1)

label = ['Randomized Search']
clf_list = [clf_RS]

fig = plt.figure(figsize=(10, 8))
gs = gridspec.GridSpec(1, 3)
grid = itertools.product([0,1,2],repeat=2)

for clf, label, grd in zip(clf_list, label, grid):
    scores = cross_val_score(clf, X, y, cv=3, scoring='accuracy')
    print ("Accuracy: %.2f (+/- %.2f) [%s]" %(scores.mean(), scores.std(), label))

    clf.fit(X, y)
    ax = plt.subplot(gs[grd[0], grd[1]])
    fig = plot_decision_regions(X=X, y=y, clf=clf, legend=2)
    plt.title(label)

plt.show()

# 정확도 약 0.1 상승






######## 3. Entropy Search
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import reggie as rg
from reggie import models
from matplotlib import pyplot as ez

from pybo.bayesopt import IndexPolicy, RecPolicy
from pybo.domains import Grid

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

# pybo recommenders.py 에 계산식 있음.
recommender = RecPolicy(domain, 'observed')

# initialize the model
model = models.make_gp(0.01, 10, [1., 1.], 0)

# set the priors and make the model sample over hyperparameters
model.params['like']['sn2'].prior = rg.core.priors.Uniform(0.05, 0.1)
model.params['kern']['rho'].prior = rg.core.priors.LogNormal(0, 100)
model.params['kern']['ell'].prior = rg.core.priors.LogNormal(0, 10)
model.params['mean']['bias'].prior = rg.core.priors.Normal(0, 20)

# make a meta-model to sample over models; add data
model = models.MCMC(model, n=20, skip=True)

model.add_data(Xs, Y)

fig = ez.figure()

n_iters = 5
for i in range(n_iters):
    print('Iteration: ' + str(i))
    # get the recommendation and the next query
    xbest = recommender(model, Xs, Y)
    xnext = policy(model, Xs)
    ynext = cross_val_score(RandomForestClassifier(random_state=0, n_estimators=int(xnext[0]), max_depth=int(xnext[1])),X, y, scoring='accuracy', cv=3).mean()

    # record our data and update the model
    Xs.append(xnext)
    Y.append(ynext)
    F.append(cross_val_score(RandomForestClassifier(random_state=0, n_estimators=int(xbest[0]), max_depth=int(xbest[1])),X, y, scoring='accuracy', cv=3).mean())
    model.add_data(xnext, ynext)

    # PLOT EVERYTHING
    fig.clear()
    ax1 = ez.subplot2grid((2, 2), (0, 0))
    ax2 = ez.subplot2grid((2, 2), (1, 0), sharex=ax1)
    ax3 = ez.subplot2grid((2, 2), (0, 1), rowspan=2)
    ax1.get_xaxis().set_visible(False)

    # plot the posterior and data
    X1, X2 = np.meshgrid(n_estimators, max_depths)
    Xs_ = np.array(Xs)

    ax1.contourf(X1, X2, np.array(real_loss).reshape(X1.shape), alpha=0.5)
    ax1.scatter(Xs_[:-1, 0],Xs_[:-1, 1], marker='.')
    ax1.scatter(xbest[0], xbest[1], linewidths=3, marker='o', color='r')
    ax1.set_title('current model (xbest and xnext)')

   # plot the acquisition function
    ax2.contourf(X1, X2, policy._index(domain.X).reshape(X1.shape), alpha=0.5)
    ax2.scatter(xbest[0], xbest[1], linewidths=3, marker='o', color='r')
    ax2.scatter(xnext[0], xnext[1], linewidths=3, marker='o', color='g')
    ax2.set_xlim(*bounds[0])
    ax2.set_ylim(*bounds[1])
    ax2.set_title('current policy (xnext)')

    # plot the latent function at recomended points
    ax3.axhline(fopt)
    ax3.plot(F)
    ax3.set_ylim(0., 1)
    ax3.set_title('Accuracy of recommendation')

    # draw
    fig.canvas.draw()
    ez.show(block=False)


print("Best parameter of Entropy Search:" "\n", x_best)
