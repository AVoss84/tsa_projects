#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
#from scipy.stats import wishart, multivariate_normal, bernoulli, multinomial
#from scipy.sparse import csr_matrix
#from sklearn.model_selection import train_test_split
import os, pickle
import numpy as np
import math
#from numpy import log, sum, exp, prod
from numpy.random import beta, binomial, normal, dirichlet, uniform, gamma, seed, multinomial, gumbel, rand
from imp import reload
from copy import deepcopy
from math import cos, pi
#import seaborn as sns
import pandas as pd
#import time
#from scipy.spatial.distance import euclidean
#import itertools
#from itertools import chain, combinations
import tsa_utils as utils

from bayes_regression import variational_linear_regression as bayes

reload(utils)


def cycle(N, nof_cycles = 1):
  return np.cos(2*pi*np.arange(0,N)*nof_cycles/N)

def simAR1(N, phi, sigma, const = 0, burn=100):
  y = np.zeros((N+burn))
  for t in range(N+burn-1):
    y[t+1] = const + phi*y[t] + normal(scale = sigma, size=1)     
  return y[burn:]

#np.random.seed(0)   # set seed

N = 1*10**2
omega = 1
phi_true = 0.77
sigma_true = 0.5

N1 = 5*10**1
N2 = 5*10**1
N3 = 5*10**1

omega = 1
phi_true = 0.77
sigma_true = 0.5

cps = [N1+1]

#cps = [N1+1, N1+N2+1]

y1 = simAR1(N1, phi = phi_true, sigma = sigma_true, const = 0)    # Regime 1
y2 = simAR1(N2, phi = .4*phi_true, sigma = sigma_true, const = 1.5)    # Regime 2
y3 = simAR1(N3, phi = phi_true, sigma = 0.9*sigma_true, const = 0.2)    # Regime 2

y = np.concatenate((y1, y2),axis=0)              # data with level shift

n = len(y)
r = 1
nNew = n-r


X_lagged = utils.embed(pd.DataFrame(y, columns=['y']), lags = r)
#X_lagged.tail()
print(X_lagged.shape)

LeftY = X_lagged[['y_lag0']].to_numpy()
X_lagged.drop(columns=['y_lag0'], inplace=True)

RiteX = X_lagged.to_numpy()
print(RiteX.shape)
#RiteX['const'] = 1.


#breg = bayes.BayesianRegression()
breg = bayes.VariationalLinearRegression()

breg.fit(X = RiteX, t = LeftY)

yhat, yhat_std = breg.predict(X = RiteX, return_std = True)

rmse = np.sqrt(np.average((LeftY - yhat)**2))
rmse