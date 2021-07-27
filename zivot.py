import matplotlib.pyplot as plt
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
import time
#from scipy.spatial.distance import euclidean
import itertools
from itertools import chain, combinations
import tsa_utils as utils

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
cps

burn = 300
gibbs = 2000
nofb = 1
r = 1
fixbreaks = None
conflevel =.05
p = 1

X_lagged = utils.embed(pd.DataFrame(y, columns=['y']), lags = p)
X_lagged.tail()

print(X_lagged.shape)
lhs = X_lagged[['y_lag0']].to_numpy()
X_lagged.drop(columns=['y_lag0'], inplace=True)
rhs = X_lagged.to_numpy()
print(rhs.shape)
#X_til['const'] = 1.


n = len(y)
tau = 3

n - p
k = [0, np.floor((n - 1) * 0.5), n - p]     # single break only!
k = [0, 20, 70, n - p]     
k

range(k[len(k)-1])
k[len(k)-1]

tim = np.array(range(k[len(k)-1]))
ei1 = np.zeros((k[len(k)-1], len(k) - 1))
ei2 = np.zeros((k[len(k)-1], len(k) - 1))
ei1.shape
len(tim)

i = 1
while i < len(k):
  ei1[:, i - 1] = (k[i - 1] <= tim) & (tim < k[i])
  ei2[:, i - 1] = ei1[:, i - 1] * tim
  i += 1
ei1
ei2


def indBrkL(k : np.array):
  """
	Break indicator matrices for drift/trend breaks
	"""
  k[0] = 0
  nofb = len(k) - 2    # substract start and end date
	tim = np.array(range(k[len(k)-1]))
	ei1 = np.zeros((k[len(k)-1], len(k) - 1))    # drift
	ei2 = np.zeros((k[len(k)-1], len(k) - 1))    # trend
	i = 1
	while i < len(k):
		ei1[:, i - 1] = (k[i - 1] <= tim) & (tim < k[i])     # drift break
		ei2[:, i - 1] = ei1[:, i - 1] * tim     # trend break
		i += 1
	drift = pd.DataFrame(ei1, columns=['Drift_'+str(s) for s in range(nofb+1)])	
	trend = pd.DataFrame(ei2, columns=['Trend_'+str(s) for s in range(nofb+1)])	
	return	dict(ei=np.concatenate((ei1,ei2), axis=1), ei1=ei1)



kBeg = k[0] + tau + p
print(kBeg)
kEnd = k[2] - 1
print(kEnd)
prob = np.zeros(kEnd - kBeg + 1)


beta = np.linalg.pinv(rhs.T @ rhs) @ rhs.T @ lhs
beta











