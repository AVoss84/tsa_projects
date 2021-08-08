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

n = len(y)
burn = 300
gibbs = 2000
nofb = 2
r = 1
nNew = n-r
fixbreaks = None
conflevel =.05
#p = 1


X_lagged = utils.embed(pd.DataFrame(y, columns=['y']), lags = r)
#X_lagged.tail()
print(X_lagged.shape)

LeftY = X_lagged[['y_lag0']].to_numpy()
X_lagged.drop(columns=['y_lag0'], inplace=True)

RiteX = X_lagged.to_numpy()
print(RiteX.shape)
#X_til['const'] = 1.

tau = 3

#k = [0, np.floor((n - 1) * 0.5), n - r]     # single break only!
#k = [0, 20, 70, n - r]     

reload(utils)

# Initialisations:
#-------------------#
# Specify the hyperparameters:
# Multivariate Normal prior for beta ("B")
#-----------------------------------------#
a0 = np.zeros((2*nofb+2+r,1)) 		     #mean of betas
siga = np.eye(2*nofb+2+r)/1000   # Inverse prior covariance matrix of betas

# Gamma Prior for sigma :
#-------------------------#
v0 = 2.001 ; d0 = 0.001 

# Step 1:
#---------
# Prior specification of break dates according to a uniform distr.:
# vector of initial break dates; Dimension: m+2!!  
k = np.concatenate((np.array([0]), np.array(range(1, nofb+1))*np.floor(nNew/(nofb+1)), np.array([nNew]))).astype(int)

print("\nInitial break dates:",k,"\n")

ei = utils.indBrkL(k)['ei']   #Indicator matrix for timing of breaks
ei1 = utils.indBrkL(k)['ei1']   
#ei.shape
#ei1.shape
#print(RiteX.shape)
X = np.concatenate((ei,RiteX), axis=1) 
#pd.DataFrame(X).head()
b = np.linalg.pinv(X.T @ X) @ X.T @ LeftY
err = LeftY - X @ b 				#Ols errors

# Initialise Standard errors of beta_hats in each break regime:
#---------------------------------------------------------------#
std = np.zeros((nofb+1,1))		#standard errors (see definition in paper)

for i in range(1,(nofb+2)):
    u2i = err[k[i-1]:(k[i]-1)]  #errors only using data in break interval i	
    std[i-1] = np.sqrt((u2i.T @ u2i)/(k[i]-k[i-1]))    # sigma_i's

########### Gibbs-Sampling #####################
################################################

M = burn + gibbs;		#number of iterations

g=1
while g <= M :       #Gibbs iterations
    
#Step 2:
#----------
i = 1					
kBeg = k[i-1]+2;					#defining lower and upper margins of break regime i
kEnd = k[i+1]-1;				#initial upper margin     			
#initial lower margin (vgl. Ursprungsfile + 2)		
prob = np.zeros((kEnd-kBeg+1)) 	#vector of logLikelihood values in break interval i
prob

j = kBeg;   			 #running all dates between k_i = [kBeg;kEnd]
while j < kEnd :   
    #print(j)
    k[i] = j         #flexible upper margin up to maximum "kEnd"
    iVar = utils.indBrkL(k)['ei1']  		
    #update indicator matrix!
    sigma = iVar @ std      			#Dimension: (T-r) x 1 
    #calculate vector with residual stand.dev. s_t = sigma_i (-> Denominator)
    #with std unchanged      
    sigma2in = 1/(sigma[kBeg:kEnd]*sigma[kBeg:kEnd]) 
    ei = utils.indBrkL(k)['ei']   			

    #configure indicator matrix according to break date kBeg
    X = np.concatenate((ei,RiteX), axis=1)     
    err = LeftY[kBeg:kEnd] - X[kBeg:kEnd,] @ b 
    #using only data between k_i-1 and k_i+1; see equation (4) in ZW(2000) (-> Numerator)
    prob[j-k[i-1]-2] = - np.sum(np.log(sigma[kBeg:kEnd])) -.5*np.sum(err*err*sigma2in)
    j = j + 1

prob = np.exp(prob-np.max(prob))  	 #likelihood values (strictly increasing transformation)     		
prob = prob/np.sum(prob)	
prob

# Draw k_i from multinom. cond. posterior and set new upper margin
#-------------------------------------------------------------------
postki = np.argmax(multinomial(len(prob), pvals=prob))
k[i] = postki + k[i-1] + 1

#Step i
#--------
i = 3
while i <= (nofb+1):			#over all possible break regimes
  kBeg = k[i-1]+1; 
  kEnd = k[i+1]-1;			   				
  prob = np.zeros((kEnd-kBeg+1)) 	#vector of logLikelihood values in break interval i

  j = kBeg;   			 #running all break dates between k_i = [kBeg;kEnd]
  while j < kEnd :   
      #print(j)
      k[i] = j         #flexible upper margin up to maximum "kEnd"
      iVar = utils.indBrkL(k)['ei1']  		
      #update indicator matrix!
      sigma = iVar @ std      			#Dimension: (T-r) x 1 
      #calculate vector with residual stand.dev. s_t = sigma_i (-> Denominator)
      #with std unchanged      
      sigma2in = 1/(sigma[kBeg:kEnd]*sigma[kBeg:kEnd]) 
      ei = utils.indBrkL(k)['ei']   			

      #configure indicator matrix according to break date kBeg
      X = np.concatenate((ei,RiteX), axis=1)     
      err = LeftY[kBeg:kEnd] - X[kBeg:kEnd,] @ b 
      #using only data between k_i-1 and k_i+1; see equation (4) in ZW(2000) (-> Numerator)
      prob[j-k[i-1]-2] = - np.sum(np.log(sigma[kBeg:kEnd])) -.5*np.sum(err*err*sigma2in)
      j = j + 1

  prob = np.exp(prob-np.max(prob))  	 #likelihood values (strictly increasing transformation)     		
  prob = prob/np.sum(prob)	
  prob

# Draw k_i from multinom. cond. posterior and set new upper margin
#-------------------------------------------------------------------#				     
postki = np.argmax(multinomial(len(prob), pvals=prob))
k[i] = postki + k[i-1]

# Step m+2:
#-----------
iVar = utils.indBrkL(k)['ei1']
ei = utils.indBrkL(k)['ei']
X = np.concatenate((ei,RiteX), axis=1) 
s = (iVar @ std).squeeze()
sigma = np.eye(nNew) * np.tile(s, (nNew,1))
sigma2in = np.linalg.inv(sigma*sigma)

tvar = siga + X.T @ sigma2in.T @ X
tmean = np.linalg.inv(tvar) @ (siga @ a0 + X.T @ sigma2in.T @ LeftY).squeeze()  #with Prior Mean B0 = 0

# Draw from conditional posterior distr. of Beta:
#--------------------------------------------------
b = np.random.multivariate_normal(mean = tmean, cov = np.linalg.inv(tvar), size=1).T
err = LeftY - X @ b


#Step m+3:
#------------
for i in range(1, nofb+2):
  u2i = err[k[i-1]:(k[i]-1)]  #errors only using data in break interval i	
  v = v0 + (k[i]-k[i-1])/2
  d = d0 + u2i.T @ u2i
  std[i-1] = 1/np.sqrt(np.random.gamma(shape=v, scale=2/d[0], size=1))

s = (iVar @ std).squeeze()
sigma = np.eye(nNew) * np.tile(s, (nNew,1))    #Updaten für Berechnung der marginalen Likelihood unten
sigma2in = np.linalg.inv(sigma*sigma)

#k = np.concatenate((np.array([0]), np.array(range(1, nofb+1))*np.floor(nNew/(nofb+1)), np.array([nNew]))).astype(int)

#k = c(1, k[-c(1,length(k))] + r, k[length(k)]);     # Correct break dates for lagging







