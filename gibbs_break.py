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
#from scipy.spatial.distance import euclidean
import tsa_utils as utils
from tqdm import tqdm

# compare: zivot_wang_gibbs_analysis.R

def zivot(y, burn = 300, gibbs = 2000, tau = 3, nofb = 1, r = 1):

    n = len(y)
    nNew = n-r
    M = burn + gibbs;		#number of iterations

    X_lagged = utils.embed(pd.DataFrame(y, columns=['y']), lags = r)
    LeftY = X_lagged[['y_lag0']].to_numpy()
    X_lagged.drop(columns=['y_lag0'], inplace=True)
    RiteX = X_lagged.to_numpy()
    #X_til['const'] = 1.
    k_draws = np.zeros((M,nofb+2))

    # Initialisations:
    #-------------------
    # Specify the hyperparameters:
    # Multivariate Normal prior for beta ("B")
    #------------------------------------------
    a0 = np.zeros((2*nofb+2+r,1)) 		     #mean of betas
    siga = np.eye(2*nofb+2+r)/1000   # Inverse prior covariance matrix of betas

    # Gamma Prior for sigma :
    #-------------------------
    v0 = 2.001 ; d0 = 0.001 

    # Step 1:
    #---------
    # Prior specification of break dates according to a uniform distr.:
    # vector of initial break dates; Dimension: m+2!!  
    k = np.concatenate((np.array([0]), np.array(range(1, nofb+1))*np.floor(nNew/(nofb+1)), np.array([nNew]))).astype(int)
    print("\nInitial break dates:",k,"\n")

    ei = utils.indBrkL(k)['ei']   #Indicator matrix for timing of breaks
    ei1 = utils.indBrkL(k)['ei1']   
    X = np.concatenate((ei,RiteX), axis=1) 
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

    for g in tqdm(range(M)):             #Gibbs iterations

        k_draws[g,:] = k     # save draws
       
        #Step 2:
        #----------
        i = 1					
        kBeg = k[i-1]+2;					#defining lower and upper margins of break regime i
        kEnd = k[i+1]-1;				#initial upper margin     				
        prob = np.zeros((kEnd-kBeg+1)) 	#vector of logLikelihood values in break interval i

        j = kBeg;   			 #running all dates between k_i = [kBeg;kEnd]
        while j < kEnd :   
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

        # Draw k_i from multinom. cond. posterior and set new upper margin
        #-------------------------------------------------------------------
        postki = np.argmax(multinomial(len(prob), pvals=prob))
        k[i] = postki + k[i-1] + 1

        # Step i
        #--------
        i = 2
        while i <= nofb:			#over all possible break regimes
            kBeg = k[i-1]+1; 
            kEnd = k[i+1]-1;			   				
            prob = np.zeros((kEnd-kBeg+1)) 	#vector of logLikelihood values in break interval i

            j = kBeg;   			 #running all break dates between k_i = [kBeg;kEnd]
            while j < kEnd :   
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
                prob[j-k[i-1]-2] = -np.sum(np.log(sigma[kBeg:kEnd])) -.5*np.sum(err*err*sigma2in)
                j = j + 1

            prob = np.exp(prob-np.max(prob))  	 #likelihood values (strictly increasing transformation)     		
            prob = prob/np.sum(prob)	

            # Draw k_i from multinom. cond. posterior and set new upper margin
            #-------------------------------------------------------------------#				     
            postki = np.argmax(multinomial(len(prob), pvals=prob))
            k[i] = postki + k[i-1]
            i = i + 1

            print('k draws {}'.format(k))

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
            for t in range(1, nofb+2):
                u2i = err[k[t-1]:(k[t]-1)]  #errors only using data in break interval i	
                v = v0 + (k[t]-k[t-1])/2
                d = d0 + u2i.T @ u2i
                std[t-1] = 1/np.sqrt(np.random.gamma(shape=v, scale=2/d[0], size=1))

            s = (iVar @ std).squeeze()
            sigma = np.eye(nNew) * np.tile(s, (nNew,1))    #Updaten für Berechnung der marginalen Likelihood unten
            sigma2in = np.linalg.inv(sigma*sigma)

    return k_draws[burn:,]












