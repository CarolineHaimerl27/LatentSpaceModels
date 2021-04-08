# created by Pedro Herrero-Vidal

import numpy as np
import random

def FA_EM(X_cov, xDim, zDim, eps, T, penalty=3, TL_flag=0):
    A = abs(np.random.uniform(0,1, (xDim,zDim))/np.sqrt(zDim))  # initiate A
    R = np.diag(np.diag(X_cov))                                 # initiate R           

    # eps = 1e-7                                                # define stopping value
    LL_prev = 0                                                 # initiate LL reference
    LL_step = eps+1                                             # non-stop value for while loop
    LL_cache = []                                               # array with LL values
    counter = 0 
    while LL_step > eps:                                        # EM FA
        # E-step
        delta = np.linalg.pinv(A @ A.T + R)
        beta = A.T @ delta

        # M-step
        A = (X_cov @ beta.T @ 
             np.linalg.pinv(np.identity(zDim) - beta @ A + beta @ X_cov @ beta.T))
        R = np.diag(np.diag(X_cov - A @ beta @ X_cov))

        # avg. LL
        if np.linalg.slogdet(delta)[0] > 0:
            LL = -T/2*np.trace(delta @ X_cov) + T/2*np.linalg.slogdet(delta)[1] - T*xDim/2*np.log(2*np.pi) 
                                            # N*sum(log(diag(chol(MM))))
        elif np.linalg.slogdet(delta)[0] < 0:
#             print(str(zDim)+'Negative determinant')
            LL = -T/2*np.trace(delta @ X_cov) + T/2*np.linalg.slogdet(delta)[1] - T*xDim/2*np.log(2*np.pi) 
                                                                
        LL_step = abs((LL-LL_prev)/abs(LL))
        LL_prev = LL
        LL_cache.append(LL)
        counter += 1
        if counter > 1e4:
            break

    LL_corrected = LL #- zDim ** penalty;

    return A, R, LL_cache, LL_corrected

def FA_project(A, R, X, X_mu):
    return A.T @ np.linalg.pinv(A @ A.T + R) @ (X.T - np.tile(X_mu, (X.shape[0], 1)).T)


def FA_LNOCV(C, R, X):
    '''
    Performs leave-neuron-out (LNO) error for factor analysis
    @ arguments:
    - C loading factors: 2D numpy array [xDim x zDim]
    - R observation noise: 2D numpy array [xDim x xDim]
    - X data: 2D numpy array [observations x xDim]
    - X_mu data mean: 1D numpy array [xDim]
    @ output:
    - err LNO-CV error: scalar
    '''
    xDim, zDim = C.shape
    I = np.eye(zDim)

    Xcs = np.zeros((X).shape)*np.nan
#     Vcs = np.zeros((xDim, 1))*np.nan

    for ii in range(xDim):
        idx = np.ones(xDim, dtype='bool')
        idx[ii] = False

        Rinv   = 1 / np.diag(R)[idx]                                # [ xDim-1 x 1      ]
        CRinv  = (C[idx, :] * (np.tile(Rinv, (zDim, 1)).T)).T       # [   zDim x xDim-1 ]
        CRinvC = np.dot(CRinv, C[idx, :])                           # [   zDim x zDim   ]

        term2  = np.dot(C[ii, :], (I - np.dot(CRinvC , np.linalg.pinv(I + CRinvC)))) # [ zDim ]

        dif    = (X[idx, :].T - np.mean(X[idx, :], 1)).T            # [ xDim-1 x observations ]
        Xcs[ii,:] = np.mean(X, 1)[ii] + np.dot(np.dot(term2, CRinv), dif) # [ observations ] 
#         Vcs[ii] = C[~idx, :] @ C[~idx, :].T + np.diag(R)[~idx] - term2 @ CRinvC @ C[~idx, :].T
        
    err = np.mean((Xcs.ravel() - X.ravel()) ** 2)  
    return err, Xcs

def FA_EMd(X_cov, xDim, zDim, eps, T, penalty=3, TL_flag=0):
    A = abs(np.random.uniform(0,1, (xDim,zDim))/np.sqrt(zDim))  # initiate A
    R = np.diag(np.diag(X_cov))                                 # initiate R           

    # eps = 1e-7                                                # define stopping value
    LL_prev = 0                                                 # initiate LL reference
    LL_step = eps+1                                             # non-stop value for while loop
    LL_cache = []                                               # array with LL values
    counter = 0 
    while LL_step > eps:                                        # EM FA
        # E-step
        delta = np.linalg.pinv(np.dot(A, A.T) + R)
        beta = np.dot(A.T, delta)

        # M-step
        A = (np.dot(np.dot(X_cov, beta.T),
             np.linalg.pinv(np.eye(zDim) - np.dot(beta, A) + np.dot(np.dot(beta, X_cov), beta.T) ) ))
        R = np.diag(np.diag(X_cov - np.dot(np.dot(A, beta), X_cov)))

        # avg. LL
        if np.linalg.slogdet(delta)[0] > 0:
            LL = -T/2*np.trace(np.dot(delta, X_cov)) + T/2*np.linalg.slogdet(delta)[1] - T*xDim/2*np.log(2*np.pi) 
                                            # N*sum(log(diag(chol(MM))))
        elif np.linalg.slogdet(delta)[0] < 0:
#             print(str(zDim)+'Negative determinant')
            LL = -T/2*np.trace(np.dot(delta, X_cov)) + T/2*np.linalg.slogdet(delta)[1] - T*xDim/2*np.log(2*np.pi) 
                                                                
        LL_step = abs((LL-LL_prev)/abs(LL))
        LL_prev = LL
        LL_cache.append(LL)
        counter += 1
        if counter > 1e4:
            break

    LL_corrected = LL #- zDim ** penalty;

    return A, R, LL_cache, LL_corrected


