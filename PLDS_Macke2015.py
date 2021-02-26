# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 10:13:20 2018

@author: caroline
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.optimize import minimize
from scipy.linalg import solveh_banded

class PLDS:
    
    def __init__(self, xdim, ydim, n_step, C, Q0, A, Q, x0, B, R,
           #estA=None, estC=None, estQ=None, estQ0=None, estx0=None, estB = None, scal = .1,
           Ttrials=1):
        #### model hyperparameters ####
        # number of trials
        self.Ttrials = Ttrials
        # maximum number of time points in a trial
        self.maxn_step = np.max(n_step)
        # number of time points sampled in each trial
        if (len(n_step)==1) & (Ttrials>1):
            self.n_step = np.repeat(n_step, Ttrials)
        else:
            self.n_step = n_step
        # xdim = number of latent dimensions
        # ydim = number of observed dimensions
        if xdim<1:
            print('invalid number of latent dimensions! (<1)')
        self.xdim = xdim
        self.ydim = ydim
        #### ground truth parameters ####
        # C = mapping function
        self.C = C
        # A = temporal transfer function
        self.A = A
        # Q0 = prior noise covariance of latent
        self.Q0 = Q0
        # Q = noise covariance of latent
        self.Q = Q
        # x0 = prior mean of latent
        self.x0 = x0
        # if Gaussian noise, observable noise term
        self.R = R

        # B = stimulus coefficients
        if B is None:
            self.B = None
        else:
            self.B = B


    def sample(self, X=None, seed=None, poisson=True):
        # X = stimulus matrix of the form time points x dimensions x trials
        if X is None:
            # if none -> only baseline firing rate, no stimulus
            X = np.ones([self.maxn_step, 1, self.Ttrials])
            for tt in range(self.Ttrials):
                X[self.n_step[tt]:, :, tt] = np.nan
        if seed is not None:
            np.random.seed(seed)
        # placeholders for latent x and observed y
        self.x = np.zeros([self.maxn_step, self.xdim, self.Ttrials]) * np.nan
        self.y = np.zeros([self.maxn_step, self.ydim, self.Ttrials]) * np.nan
        # cycle through trials (may differ in length!)
        for ttrials in range(self.Ttrials):
            # stimulus drive in log space
            d = (self.B.dot(X[:self.n_step[ttrials],:, ttrials].T)).T
            # initialize the latent
            if self.xdim == 1:
                xold = self.x0 + np.sqrt(self.Q0) * np.random.randn(1)
            else:
                xold = np.random.multivariate_normal(mean=self.x0.T, cov=self.Q0)
            # go through all time points in trial
            for tt in range(self.n_step[ttrials]):
                # update latent
                if tt>0:
                    if self.xdim==1:
                        Ax = self.A*xold
                        xold = Ax + np.sqrt(self.Q)* np.random.randn(1)
                    else:
                        Ax = np.matmul(self.A, xold)
                        xold = np.random.multivariate_normal(Ax, self.Q)
                # project to observed space
                z = self.C.dot(xold) + d[tt,:]
                if poisson:
                    self.y[tt,:, ttrials] = np.random.poisson(np.exp(z))
                else:
                    self.y[tt, :, ttrials] = z + np.random.multivariate_normal(np.zeros(len(z)), self.R)
                # update latent
                self.x[tt, :, ttrials] = xold.copy()

    # visualize
    def vis_xy(self, cho_xdim=0, ttrial=0):
        plt.figure(figsize = (15, 4))
        axxz = plt.subplot2grid((1, 2), (0,0), rowspan = 1, colspan=1)
        axy = plt.subplot2grid((1, 2), (0,1), rowspan = 1, colspan=1)
        if self.xdim==1:
            axxz.plot(self.x[:,0, :], 'k')
            axxz.plot(self.x[:, 0, ttrial], 'r', label='example trial')
        elif self.xdim == 2:
            axxz.plot(self.x[:, 0, :], self.x[:, 1, :], 'k')
            axxz.plot(self.x[:, 0, ttrial], self.x[:, 1, ttrial], 'r', label='example trial')
        elif self.xdim>2:
            axxz.plot(self.x[:, cho_xdim, :], 'k')
            axxz.plot(self.x[:, cho_xdim, ttrial], 'r', label='example trial')
        axxz.set_title('latent trajectories')
        if np.max(self.y)==1:
            spksT = np.arange(self.y[:, :, ttrial].shape[0])
            spks = [spksT[self.y[:, ii, ttrial] > 0] for ii in range(self.y.shape[1])]
            axy.eventplot(spks)
        else:
            axy.plot(self.y[:,:,ttrial], '-k');
        axxz.legend()
        axxz.set_xlabel('time')
        axy.set_title('observed spike count for neurons in trial %.0f' %(ttrial+1))
        axy.set_xlabel('time')
        return axxz, axy

    #######################################################################
    ########################### inference #################################
    #######################################################################

    def log_lik(self, xtmp, ytmp, Btmp, Ctmp, X=None, Rtmp=None, poisson=True):
        # compute negative unnormalized log-likelihood of data
        if X is None: X = np.ones([xtmp.shape[0], 1])
        # log rate is CxT+BXT
        lograte = Ctmp.dot(xtmp.T)+Btmp.dot(X.T)
        # run tests to make sure dimensions all fit
        if (ytmp.shape[1]!=self.ydim)|(xtmp.shape[1]!=self.xdim):
            return 'error in shapes: xtmp and ytmp need to be of T times xdim or ydim'
        if lograte.shape[0] != ytmp.shape[1]:
            return 'error in shapes: lograte.shape[0] != ytmp.shape[1]'
        if poisson:
            L = np.sum((lograte)*(ytmp.T)-self.bounded_exp(lograte))
        else:
            L = np.sum(np.diag(-.5*(ytmp.T-lograte).T.dot(np.linalg.inv(Rtmp)).dot(ytmp.T-lograte)))
        return -L

    def test_log_lik(self, xtmp, Btmp, Ctmp, X=None, Rtmp=None, poisson=True):
        # test that the negative likelihood is lower if there is a noiseless x-data relationship
        # than if there was a DIFFERENT x
        if X is None: X = np.ones([xtmp.shape[0], 1])
        if poisson:
            ytest = np.exp(Ctmp.dot(xtmp.T) + Btmp.dot(X.T)).T
        else:
            ytest = (Ctmp.dot(xtmp.T)).T
        lik_noiseless = self.log_lik(xtmp, ytest, Btmp, Ctmp, X=X, Rtmp=Rtmp, poisson=poisson)
        lik_otherx = self.log_lik(xtmp+.1*np.random.randn(xtmp.shape[0], xtmp.shape[1]),
                                  ytest, Btmp, Ctmp, X=X, Rtmp=Rtmp, poisson=poisson)
        if lik_otherx>lik_noiseless:
            print(bcolors.OKGREEN+'----- TEST PASSED -----')
        else:
            print(bcolors.WARNING+'----- TEST FAILED -----')
            print('noiseless data likelihood: ', lik_noiseless)
            print('noise added to latent, same y: ', lik_otherx)

    def log_prior(self, xtmp, Atmp, Qtmp, Q0tmp, x0tmp):
        # computes the negative log prior of the latent: -logP(x)
        # xtmp is expected to be T by xdim
        mu = self.prior_mu(Atmp, x0tmp, xtmp.shape[0])
        sigma_inv = self.prior_sigma_inv(Atmp, Qtmp, Q0tmp, xtmp.shape[0])
        # diagonal
        L1 = [-(xtmp[tt,:]-mu[tt]).dot(sigma_inv[tt][0]).dot(xtmp[tt,:]-mu[tt]) \
             for tt in range(xtmp.shape[0])]
        # off diagonal
        L2_right = [- (xtmp[tt+1, :] - mu[tt+1]).dot(sigma_inv[tt+1][1]).dot(xtmp[tt, :] - mu[tt]) \
              for tt in range(xtmp.shape[0]-1)]
        L2_left = [- (xtmp[tt -1, :] - mu[tt -1]).dot(sigma_inv[tt][1]).dot(xtmp[tt, :] - mu[tt]) \
                    for tt in range(1,xtmp.shape[0])]
        return -.5*sum(L1+L2_right+L2_left)

    def prior_mu(self, Atmp, x0tmp, T):
        Ax = [x0tmp.copy()]
        [Ax.append(Atmp.dot(Ax[tt])) for tt in range(T-1)]
        return Ax

    def prior_sigma_inv(self, Atmp, Qtmp, Q0tmp, T):
        Qtmpinv = np.linalg.inv(Qtmp)
        sigma_inv = [[np.linalg.inv(Q0tmp) + Atmp.T.dot(Qtmpinv.dot(Atmp)), []]]
        [sigma_inv.append([Atmp.T.dot(Qtmpinv).dot(Atmp)+Qtmpinv, -Qtmpinv.dot(Atmp)])
         for tt in range(1,T-1)]
        sigma_inv.append([Qtmpinv, -Qtmpinv.dot(Atmp)])
        return sigma_inv

    def test_log_prior(self, xtmp, Atmp, Qtmp, Q0tmp, x0tmp):
        # tests that the prior probability of x is less than if it was exactly mu
        prior0 = self.log_prior(xtmp, Atmp, Qtmp, Q0tmp, x0tmp)
        xtest = np.zeros(xtmp.shape)*np.nan
        xtest[0,:] = x0tmp.copy()
        for tt in range(1,xtest.shape[0]):
            xtest[tt,:] = Atmp.dot(xtest[tt-1,:])
        priortest = self.log_prior(xtest, Atmp, Qtmp, Q0tmp, x0tmp)
        if (priortest<prior0)&(prior0>=0)&(np.abs(priortest)<1e-8):
            print(bcolors.OKGREEN+'----- TEST PASSED -----')
        else:
            print(bcolors.WARNING+'----- TEST FAILED -----')

    def log_posterior(self, xtmp, ytmp, Btmp, Ctmp, Atmp, Qtmp, Q0tmp, x0tmp, Rtmp=None,
                      X=None, poisson=True):
        # computes the negative log-posterior through: log P(x|y) ~ log P(y|x) + log P(x)
        # Xtmp is current estimation of x
        # expect Xtmp and Ytmp to be time x dimension  and for one trial
        if np.sum(np.isnan(xtmp[:,0])==False)!=np.sum(np.isnan(ytmp[:,0])==False):
            return 'error in shapes: xtmp and ytmp have differently many valid time points'
        if xtmp.shape[0]!=ytmp.shape[0]:
            return 'error in shapes: xtmp and ytmp have different time & trial dimension'
        # likelihood part
        L_yx = self.log_lik(xtmp, ytmp, Btmp, Ctmp, X, Rtmp, poisson)
        # prior part (same for Gaussian and Poisson)
        L_x = self.log_prior(xtmp, Atmp, Qtmp, Q0tmp, x0tmp)
        return L_yx+L_x

    def J_log_lik(self, xtmp, ytmp, Btmp, Ctmp, Rtmp=None, X=None, poisson=True):
        # compute derivative of negative unnormalized log-likelihood of data
        # OUTPUT: xdim*T vector so that output.reshape(T, xdim)
        if X is None: X = np.ones([xtmp.shape[0], 1])
        # run tests to make sure dimensions all fit
        if (ytmp.shape[1]!=self.ydim)|(xtmp.shape[1]!=self.xdim):
            return 'error in shapes: xtmp and ytmp need to be of T times xdim or ydim'
        if poisson:
            dL = (Ctmp.T).dot(ytmp.T)-Ctmp.T.dot(self.bounded_exp(Ctmp.dot(xtmp.T)+Btmp.dot(X.T)))
        else:
            dL = Ctmp.T.dot(np.linalg.inv(Rtmp)).dot(ytmp.T-(Ctmp.dot(xtmp.T)+Btmp.dot(X.T)))
        return -dL.T.ravel()

    def J_log_prior(self, xtmp, Atmp, Qtmp, Q0tmp, x0tmp):
        # computes the derivative of the negative log prior of the latent: -logP(x)
        # xtmp is expected to be T by xdim
        # OUTPUT: xdim*T vector so that output.reshape(T, xdim)
        mu = self.prior_mu(Atmp, x0tmp, xtmp.shape[0])
        sigma_inv = self.prior_sigma_inv(Atmp, Qtmp, Q0tmp, xtmp.shape[0])
        # diagonal
        Ldiag = [- sigma_inv[tt][0].dot(xtmp[tt, :] - mu[tt]) \
              for tt in range(xtmp.shape[0])]

        # add first off-diagonal
        Loff1 = [- sigma_inv[tt+1][1].dot(xtmp[tt+1, :] - mu[tt+1]) \
              for tt in range(xtmp.shape[0] - 1)]
        Loff2 = [- sigma_inv[tt + 1][1].dot(xtmp[tt, :] - mu[tt]) \
                 for tt in range(xtmp.shape[0] - 1)]
        # add zero array before and after
        Loff1.append(np.zeros(self.xdim))
        Loff2.insert(0, np.zeros(self.xdim))
        Ldiag = np.concatenate(Ldiag)
        Loff1 = np.concatenate(Loff1)
        Loff2 = np.concatenate(Loff2)
        L = [-(Ldiag[tt]+Loff1[tt]+Loff2[tt]) for tt in range(len(Ldiag))]
        return np.array(L)

    def test_J_log_prior(self, xtmp, Atmp, Qtmp, Q0tmp, x0tmp):
        # tests that the derivative of the prior probability of x is less than if it was exactly mu
        # and that it is 0 when x=mu
        prior0 = np.sum(self.J_log_prior(xtmp, Atmp, Qtmp, Q0tmp, x0tmp)**2)
        xtest = np.zeros(xtmp.shape) * np.nan
        xtest[0, :] = x0tmp.copy()
        for tt in range(1, xtest.shape[0]):
            xtest[tt, :] = Atmp.dot(xtest[tt - 1, :])
        priortest = np.sum(self.J_log_prior(xtest, Atmp, Qtmp, Q0tmp, x0tmp)**2)
        if (priortest < prior0) & (prior0 >= 0) & (np.abs(priortest) < 1e-8):
            print(bcolors.OKGREEN + '----- TEST PASSED -----')
        else:
            print(bcolors.WARNING + '----- TEST FAILED -----')

    def test_J_log_lik(self, xtmp, Btmp, Ctmp, Rtmp=None, X=None, poisson=True):
        # tests that the derivative of the likelihood of y if y is exactly what's expected under x
        # gives 0 (indicating an extrema)
        if X is None: X = np.ones([xtmp.shape[0], 1])
        if poisson:
            ytest = np.exp(Ctmp.dot(xtmp.T)+Btmp.dot(X.T)).T
        else:
            ytest = (Ctmp.dot(xtmp.T)+Btmp.dot(X.T)).T
        lik0 = self.J_log_lik(xtmp, ytest, Btmp, Ctmp, Rtmp=Rtmp, X=X, poisson=poisson)
        if np.abs(np.sum(lik0**2)) < 1e-8:
            print(bcolors.OKGREEN + '----- TEST PASSED -----')
        else:
            print(bcolors.WARNING + '----- TEST FAILED -----')
            print(lik0)

    def J_log_posterior(self, xtmp, ytmp, Btmp, Ctmp, Atmp, Qtmp, Q0tmp, x0tmp, Rtmp=None,
                      X=None, poisson=True):
        # computes the derivative of the negative log-posterior through: log P(x|y) ~ log P(y|x) + log P(x)
        # Xtmp is current estimation of x
        # expect Xtmp and Ytmp to be time x dimension x trial
        # likelihood
        d_L_yx = self.J_log_lik(xtmp, ytmp, Btmp, Ctmp, Rtmp=Rtmp, X=X, poisson=poisson)
        # prior
        d_L_x = self.J_log_prior(xtmp, Atmp, Qtmp, Q0tmp, x0tmp)
        return d_L_yx+d_L_x

    def H_log_lik(self, xtmp, Btmp, Ctmp, Rtmp=None, X=None, poisson=True):
        # compute Hessian of negative unnormalized log-likelihood of data
        # OUTPUT: list with xdim by xdim arrays corresponding ot each time point in a trial so that [0] is first time point
        if X is None:
            X = np.ones([xtmp.shape[0], 1])
        d = (Btmp.dot(X.T)).T # size T by ydim
        if poisson:
            HL = [[Ctmp.T.dot(np.diag(self.bounded_exp(Ctmp.dot(xtmp[tt,:].T)+d[tt]))).dot(Ctmp)]
                    for tt in range(xtmp.shape[0])]
        else:
            HL = [[Ctmp.T.dot(np.linalg.inv(Rtmp)).dot(Ctmp)] for tt in range(xtmp.shape[0])]
        return HL

    def H_log_prior(self, xtmp, Atmp, Qtmp, Q0tmp):
        # computes the Hessian of the negative log prior of the latent: -logP(x)
        # xtmp is expected to be T by xdim
        # OUTPUT: list with xdim by xdim arrays corresponding ot each time point in a trial so that [0] is first time point
        return self.prior_sigma_inv(Atmp, Qtmp, Q0tmp, xtmp.shape[0])

    def test_H_log_prior(self, xtmp, Atmp, Qtmp, Q0tmp):
        # tests that the Hessian of the prior probability of x is positive-semidefinite
        # and that it is 0 when x=mu
        Heig = np.linalg.eigvals(self.block_matrix(self.H_log_prior(xtmp, Atmp, Qtmp, Q0tmp), offdiag=1))
        if np.mean(Heig>=0)==1:
            print(bcolors.OKGREEN + '----- TEST PASSED -----')
        else:
            print(bcolors.WARNING + '----- TEST FAILED -----')

    def test_H_log_lik(self, xtmp, Btmp, Ctmp, Rtmp=None, X=None, poisson=True):
        # tests that the Hessian of the likelihood of y|x is positive-semidefinite
        # and that it is 0 when x=mu
        Heig = np.linalg.eigvals(self.block_matrix(self.H_log_lik(xtmp, Btmp, Ctmp, Rtmp=Rtmp, X=X, poisson=poisson), offdiag=0))
        if np.mean(Heig>=0)==1:
            print(bcolors.OKGREEN + '----- TEST PASSED -----')
        else:
            print(bcolors.WARNING + '----- TEST FAILED -----')

    def H_log_posterior(self, xtmp, ytmp, Btmp, Ctmp, Atmp, Qtmp, Q0tmp, x0tmp, Rtmp=None,
                      X=None, poisson=True):
        # computes the Hessian of the negative log-posterior through: log P(x|y) ~ log P(y|x) + log P(x)
        # Xtmp is current estimation of x
        # expect Xtmp and Ytmp to be time x dimension x trial
        # one trial
        H = self.block_matrix(self.H_log_lik(xtmp, Btmp, Ctmp, Rtmp=Rtmp, X=X, poisson=poisson), offdiag=0) + \
            self.block_matrix(self.H_log_prior(xtmp, Atmp, Qtmp, Q0tmp), offdiag=1)

        '''
        tt = 0
        H = self.block_matrix(self.H_log_lik(Xtmp[:, :, tt], Btmp, Ctmp, Rtmp=Rtmp, X=X, poisson=poisson), offdiag=0)+\
            self.block_matrix(self.H_log_prior(Xtmp[:, :, tt], Atmp, Qtmp, Q0tmp), offdiag=1)
        # all other trials
        for tt in range(1, Xtmp.shape[2]):
            H += self.block_matrix(self.H_log_lik(Xtmp[:, :, tt], Btmp, Ctmp, Rtmp=Rtmp, X=X, poisson=poisson), offdiag=0)+\
                    self.block_matrix(self.H_log_prior(Xtmp[:, :, tt], Atmp, Qtmp, Q0tmp), offdiag=1)'''
        return H

    def block_matrix(self, L, offdiag=0):
        # expect that every entry L[0][0], L[1][0], L[2][0] giving the diagonals, have same dimensions!
        if L[0][0].shape!=L[1][0].shape:
            return 'error: expect that every entry L[0][0], L[1][0], L[2][0] giving the diagonals, have same dimensions!'
        mdim = L[0][0].shape
        OUT = np.zeros([mdim[0]*len(L), mdim[1]*len(L)])
        # start with diagonal
        # one matrix after another
        for tt in range(len(L)):
            OUT[tt*mdim[0]:(tt+1)*mdim[0], tt*mdim[1]:(tt+1)*mdim[1]] =L[tt][0]
        # potential offdiagonals
        if offdiag>0:
            for oo in range(1, offdiag+1):
                for tt in range(len(L)-oo):
                    # upper diagonal
                    OUT[tt * mdim[0]:(tt + 1) * mdim[0], (tt+oo) * mdim[1]:(tt + oo + 1) * mdim[1]] = L[tt+1][1]
                    # lower diagonal
                    OUT[(tt + oo) * mdim[0]:(tt + oo + 1) * mdim[0], tt * mdim[1]:(tt + 1) * mdim[1]] = L[tt+1][1]
        return OUT

    def block_matrix_list(self, M, mdim, offdiag=0):
        # M is the block diagonal
        # mdim gives the dimensions of the individual blocks along i and j
        # diagonal
        L = [[M[tt*mdim[0]:(tt+1)*mdim[0], tt*mdim[1]:(tt+1)*mdim[1]]]
             for tt in range(int(M.shape[0]/mdim[0]))]
        # off diagonal
        if offdiag>0:
            [[L[tt+1].append(M[tt * mdim[0]:(tt + 1) * mdim[0], (tt+oo) * mdim[1]:(tt + oo + 1) * mdim[1]])
             for tt in range(int(M.shape[0]/mdim[0])-oo)] for oo in range(1, 1+offdiag)]
        return L

    def test_block(self, M, mdim, offdiag=0):
        # input is matrix
        if np.mean(self.block_matrix(self.block_matrix_list(M, mdim, offdiag),offdiag)==M)==1:
            print(bcolors.OKGREEN + '----- TEST PASSED -----')
        else:
            print(bcolors.WARNING + '----- TEST FAILED -----')


    # wrapper:
    def wrap_x_posterior(self, xtmp0, ytmp, Btmp, Ctmp, Atmp, Qtmp, Q0tmp, x0tmp, Rtmp=None,
               X=None, poisson=True):
        xtmp = xtmp0.reshape(ytmp.shape[0], Atmp.shape[0])
        return self.log_posterior(xtmp, ytmp, Btmp, Ctmp, Atmp, Qtmp, Q0tmp, x0tmp, Rtmp=Rtmp,
                                    X=X, poisson=poisson)
    def wrap_x_Jposterior(self, xtmp0, ytmp, Btmp, Ctmp, Atmp, Qtmp, Q0tmp, x0tmp, Rtmp=None,
               X=None, poisson=True):
        xtmp = xtmp0.reshape(ytmp.shape[0], Atmp.shape[0])
        return self.J_log_posterior(xtmp, ytmp, Btmp, Ctmp, Atmp, Qtmp, Q0tmp, x0tmp, Rtmp=Rtmp,
                                    X=X, poisson=poisson)
    def wrap_x_Hposterior(self, xtmp0, ytmp, Btmp, Ctmp, Atmp, Qtmp, Q0tmp, x0tmp, Rtmp=None,
               X=None, poisson=True):
        xtmp = xtmp0.reshape(ytmp.shape[0], Atmp.shape[0])
        return self.H_log_posterior(xtmp, ytmp, Btmp, Ctmp, Atmp, Qtmp, Q0tmp, x0tmp, Rtmp=Rtmp,
                                    X=X, poisson=poisson)

    def inference(self, Xtmp0, Ytmp, Btmp, Ctmp, Atmp, Qtmp, Q0tmp, x0tmp, Rtmp=None,
               X=None, poisson=True, disp=True): #, xtol=1e-05):
        if X is None:
            X = np.ones([Xtmp0.shape[0], 1, Xtmp0.shape[2]])
        xres = np.zeros(Xtmp0.shape)*np.nan
        for ttrial in range(Xtmp0.shape[2]):
            res = minimize(fun=self.wrap_x_posterior, x0=Xtmp0[:, :, ttrial].ravel(), method='BFGS', #method='Newton-CG',
                           jac=self.wrap_x_Jposterior, #hess=self.wrap_x_Hposterior,
                           options={'disp': disp}, # , 'xtol': xtol
                           args=(Ytmp[:, :, ttrial],
                                 Btmp, Ctmp, Atmp, Qtmp, Q0tmp, x0tmp, Rtmp, X[:,:,ttrial],
                                 poisson))
            xres[:,:,ttrial] = res.x.reshape(Xtmp0.shape[0], Xtmp0.shape[1])
        return xres

    ##################################################
    def bounded_exp(self, x, b=700):
        if np.any(x.ravel() > b):
            x[x > b] = b
            ex = np.exp(x)
            print('numerical error in exponential')
        else:
            ex = np.exp(x)
        return ex

    def bounded_log(self, x, b=1e-10):
        out = x.copy()
        out[out<=0] = b
        return np.log(out)
        
#######################################################################
########################### learning ##################################
#######################################################################

    # create mu(k) and sigma(k)
    def E_step(self, Xtmp0, Ytmp, Btmp, Ctmp, Atmp, Qtmp, Q0tmp, x0tmp, Rtmp=None,
               X=None, poisson=True, disp=True): #, xtol=1e-05):
        # computes the mean and the covariance of the true or approximated gaussian log posterior
        # covariance is the negative Hessian of the log posterior ("H_log_posterior" already computes the negative!!! because it is for the negative log posterior)
        # OUTPUT mu is an array T by xdim by #trials and sigma is a list of length #trials and entries iin block-list form with T blocks xdim by xdim
        if X is None:
            X = np.ones([Xtmp0.shape[0], 1, Xtmp0.shape[2]])
        mu = self.inference(Xtmp0, Ytmp, Btmp, Ctmp, Atmp, Qtmp, Q0tmp, x0tmp, Rtmp=Rtmp,
                            X=X, poisson=poisson, disp=disp) #, xtol=xtol)
        # get the negative Hessian of the posterior
        # then get its inverse
        '''bla = [self.scipy_block(
            self.H_log_posterior(mu[:, :, ttrial], Ytmp[:, :, ttrial], Btmp, Ctmp, Atmp, Qtmp, Q0tmp, x0tmp, Rtmp=Rtmp,
                                 X=X[:, :, ttrial], poisson=poisson))
                 for ttrial in range(Ytmp.shape[2])]'''
        sigma = []
        for ttrial in range(Ytmp.shape[2]):
            try:
                sigma.append(self.block_matrix_list(np.linalg.inv(
                    self.H_log_posterior(mu[:, :, ttrial], Ytmp[:, :, ttrial], Btmp, Ctmp, Atmp, Qtmp, Q0tmp, x0tmp,
                                         Rtmp=Rtmp,
                                         X=X[:, :, ttrial], poisson=poisson)),
                                                    mdim=[Xtmp0.shape[1], Xtmp0.shape[1]], offdiag=1))

                #sigma.append(self.block_matrix_list(solveh_banded(self.scipy_block(self.H_log_posterior(mu[:,:,ttrial], Ytmp[:,:,ttrial], Btmp, Ctmp, Atmp, Qtmp, Q0tmp, x0tmp, Rtmp=Rtmp,
                #                            X=X[:,:,ttrial], poisson=poisson)),
                #                       np.eye(Xtmp0.shape[0]*Xtmp0.shape[1]), lower=True), mdim=[Xtmp0.shape[1], Xtmp0.shape[1]], offdiag=1))
            except:
                print("\n !!!problem with inverting the hessian log posterior on trial %i!!!\n" %ttrial)
                print(self.H_log_posterior(mu[:,:,ttrial], Ytmp[:,:,ttrial], Btmp, Ctmp, Atmp, Qtmp, Q0tmp, x0tmp, Rtmp=Rtmp,
                                            X=X[:,:,ttrial], poisson=poisson))
                break
        '''sigma = [self.block_matrix_list(solveh_banded(self.scipy_block(self.H_log_posterior(mu[:,:,ttrial], Ytmp[:,:,ttrial], Btmp, Ctmp, Atmp, Qtmp, Q0tmp, x0tmp, Rtmp=Rtmp,
                                            X=X[:,:,ttrial], poisson=poisson)),
                                        np.eye(Xtmp0.shape[0]*Xtmp0.shape[1]), lower=True), mdim=[Xtmp0.shape[1], Xtmp0.shape[1]], offdiag=1)
                for ttrial in range(Ytmp.shape[2])]'''
        return mu, sigma

    # helper function for scipy's inversion of blockwise matrices
    def scipy_block(self,S, D=None):
        # https://stackoverflow.com/questions/54609533/matrix-inversion-of-banded-sparse-matrix-using-scipy
        N = np.shape(S)[0]
        if D is None:
            D = N - self.xdim * 2
        ab = np.zeros((D, N))
        for i in np.arange(1, D):
            ab[i, :] = np.concatenate((np.diag(S, k=i), np.zeros(i, )), axis=None)
        ab[0, :] = np.diag(S, k=0)
        return ab

    # update x0
    def upx0(self, mu):
        # expects mu to be an array of T by xdim by # trials
        return np.nanmean(mu[0,:,:], axis=1)

    # update Q0
    def upQ0(self, x0, mu, sigma_11):
        # expects x0 to be an array of dimension xdim
        # expects mu to be an array of T by xdim by # trials
        # expects sigma_ll to be a list of #trials entries with a matrix of xdim by xdim in each entry
        p1 = sum(sigma_11)/len(sigma_11)
        p2 = sum([np.outer(x0-mu[0,:,ttrial],x0-mu[0,:,ttrial]) for ttrial in range(mu.shape[2])])/len(sigma_11)
        return p1+p2

    # helper function M(t,s)
    def M_ts(self, mu, sigma, k, t, s):
        return sigma[k][t][t-s]+np.outer(mu[t,:,k], mu[s,:,k])

    # update A
    def upA(self, mu, sigma):
        # mean_t M(t, t-1)
        a1 = [sum([self.M_ts(mu, sigma, kk, tt, tt-1)
                   for tt in range(1,mu.shape[0])])/(mu.shape[0]-1)
              for kk in range(mu.shape[2])]
        # mean_t M(t-1, t-1)
        a2 = [sum([self.M_ts(mu, sigma, kk, tt-1, tt - 1)
                   for tt in range(1, mu.shape[0])])/(mu.shape[0]-1)
              for kk in range(mu.shape[2])]
        # mean_k mean_t M(t, t-1) [mean_k mean_t M(t-1, t-1)]^-1
        return (sum(a1)/len(a1)).dot(np.linalg.inv(sum(a2)/len(a2)))

    # update Q
    def upQ(self, Atmp, mu, sigma):
        q = [sum([self.M_ts(mu, sigma, kk, tt, tt)+Atmp.dot(self.M_ts(mu, sigma, kk, tt-1, tt - 1).dot(Atmp.T))\
                 -Atmp.dot(self.M_ts(mu, sigma, kk, tt-1, tt))-self.M_ts(mu, sigma, kk, tt, tt-1).dot(Atmp.T)
                 for tt in range(1,mu.shape[1])])/(mu.shape[0]-1)
             for kk in range(mu.shape[2])]
        return sum(q)/len(q)

    # update d=BXT analytically (works if no stimulus, just constant baseline)
    def upd(self, Ctmp, Ytmp, mu, sigma, X=None, Btmp=None, poisson=True):
        if X is None:
            if poisson:
                d1 = np.log(np.nansum(Ytmp, axis=(0,2)))
                d2 = [sum([np.exp(Ctmp.dot(mu[tt,:,kk])+.5*np.diag(Ctmp.dot(sigma[kk][tt][0].dot(Ctmp.T))))
                            for tt in range(mu.shape[0])])
                        for kk in range(mu.shape[2])]
                Bout = d1-np.log(sum(d2))
            else:
                Bout = sum([sum([Ytmp[tt,:,kk]-Ctmp.dot(mu[tt,:,kk])
                                 for tt in range(Ytmp.shape[0])])
                            for kk in range(Ytmp.shape[2])])/(Ytmp.shape[0]*Ytmp.shape[2])
        else:
            print('does not work for more than one stimulus dimension')
            # expect stimulus X to be T by S
            # expect B to be ydim by S
            d1 = sum([Ytmp[:,:,kk].T.dot(X[:,:,kk]) for kk in range(Ytmp.shape[2])])
            d2 = sum([sum([np.exp(Ctmp.dot(mu[tt,:,kk])+.5*np.diag(Ctmp.dot(sigma[kk][tt][0].dot(Ctmp.T)))
                                  ).reshape(Ctmp.shape[0],1).dot(X[tt,:,kk].reshape(1, X.shape[1]))
                           for tt in range(mu.shape[0])])
                      for kk in range(mu.shape[2])])
            # where did the stimulus show up
            ind = np.nansum(X, axis=(0,2))>0
            # update there
            Bout = Btmp.copy()
            Bout[:,ind] = self.bounded_log(d1[:,ind])-self.bounded_log(d2[:,ind])
                          # -self.bounded_log(np.sum(X[:,ind,:],axis=(0,2)))
        return Bout.reshape(Ytmp.shape[1],1)

    # lower bound dependent on latent
    def L_dyn(self, Xtmp, mu, sigma, x0tmp, Q0tmp):
        # compute the NEGATIVE part of the lower bound that depends on the latent
        # prior
        l0 = sum([-.5*np.log(np.linalg.det(Q0tmp))-.5*(Xtmp[0,:,kk]-x0tmp).dot(np.linalg.inv(Q0tmp)).dot(
                                            (Xtmp[0, :, kk] - x0tmp))
                for kk in range(Xtmp.shape[2])])
        ltt = sum([sum([-Xtmp.shape[0]/2*np.log(np.linalg.det(sigma[kk][tt][0]))-
                        .5*(Xtmp[tt,:,kk]-mu[tt,:,kk]).dot(np.linalg.inv(sigma[kk][tt][0])).dot(
                        (Xtmp[tt, :, kk] - mu[tt, :, kk]))
                        for tt in range(Xtmp.shape[0])])
                   for kk in range(Xtmp.shape[2])])
        return -(l0+ltt)

    # lower bound dependent on observed
    def L_obs(self, Ctmp, Btmp, Ytmp, mu, sigma, X=None, Rtmp=None, poisson=True):
        # computes the NEGATIVE part of the lower bound that depends on the observed variable
        if X is None:
            X = np.ones([Ytmp.shape[0], 1, Ytmp.shape[2]])
        if poisson:
            l1 = sum([sum([Ytmp[tt,:,kk].dot(Ctmp.dot(mu[tt,:,kk].T)+Btmp.dot(X[tt,:,kk]))
                            for tt in range(Ytmp.shape[0])])
                    for kk in range(Ytmp.shape[2])])
            l2 = sum([sum([np.exp(Ctmp.dot(mu[tt,:,kk])+Btmp.dot((X[tt,:,kk]))+
                                  .5*np.diag(Ctmp.dot(sigma[kk][tt][0].dot(Ctmp.T))))
                           for tt in range(Ytmp.shape[0])])
                      for kk in range(Ytmp.shape[2])])
            l=(l1 - sum(l2))
        else:
            l = -sum([sum([(Ytmp[tt,:,kk]-Ctmp.dot(mu[tt,:,kk])-Btmp.dot(X[tt,:,kk])).dot(np.linalg.inv(Ctmp.dot(sigma[kk][tt][0].dot(Ctmp.T))+Rtmp)).dot(
                            (Ytmp[tt, :, kk] - Ctmp.dot(mu[tt,:,kk])-Btmp.dot(X[tt,:,kk])).T)
                            for tt in range(Ytmp.shape[0])])
                    for kk in range(Ytmp.shape[2])])
        return -l

    def J_L_obs_C(self, Ctmp, Btmp, Ytmp, mu, sigma, X=None, Rtmp=None, poisson=True):
        # computes the derivative of the NEGATIVE with respect to C
        # part of the lower bound that depends on the observed variable
        if X is None:
            X = np.ones([Ytmp.shape[0], 1, Ytmp.shape[2]])
        if poisson:
            l1 = sum([sum([np.outer(Ytmp[tt, :, kk],mu[tt, :, kk])
                           for tt in range(Ytmp.shape[0])])
                      for kk in range(Ytmp.shape[2])])
            l2 = sum([sum([np.diag(np.exp(Ctmp.dot(mu[tt, :, kk]) + Btmp.dot((X[tt, :, kk])) +
                                  .5 * np.diag(Ctmp.dot(sigma[kk][tt][0].dot(Ctmp.T))))).dot(
                            (np.outer(np.ones(Ytmp.shape[1]), mu[tt,:,kk])+Ctmp.dot(sigma[kk][tt][0])))
                           for tt in range(Ytmp.shape[0])])
                      for kk in range(Ytmp.shape[2])])
            l = l1-l2
        else:
            l1 = sum([sum([np.linalg.inv(Ctmp.dot(sigma[kk][tt][0]).dot(Ctmp.T)+Rtmp).dot(
                        Ytmp[tt,:,kk] - Ctmp.dot(mu[tt, :, kk]) - Btmp.dot(X[tt, :, kk])).reshape(Ytmp.shape[1],1).dot(
                        mu[tt,:,kk].reshape(1,mu.shape[1]))
                        for tt in range(Ytmp.shape[0])])
                    for kk in range(Ytmp.shape[2])])
            l2 = sum([sum([(np.linalg.inv(Ctmp.dot(sigma[kk][tt][0]).dot(Ctmp.T)+Rtmp)).dot(
                np.outer(Ytmp[tt,:,kk]-Ctmp.dot(mu[tt,:,kk])-Btmp.dot(X[tt,:,kk]),
                         Ytmp[tt,:,kk]-Ctmp.dot(mu[tt,:,kk])-Btmp.dot(X[tt,:,kk]))).dot(
                    (np.linalg.inv(Ctmp.dot(sigma[kk][tt][0]).dot(Ctmp.T)+Rtmp)).T).dot(Ctmp.dot(sigma[kk][tt][0]))
                    for tt in range(Ytmp.shape[0])])
                for kk in range(Ytmp.shape[2])])
            l = 2*l1+2*l2
        return -l

    def J_L_obs_B(self, Ctmp, Btmp, Ytmp, mu, sigma, X=None, Rtmp=None, poisson=True):
        # computes the derivative of the NEGATIVE with respect to B
        # part of the lower bound that depends on the observed variable
        if X is None:
            X = np.ones([Ytmp.shape[0], 1, Ytmp.shape[2]])
        if poisson:
            l1 = sum([sum([np.outer(Ytmp[tt, :, kk],X[tt,:,kk])
                           for tt in range(Ytmp.shape[0])])
                      for kk in range(Ytmp.shape[2])])
            l2 = sum([sum([np.outer(np.exp(Ctmp.dot(mu[tt, :, kk]) + Btmp.dot((X[tt, :, kk])) +
                                  .5 * np.diag(Ctmp.dot(sigma[kk][tt][0].dot(Ctmp.T)))),X[tt,:,kk])
                           for tt in range(Ytmp.shape[0])])
                      for kk in range(Ytmp.shape[2])])
            l = l1-l2
        else:
            l = sum([sum([(X[tt,:,kk].reshape(X.shape[1],1).dot(((Ytmp[tt, :, kk] - Ctmp.dot(mu[tt, :, kk]) - Btmp.dot(X[tt, :, kk])).dot(
                np.linalg.inv(Ctmp.dot(sigma[kk][tt][0].dot(Ctmp.T)) + Rtmp))).reshape(1,Ytmp.shape[1]))).T
                for tt in range(Ytmp.shape[0])])
                for kk in range(Ytmp.shape[2])])
        return - l

    def wrap_L_obs_C(self, Ctmp0, Btmp, Ytmp, mu, sigma, X, Rtmp=None, poisson=True):
        Ctmp = Ctmp0.reshape(Ytmp.shape[1], mu.shape[1])
        return self.L_obs(Ctmp, Btmp, Ytmp, mu, sigma, X=X, Rtmp=Rtmp, poisson=poisson)
    def wrap_J_L_obs_C(self, Ctmp0, Btmp, Ytmp, mu, sigma, X, Rtmp=None, poisson=True):
        Ctmp = Ctmp0.reshape(Ytmp.shape[1], mu.shape[1])
        return self.J_L_obs_C(Ctmp, Btmp, Ytmp, mu, sigma, X=X, Rtmp=Rtmp, poisson=poisson).ravel()

    def wrap_L_obs_B(self, Btmp0, Ctmp, Ytmp, mu, sigma, X, Rtmp=None, poisson=True):
        Btmp = Btmp0.reshape(Ytmp.shape[1], X.shape[1])
        return self.L_obs(Ctmp, Btmp, Ytmp, mu, sigma, X=X, Rtmp=Rtmp, poisson=poisson)
    def wrap_J_L_obs_B(self, Btmp0, Ctmp, Ytmp, mu, sigma, X, Rtmp=None, poisson=True):
        Btmp = Btmp0.reshape(Ytmp.shape[1], X.shape[1])
        return self.J_L_obs_B(Ctmp, Btmp, Ytmp, mu, sigma, X=X, Rtmp=Rtmp, poisson=poisson).ravel()

    def upB(self, Btmp0, Ctmp, Ytmp, mu, sigma, X=None, Rtmp=None,
            disp=True, gtol=1e-05, maxiter=100, poisson=True):
        if X is None:
            X = np.ones([Ytmp.shape[0], 1, Ytmp.shape[2]])
        if (X.shape[1] == 1) & (np.mean(X) == 1):
            print('B closed form solution')
            # closed form solution if B is just an offset and X constant
            Bout = self.upd(Ctmp, Ytmp, mu, sigma, X=None, Btmp=Btmp0, poisson=poisson)
        else:
            res = minimize(fun=self.wrap_L_obs_B, x0=Btmp0.ravel(), method='BFGS',
                               jac=self.wrap_J_L_obs_B,
                               options={'disp': disp, 'gtol': gtol, 'maxiter':maxiter},
                               args=(Ctmp, Ytmp, mu, sigma, X, Rtmp, poisson))
            Bout = res.x.reshape(Ytmp.shape[1], X.shape[1])
        return Bout

    def upC(self, Ctmp0, Btmp, Ytmp, mu, sigma, X=None, Rtmp=None, poisson=True,
            disp=True, gtol=1e-05, maxiter=100):
        if X is None:
            X = np.ones([Ytmp.shape[0], 1, Ytmp.shape[2]])
        res = minimize(fun=self.wrap_L_obs_C, x0=Ctmp0.ravel(), method='BFGS',
                       jac=self.wrap_J_L_obs_C,
                       options={'disp': disp, 'gtol': gtol, 'maxiter':maxiter},
                       args=(Btmp, Ytmp, mu, sigma, X, Rtmp, poisson))
        return res.x.reshape(Ytmp.shape[1], mu.shape[1])

#######################################################################
########################### EM ########################################
#######################################################################

class EM:

    def __init__(self, gtol=1e-05, maxiter=10):
        # optimization parameters for numerical fitting for C and B
        self.gtol = gtol
        self.maxiter = maxiter

    def starters(self, xdim, ydim, sdim, seed,
                 C=None, Q0=None, A=None, Q=None, x0=None, B=None, R=None,
                 cscal=2, sigQ = 0.001 , a=.1, sigR=.1):
        # here intelligent ways of choosing starting parameters
        # can be implemented
        np.random.seed(seed)
        if A is None:
            self.A = a * np.eye(xdim)
        else:
            self.A=A
        if Q is None:
            self.Q = np.eye(xdim) * sigQ
        else:
            self.Q=Q
        if Q0 is None:
            self.Q0 = np.eye(xdim) * sigQ
        else:
            self.Q0=Q0
        if x0 is None:
            self.x0 = np.random.randn(xdim)
        else:
            self.x0 = x0
        # observed
        if C is None:
            self.C = cscal * np.random.rand(ydim * xdim).reshape(ydim, xdim)
        else:
            self.C = C
        if R is None:
            self.R = np.eye(ydim) * sigR
        else:
            self.R = R
        # stimulus
        if B is None:
            self.B = cscal * np.random.randn(ydim, sdim)
        else:
            self.B = B

    def backup(self, MOD0):
        MOD_back = PLDS(MOD0.xdim, MOD0.ydim, n_step=MOD0.n_step,
                    C=MOD0.C, Q0=MOD0.Q0, A=MOD0.A, Q=MOD0.Q, x0=MOD0.x0,
                    B=MOD0.B, R=MOD0.R, Ttrials=MOD0.Ttrials)
        MOD_back.x = MOD0.x.copy()
        return MOD_back

    def fit(self, data, xdim, poisson, seed, maxiterem = 10, ltol=1e-1,
            C=None, Q0=None, A=None, Q=None, x0=None, B=None, R=None, S=None,
            cscal=2, sigQ = 0.001 , a=.1, sigR=.1):
        # expect data to be T by ydim by Trials
        if S is None:
            S = np.ones([data.shape[0], 1, data.shape[2]])
        # starting parameters:
        self.starters(xdim, data.shape[1], S.shape[1], seed,
                 C=C, Q0=Q0, A=A, Q=Q, x0=x0, B=B, R=R,
                 cscal=cscal, sigQ = sigQ , a=a, sigR=sigR)
        MOD0 = PLDS(xdim, data.shape[1], n_step=[data.shape[0]],
                    C=self.C, Q0=self.Q0, A=self.A, Q=self.Q, x0=self.x0,
                    B=self.B, R=self.R,Ttrials=data.shape[2])
        MOD0.x = np.random.randn(np.max(MOD0.n_step), MOD0.xdim, MOD0.Ttrials)
        start = time.time()
        for ii in range(maxiterem):
            print('--- iter '+np.str(ii+1)+' ---')
            start_ii = time.time()
            ##### E-step #####
            mu, sigma = MOD0.E_step(MOD0.x, data, MOD0.B, MOD0.C, MOD0.A, MOD0.Q,
                                   MOD0.Q0, MOD0.x0, MOD0.R, X=S,
                                   poisson=poisson, disp=False)
            # update latent
            MOD0.x = mu.copy()
            if ii==0:
                # lower bound before first update (misses the latent part)
                L0 = [[MOD0.L_obs(MOD0.C, MOD0.B, data, mu, sigma, X=S, Rtmp=MOD0.R, poisson=poisson),
                      MOD0.L_dyn(MOD0.x, mu, sigma, MOD0.x0, MOD0.Q0)]]
                print('   lower bound at start ', sum(L0[-1]))
            # make a backup
            MOD_back = self.backup(MOD0)
            ##### M-step #####
            # update parameters for latent
            MOD0.x0 = MOD0.upx0(mu)
            MOD0.Q0 = MOD0.upQ0(MOD0.x0, mu, [sigma[kk][0][0] for kk in range(MOD0.Ttrials)])
            MOD0.A = MOD0.upA(mu, sigma)
            MOD0.Q = MOD0.upQ(MOD0.A, mu, sigma)
            # update parameters for observed
            MOD0.B = MOD0.upB(MOD0.B, MOD0.C, data, mu, sigma, X=S, Rtmp=MOD0.R,
                            disp=False, gtol=self.gtol, maxiter=self.maxiter, poisson=poisson)
            MOD0.C = MOD0.upC(MOD0.C, MOD0.B, data, mu, sigma, X=S, Rtmp=MOD0.R,
                            disp=False, gtol=self.gtol, maxiter=self.maxiter, poisson=poisson)
            # update lower bound
            L0.append([MOD0.L_obs(MOD0.C, MOD0.B, data, mu, sigma, X=S, Rtmp=MOD0.R, poisson=poisson),
                      MOD0.L_dyn(MOD0.x, mu, sigma, MOD0.x0, MOD0.Q0)])
            print('   lower bound after iteration ', ii, ': ', sum(L0[-1]))
            print('     decrease achieved (old neglik - new neglik): ', (sum(L0[ii])-sum(L0[ii+1])))
            if (sum(L0[ii])-sum(L0[ii+1]))<=ltol:
                print('----------\n stopped early at iteration', ii, ': ')
                print('difference in last two lower bounds: ', sum(L0[ii])-sum(L0[ii+1]))
                print('   neg lower bound for observed went from ', L0[ii][0], ' to ', L0[ii+1][0])
                print('   neg lower bound for latent went from ', L0[ii][1], ' to ', L0[ii+1][1])
                break
            else:
                del MOD_back
            end_ii = time.time()
            print('time needed: ', end_ii-start_ii)
            # make sure A does not increase above 1 (otherwise divergence)
            if (MOD0.xdim==1):
                if (MOD0.A>1):
                    print('(correcting A to be <1)')
                    MOD0.A=.99
            elif any(np.linalg.eigvals(MOD0.A) > 1):
                print('(correcting A to be <1)')
                u, s, v = np.linalg.svd(MOD0.A)
                s[s >= 1] = .99
                MOD0.A = np.real(u.dot(np.diag(s)).dot(v))

        end = time.time()
        print('time total needed ', end-start)
        if sum(L0[-2])<sum(L0[-1]):
            print('last iteration did not improve fit, instead returning previous iteration parameters')
            return MOD_back
        else:
            MOD0.x, sigma = MOD0.E_step(MOD0.x, data, MOD0.B, MOD0.C, MOD0.A, MOD0.Q,
                                    MOD0.Q0, MOD0.x0, MOD0.R, X=S,
                                    poisson=poisson, disp=False)
            return MOD0, sigma

    def reconstruction(self, data_test, S_test, MOD0, poisson=True, neurons=None):
        # leave one neuron out, infer the latent from the remaining population
        # then predict the left out neuron's activity
        if S_test is None:
            S_test = np.ones([data_test.shape[0], 1, data_test.shape[2]])
        pred = np.zeros(data_test.shape) * np.nan
        # loop over neurons
        if neurons is None:
            neurons = np.arange(MOD0.ydim)
        for nn in neurons:
            start = time.time()
            print('prediction for neuron '+np.str(nn))
            mask = np.ones(MOD0.ydim, dtype='bool')
            mask[nn] = False
            data_nn = data_test[:, mask, :]
            if poisson:
                Rtmp = None
            else:
                Rtmp = MOD0.R[mask, :].copy()
                Rtmp = Rtmp[:, mask]
            # create model that leaves one neuron out
            MOD_nn = PLDS(MOD0.xdim, MOD0.ydim - 1, n_step=MOD0.n_step,
                          C=MOD0.C[mask, :], Q0=MOD0.Q0, A=MOD0.A, Q=MOD0.Q, x0=MOD0.x0,
                          B=MOD0.B[mask, :], R=Rtmp,
                          Ttrials=data_nn.shape[2])
            # infer latent given remaining population
            MOD_nn.x = np.random.randn(np.max(MOD_nn.n_step), MOD_nn.xdim, MOD_nn.Ttrials)*.1
            mu, sigma = MOD_nn.E_step(MOD_nn.x, data_nn, MOD_nn.B, MOD_nn.C, MOD_nn.A, MOD_nn.Q,
                                      MOD_nn.Q0, MOD_nn.x0, MOD_nn.R, X=S_test,
                                      poisson=poisson, disp=False)
            # predict the left out neuron's activity
            if poisson:
                for tt in range(MOD_nn.Ttrials):
                    pred[:, nn, tt] = np.exp(MOD0.C.dot(mu[:, :, tt].T) + MOD0.B.dot(S_test[:, :, tt].T))[nn, :]
            else:
                for tt in range(MOD_nn.Ttrials):
                    pred[:, nn, tt] = (MOD0.C.dot(mu[:, :, tt].T) + MOD0.B.dot(S_test[:, :, tt].T))[nn, :]
            end = time.time()
            print('------- ', np.round(end-start,3), 'sec -------')
        # compute the mean squared error for every neuron
        mse = np.mean((data_test - pred) ** 2, axis=(0, 2))
        return pred, mse, mu


def print_par(MOD, obs=False):
    print('---- latent var parameters ------')
    print('A: ', np.linalg.eigvals(MOD.A))
    print('Q: ', np.linalg.eigvals(MOD.Q))
    print('---- prior parameters -----------')
    print('x0: ', MOD.x0)
    print('Q0: ', np.linalg.eigvals(MOD.Q0))
    if obs:
        print('---- observed var parameters ----')
        print('C: ', MOD.C)
        print('B: ', MOD.B)
    print(' ')

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[91m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

