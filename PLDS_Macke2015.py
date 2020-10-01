# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 10:13:20 2018

@author: caroline
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as linalg
import math
import pandas as pd
import time
from scipy.optimize import minimize
#from LSM_check_functions import Circle


class PLDS:
    
    def par(self, xdim, ydim, n_step, C=None, Q0=None, A=None, Q=None, x0=None, seed=None, B = None, est = False,
           estA=None, estC=None, estQ=None, estQ0=None, estx0=None, estx=None, estB = None,
           x = 0, y = 0, z = 0, Ttrials=1, scal = .1, X=None):
        self.Ttrials = Ttrials
        self.maxn_step = np.max(n_step)
        if (len(n_step)==1) & (Ttrials>1):
            self.n_step = np.repeat(n_step, Ttrials)
        else:
            self.n_step = n_step
        if X is None:
            R = 1
            X = self.stand_X()
        else:
            R = X.shape[1]
        # xdim = number of latent dimensions
        # ydim = number of observed dimensions
        # n_step = number of samples (in each trial)
        # C = mapping function
        # A = temporal transfer function
        # Q0 = prior noise covariance of latent
        # Q = noise covariance of altent
        # x0 = prior mean of latent
        # seed = fixes randomness (possibly not everywhere - need to check)
        # B = stimulus coefficients
        # u = 
        if seed is not None:
            self.seed = seed
            np.random.seed(seed)
        else:
            self.seed= np.random.choice(100000,1)
        # fixed over trials:
        if xdim<1:
            print('invalid number of latent dimensions! (<1)')
        self.xdim = xdim
        self.ydim = ydim

        self.C = C
        self.Q0 = Q0
        self.A = A
        self.Q = Q
        self.x0 = x0 # this is now the mean of starting distribution of latent (given many trials)
        self.EMiter = np.nan
        self.EMtime = np.nan
        self.EMlik = np.nan
        self.EMconv = np.nan

        
        # variable values
        if np.sum(x)==0:
            self.x = np.zeros([self.maxn_step, xdim, Ttrials])*np.nan
            for ttrial in range(self.Ttrials):
                self.x[:self.n_step[ttrial],:,ttrial] = np.zeros([self.n_step[ttrial], xdim])
        else:
            self.x = x
        if np.sum(y)==0:
            self.y = np.zeros([self.maxn_step, ydim, Ttrials])*np.nan
            for ttrial in range(self.Ttrials):
                self.y[:self.n_step[ttrial],:,ttrial] = np.zeros([self.n_step[ttrial], ydim])
        else:
            self.y = y
        if np.sum(z)==0:
            self.z = np.zeros([self.maxn_step, ydim, Ttrials])*np.nan
            for ttrial in range(self.Ttrials):
                self.z[:self.n_step[ttrial],:,ttrial] = np.zeros([self.n_step[ttrial], ydim])
        else:
            self.z = z

        if B is None:
            self.B = None
            self.d = np.zeros([self.maxn_step, ydim, Ttrials]) * np.nan
            for ttrial in range(self.Ttrials):
                self.d[:self.n_step[ttrial], :, ttrial] = np.zeros([self.n_step[ttrial], ydim])
            #self.B = np.ones([ydim, R])
        else:
            self.B = B
            self.d = np.zeros([self.maxn_step, ydim, Ttrials]) * np.nan
            for ttrial in range(self.Ttrials):
                self.d[:self.n_step[ttrial], :, ttrial] = X[:self.n_step[ttrial],:, ttrial].dot(self.B.T)

        # estimations
        if est:
            if estA is None:
                if xdim==1:
                    self.estA =(np.random.rand(xdim)) * scal
                else:
                    self.estA = np.eye(xdim)*(np.random.rand(xdim))*scal
            else:
                self.estA = estA
            if estC is None:
                self.estC = np.random.randn(ydim*xdim).reshape(ydim, xdim)*scal
            else:
                self.estC = estC
            if estQ is None:
                if xdim==1:
                    self.estQ =(np.random.rand(xdim)) * scal
                else:
                    self.estQ = np.eye(xdim)*np.abs(np.random.randn(xdim))*scal
            else:
                self.estQ = estQ
            if estQ0 is None:
                if xdim==1:
                    self.estQ0 =(np.random.rand(xdim)) * scal
                else:
                    self.estQ0 = np.eye(xdim)*np.abs(np.random.randn(xdim))*scal
            else:
                self.estQ0 = estQ0
            if estx0 is None:
                self.estx0 = np.random.randn(xdim)*scal
            else:
                self.estx0 = estx0
            if estx is None:
                self.estx = np.zeros([self.maxn_step, xdim, Ttrials])*np.nan
                for ttrial in range(self.Ttrials):
                    self.estx[:self.n_step[ttrial],:,ttrial] = np.zeros([self.n_step[ttrial], xdim])
            else:
                self.estx = estx
            if estB is None:
                self.estB = np.ones([ydim, R])
            else:
                self.estB=estB
            self.estd = np.zeros([self.maxn_step, ydim, Ttrials]) * np.nan
            for ttrial in range(self.Ttrials):
                self.estd[:self.n_step[ttrial], :, ttrial] = X[:self.n_step[ttrial], :, ttrial].dot(self.estB.T)

    def stand_X(self):
        X = np.ones([self.maxn_step, 1, self.Ttrials])
        for tt in range(self.Ttrials):
            X[self.n_step[tt]:, :, tt] = np.nan
        return X

    def bounded_exp(self, xtmp, b=700, warn=False, returndiff=False):
        if np.any(xtmp>b):
            xold = np.copy(xtmp)
            xtmp[xtmp>b] = b
            ex = np.exp(xtmp)
            if warn:
                print('x: ', self.estx)
                print('C: ', self.estC)
                print('A: ', self.estA)
                print('d: ', self.estd)
            print('exp explosion')

        else:
            ex = np.exp(xtmp)
            xold = np.copy(xtmp)
        if returndiff:
            ex = list([True, xold - b, ex])
        return ex

    # sample

    def sample(self, X=None, seed=None):
        if X is None:
            X = self.stand_X()
        if seed is None:
            np.random.seed(self.seed)
        else:
            np.random.seed(seed)
        for ttrials in range(self.Ttrials):
            self.x[0,:, ttrials] = np.random.multivariate_normal(self.x0, self.Q0) #np.random.randn(self.xdim).dot(np.linalg.cholesky(self.Q0)) + self.x0
            self.z[0,:, ttrials] = self.C.dot(self.x[0,:, ttrials]) + self.d[0,:, ttrials]
            self.y[0,:, ttrials] = np.random.poisson(np.exp(self.z[0,:, ttrials]))
            #if X is None:
            for tt in range(1, self.n_step[ttrials]):
                if self.xdim==1:
                    Ax = self.A*self.x[tt-1,:, ttrials]
                    self.x[tt, :, ttrials] = Ax + np.sqrt(self.Q)* np.random.randn(1)# Ax + np.random.randn(self.xdim).dot(np.linalg.cholesky(self.Q))

                else:
                    #Ax = self.A.dot(self.x[tt-1,:, ttrials])
                    Ax = np.matmul(self.A, self.x[tt - 1, :, ttrials])
                    self.x[tt, :, ttrials] = np.random.multivariate_normal(Ax,
                                                                           self.Q)  # Ax + np.random.randn(self.xdim).dot(np.linalg.cholesky(self.Q))
                #print(self.A, Ax)

                self.z[tt,:, ttrials] = self.C.dot(self.x[tt,:, ttrials]) + self.d[tt,:, ttrials]
                self.y[tt,:, ttrials] = np.random.poisson(np.exp(self.z[tt,:, ttrials]))
            '''else:
                 for tt in range(1, self.n_step[ttrials]):
                    if self.xdim==1:
                        Ax = self.A*self.x[tt-1,:, ttrials] + X[tt,:, ttrials].dot(self.B.T) # self.B.dot(self.u[tt])
                    else:
                        Ax = self.A.dot(self.x[tt-1,:, ttrials]) +  X[tt,:, ttrials].dot(self.B.T) # self.B.dot(self.u[tt])
                    self.x[tt,:, ttrials] = Ax + np.random.randn(self.xdim).dot(np.linalg.cholesky(self.Q))
                    self.z[tt,:, ttrials] = self.C.dot(self.x[tt,:, ttrials]) + self.d[tt,:, ttrials]
                    self.y[tt,:, ttrials] = np.random.poisson(self.bounded_exp(self.z[tt,:, ttrials]))'''
    # visualize
    def vis_xzy(self, cho_xdim=0, cho_ydim=0):
        plt.figure(figsize = (15, 4))
        axxz = plt.subplot2grid((1, 2), (0,0), rowspan = 1, colspan=1)
        axy = plt.subplot2grid((1, 2), (0,1), rowspan = 1, colspan=1)
        for ttrials in range(self.Ttrials):
            axxz.plot(self.x[:,cho_xdim, ttrials], 'r')
        for ttrials in range(self.Ttrials):
            axxz.plot(self.z[:,cho_ydim, ttrials], '--b')
        axxz.set_title('latent (x, red) dim %.0f and rate (z, blue) of dim %.0f, all trials' %(cho_xdim, cho_ydim))
        axxz.set_xlabel('time')
        for ttrials in range(self.Ttrials):
            axy.plot(self.y[:,cho_ydim, ttrials], 'x')
        axy.set_title('observed spike count of x dim %.0f, all trials' %cho_ydim)
        axy.set_xlabel('time')
        plt.show()
        if self.xdim==2:
            plt.figure(figsize = (10, 4))
            ax1 = plt.subplot2grid((1,2), (0,0))
            ax2 = plt.subplot2grid((1,2), (0,1))
            for ttrials in range(self.Ttrials):
                ax1.plot(self.x[:,0, ttrials], self.x[:,1, ttrials])
            ax1.set_xlabel('first latent dimension')
            ax1.set_ylabel('second latent dimension')
            ax2.plot(self.C)
            ax2.set_xlabel('cells')
            ax2.set_ylabel('latent')
            ax2.set_title('C')
            plt.show()

    #######################################################################
    ########################### inference #################################
    #######################################################################


    # wrapper function for multiplication depending on matrix or scalar
    def mult(self, x, y, xdim, ydim):
        if (xdim==1)|(ydim==1):
            tmp = x*y
        else:
            tmp = x.dot(y)
        return tmp
    
    def expand_C(self, est=False, Ctmp = None, ttrial = 0):
        n_step = self.n_step[ttrial]
        if Ctmp is None:
            if est:
                Ctmp = self.estC
            else:
                Ctmp = self.C
        # Ctilde
        Ctil = np.zeros([n_step*self.ydim, n_step*self.xdim])
        for ii in range(n_step):
            Ctil[(ii*self.ydim):((ii+1)*self.ydim),(ii*self.xdim):((ii+1)*self.xdim)] = Ctmp  
        return Ctil
    
    def expand_d(self, est=False, dtmp=None, ttrial=0, n_step=None):
        if dtmp is None:
            if est:
                dtmp = self.estd[:self.n_step[ttrial], :, ttrial]
            else:
                dtmp = self.d[:self.n_step[ttrial], :, ttrial]
            if n_step is None:
                n_step = self.n_step[ttrial]
        dtil = np.zeros([n_step*self.ydim])
        for ii in range(n_step):
            dtil[(ii*self.ydim):((ii+1)*self.ydim)] = dtmp[ii, :]
        return dtil
    
    
    def expand_S(self, estA=False, estQ=False, estQ0=False, ttrial=0):
        if estA:
            Atmp = self.estA
        else:
            Atmp = self.A
        if estQ:
            Qtmp = self.estQ
        else:
            Qtmp = self.Q
        if estQ0:
            Q0tmp = self.estQ0
        else:
            Q0tmp = self.Q0
        if self.xdim == 1:
            AT = Atmp
            Q0inv = 1/Q0tmp
            Qinv = 1/Qtmp
        else:
            AT =  Atmp.T
            Q0inv = np.linalg.inv(Q0tmp)
            Qinv = np.linalg.inv(Qtmp)
        Sdiag0 = Q0inv + \
            self.mult(self.mult(AT, Qinv, self.xdim, self.xdim), \
                      Atmp, self.xdim, self.xdim)
        Sdiag = self.mult(self.mult(AT, Qinv, self.xdim,self.xdim),Atmp, self.xdim, self.xdim) + Qinv
        Sldiag = -self.mult(Qinv, Atmp, self.xdim, self.xdim)
        Sudiag = -self.mult(AT, Qinv, self.xdim, self.xdim)
        n_step = self.n_step[ttrial]
        S = np.zeros([n_step*self.xdim, n_step*self.xdim])
        for ii in range(n_step-1):
            S[(ii*self.xdim):((ii+1)*self.xdim),(ii*self.xdim):((ii+1)*self.xdim)] = Sdiag
            S[(ii*self.xdim):((ii+1)*self.xdim), ((ii+1)*self.xdim):((ii+2)*self.xdim)] = Sudiag
            S[((ii+1)*self.xdim):((ii+2)*self.xdim), (ii*self.xdim):((ii+1)*self.xdim)] = Sldiag
        S[((n_step-1)*self.xdim):((n_step)*self.xdim),((n_step-1)*self.xdim):\
          (n_step*self.xdim)] = Qinv
        S[0:self.xdim,0:self.xdim] = Sdiag0
        return S

    def expand_mu(self, estA=False, estx0=False):
        if estA:
            Atmp = self.estA
        else:
            Atmp = self.A
        if estx0:
            x0tmp = self.estx0
        else:
            x0tmp = self.x0
        mu = np.zeros(self.maxn_step*self.xdim)
        for ii in range(self.maxn_step):
            if self.xdim>1:
                mu[(ii*self.xdim):((ii+1)*self.xdim)] = np.linalg.matrix_power(Atmp, ii).dot(x0tmp)
            else:
                mu[(ii*self.xdim):((ii+1)*self.xdim)] = (Atmp**ii)*(x0tmp)
        return mu
    
    def expand_xy(self, est=False, ttrial=0, xtmp=None):
        if xtmp is not None:
            xstack = xtmp[:self.n_step[ttrial], :, ttrial].reshape(self.n_step[ttrial] * self.xdim, 1)[:, 0]
        else:
            if est:
                xstack = self.estx[:self.n_step[ttrial], :, ttrial].reshape(self.n_step[ttrial]*self.xdim, 1)[:, 0]
            else:
                xstack = self.x[:self.n_step[ttrial], :, ttrial].reshape(self.n_step[ttrial]*self.xdim, 1)[:, 0]
        ystack = self.y[:self.n_step[ttrial], :, ttrial].reshape(self.n_step[ttrial]*self.ydim, 1)[:, 0]
        return xstack, ystack
    
    # fitting
    def expand(self, est = False, ttrial=0):
        Ctil = self.expand_C(est, ttrial=ttrial)
        dtil = self.expand_d(est, ttrial=ttrial)
        S = self.expand_S(estA=est, estQ=est, estQ0=est, ttrial=ttrial)
        mu = self.expand_mu(estA=est, estx0=est)
        ### x and y
        xstack, ystack = self.expand_xy(est, ttrial=ttrial) 
        return Ctil, S, mu, xstack, ystack, dtil

    def Lik(self, xtmp, ytmp, S, mu, Ctil, dtil, ttrial=0):
        # print('compute lik')
        n_step = self.n_step[ttrial]
        e = np.ones(self.ydim*n_step)
        expCx = self.bounded_exp(Ctil.dot(xtmp)+dtil, returndiff=True)
        if expCx[0]:
            expCx = np.copy(expCx[2]) # here something needs to be done!!!!!!!!
        else:
            expCx = np.copy(expCx[2])
        L = np.dot(ytmp.T, Ctil.dot(xtmp)+dtil)-np.dot(e.T, expCx)-0.5*np.dot((xtmp-mu[:(self.xdim*n_step)]).T, S.dot(xtmp-mu[:(self.xdim*n_step)]))
        return L

    def dLik(self, xtmp, ytmp, S, mu, Ctil, dtil, ttrial=0):
        
        n_step = self.n_step[ttrial]
        dL = np.dot(Ctil.T, ytmp)-\
            np.dot(Ctil.T, self.bounded_exp(np.dot(Ctil, xtmp)+dtil))-\
            np.dot(S, xtmp-mu[:(self.xdim*n_step)])
        return dL

    def HLik(self, xtmp, S, Ctil, dtil, ttrial=0): # this is also the covariance matrix of the latent
        n_step = self.n_step[ttrial]
        HL = -np.dot(Ctil.T,np.eye(n_step*self.ydim)*(self.bounded_exp(np.dot(Ctil, xtmp)+dtil))).dot(Ctil) - S
        return HL

    def bandH(self, HL, ttrial=0):
        n_step = self.n_step[ttrial]
        bandedH = np.zeros([4*self.xdim-1, n_step*self.xdim])
        ind = np.arange(-(2*self.xdim-1), (2*self.xdim))
        for ii in ind:
            ind = np.arange(np.max([-ii,0]), np.min([n_step*self.xdim, n_step*self.xdim-ii]))
            bandedH[2*self.xdim-1+ii,ind] = np.diag(HL, k=-ii)
        return bandedH
 
    def invHLik(self, xtmp, S, Ctil, dtil, ttrial=0): # this is the covariance of the log posterior: logP(x|y)
        n_step = self.n_step[ttrial]
        bandedH = self.bandH(self.HLik(xtmp, S, Ctil, dtil, ttrial=ttrial), ttrial=ttrial)
        iHL = linalg.solve_banded((2*self.xdim-1, 2*self.xdim-1), bandedH, np.eye(n_step*self.xdim) )
        return iHL # np.linalg.inv(self.HLik(xtmp, S, Ctil, dtil))

    def invnegHLik(self, xtmp, S, Ctil, dtil, ttrial=0): # this is the covariance of the log posterior: logP(x|y)
        n_step = self.n_step[ttrial]
        bandedH = self.bandH(-self.HLik(xtmp, S, Ctil, dtil, ttrial=ttrial), ttrial=ttrial)
        iHL = linalg.solve_banded((2*self.xdim-1, 2*self.xdim-1), bandedH, np.eye(n_step*self.xdim) )
        return iHL # np.linalg.inv(self.HLik(xtmp, S, Ctil, dtil))
    
    '''
    def invHLik(self, xtmp, S, Ctil, dtil):
        return np.linalg.inv(self.HLik(xtmp, S, Ctil, dtil))'''

    
    def xNewton(self, Ctil, S, mu, ystack, dtil, imax = 1000, maxtim = 60, ttrial=0):
        n_step = self.n_step[ttrial]

        xest = mu[:(self.xdim*n_step)]
        ii = 0
        conv = False
        #X0 = np.zeros([imax+1, n_step*self.xdim])+np.nan
        #X0[0,:] = xest
        L = np.zeros(imax+1)*np.nan
        start = time.time()
        while (conv==False) & (ii<imax):
            h = self.invHLik(xest, S, Ctil, dtil, ttrial=ttrial)
            xch = h.dot(self.dLik(xest, ystack, S, mu, Ctil, dtil, ttrial=ttrial))
            xest = xest-xch
            y0 = np.sum(np.abs(self.dLik(xest, ystack, S, mu, Ctil, dtil, ttrial=ttrial)))
            ii = ii+1
            #X0[ii,:] = xest
            L[ii] = self.Lik(xest, ystack, S, mu, Ctil, dtil, ttrial=ttrial)
            if y0<1e-6:
                conv = True
            end = time.time()
            if (end-start)>maxtim: 
                print('Newton optimisation of latent: max time')
                break
        Xfin = xest.reshape(n_step, self.xdim) #X0[ii,:].reshape(n_step, self.xdim)
        X0 = None
        return Xfin, X0, ii, y0, Ctil, S, conv, L

    # wrapper functions for scipy implementation

    def wrap_Lik(self, xtmp, ytmp, S, mu, Ctil, dtil, ttrial=0):
        return -self.Lik(xtmp, ytmp, S, mu, Ctil, dtil, ttrial=ttrial)

    def wrap_HLik(self, xtmp, ytmp, S, mu, Ctil, dtil, ttrial=0):
        return -self.HLik(xtmp, S, Ctil, dtil, ttrial=ttrial)

    def wrap_dLik(self, xtmp, ytmp, S, mu, Ctil, dtil, ttrial=0):
        return -self.dLik(xtmp, ytmp, S, mu, Ctil, dtil, ttrial=ttrial)



    def MA(self, xsig, win, scalsig = -1):
        # I think this function was just for the combination with the Deneve model to rescale the latent signal to be encoded by the population of neurons
        ma_z = np.copy(xsig)
        for ii in range(win, len(ma_z)-win):
                ma_z[ii] = np.mean(ma_z[ii-win:ii+win])
        ma_z[0:win] = np.mean(ma_z[0:win])
        ma_z[len(ma_z)-win:len(ma_z)] = np.mean(ma_z[len(ma_z)-win:len(ma_z)])
        if scalsig>=0:
            ma_z = ma_z/np.max(np.abs(ma_z)) * scalsig
        else:
            scalsig = np.max(np.abs(ma_z))    
        return ma_z, scalsig


    def fit_cor(self, smoothed_state_means, Xfin, xtmp):
        CC = np.zeros([self.xdim, 2])
        for ii in range(self.xdim):
            CC[ii,:] = [np.corrcoef(xtmp[:,ii], smoothed_state_means[:,ii])[0,1],\
                          np.corrcoef(xtmp[:,ii], Xfin[:,ii])[0,1]]
        CCalt = np.zeros([self.xdim, 2])*np.nan
        cc_klalt =0
        cc_pldsalt=0
        if self.xdim>1:
            for ii in range(self.xdim):
                CCalt[ii,:] = [np.corrcoef(xtmp[:,1-ii], smoothed_state_means[:,ii])[0,1],\
                              np.corrcoef(xtmp[:,1-ii], Xfin[:,ii])[0,1]]
            cc_klalt = np.mean([np.abs(CCalt[:,0])])
            cc_pldsalt = np.mean([np.abs(CCalt[:,1])])
        cc_kl = np.mean([np.abs(CC[:,0])])
        cc_plds = np.mean([np.abs(CC[:,1])])
        return np.max([cc_kl, cc_klalt]), np.max([cc_plds, cc_pldsalt]), CC, CCalt
    
    def fit_cor2(self, smoothed_state_means, Xfin, xtmp):
        CC = np.zeros([self.xdim, 2])*np.nan
        indCC = np.zeros([self.xdim, 2])*np.nan
        # for ii in range(self.xdim):
        ii = 0 # test the first dimension, second dimension must then be the other...
        CCtmp = np.zeros([self.xdim, 2])*np.nan
        for jj in range(self.xdim):
            CCtmp[jj, 0] = np.corrcoef(xtmp[:,jj], smoothed_state_means[:,ii])[0,1]
            CCtmp[jj, 1] = np.corrcoef(xtmp[:,jj], Xfin[:,ii])[0,1]
        for ttype in range(2):    
            CC[ii,ttype] = np.max(np.abs(CCtmp[:,ttype]))
            indCC[ii,ttype] = np.argmax(np.abs(CCtmp[:,ttype]))
        # correlation with the other dimension:            
        CC[ii+1,ttype] = np.corrcoef(xtmp[:,np.int(np.array([1-indCC[ii,0]]))], smoothed_state_means[:,ii+1])[0,1]
        CC[ii+1,ttype] = np.corrcoef(xtmp[:,np.int(np.array([1-indCC[ii,1]]))], Xfin[:,ii+1])[0,1]    

        cc_kl = np.mean([np.abs(CC[:,0])])
        cc_plds = np.mean([np.abs(CC[:,1])])
        return cc_kl, cc_plds, CC, indCC

    def visfitting(self, smoothed_state_means, Xfin, scal_kal=1, scal_plds=1, off = False,\
                  C_KF=None, C_PLDS=None, C_FA=None, ttrial=0, otherX=None, otherXname='other', scal_otherX=1,
                    otherX2=None, otherXname2='other', scal_otherX2=1):
        xtmp = self.x[:,:,ttrial]
        # compute  correlation
        cc_kl, cc_plds, CC, CCalt = self.fit_cor(smoothed_state_means, Xfin, xtmp)
        print('KF correlation %.3f' %cc_kl)
        print('PLDS correlation %.3f' %cc_plds)
        scal_kal = np.repeat(scal_kal, self.xdim) * np.sign(CC[:,0])
        scal_plds = np.repeat(scal_plds, self.xdim) * np.sign(CC[:,1])
        if otherX is not None:
            _, _, CC, _ = self.fit_cor(smoothed_state_means, otherX, xtmp)
            scal_otherX = np.repeat(scal_otherX, self.xdim) * np.sign(CC[:,1])
        if otherX2 is not None:
            _, _, CC, _ = self.fit_cor(smoothed_state_means, otherX2, xtmp)
            scal_otherX2 = np.repeat(scal_otherX2, self.xdim) * np.sign(CC[:,1])
        if self.xdim==1:
            # plot latent
            if off:
                off_kal = (xtmp[0,:]-smoothed_state_means[0,:]*scal_kal)
                off_plds = (xtmp[0,:]-Xfin[0,:]*scal_plds)
            else:
                off_kal = 0
                off_plds = 0
            
            plt.figure(figsize=(20,4))
            ax1 = plt.subplot2grid((1,4), (0,0))
            ax2 = plt.subplot2grid((1,4), (0,1))
            ax3 = plt.subplot2grid((1,4), (0,2))
            ax4 = plt.subplot2grid((1,4), (0,3))
            if C_PLDS is None:
                ax1.plot(self.estC, '--b', label='PLDS')
                ax1.plot(self.C, 'r', label='real')
            else:
                ax1.plot(C_KF, '--k', label='KF')
                ax1.plot(C_PLDS, '--b', label='PLDS')
                ax1.plot(self.C, 'r', label='real')
            ax1.legend()
            if C_FA is not None:
                ax1.plot(C_FA.T, '--g', label='FA')
            ax1.set_title('C')
            ax2.hist(xtmp, color='r', alpha=0.5, label='real')
            ax2.hist(Xfin, color='b', alpha=0.5, label='PLDS')
            ax2.hist(smoothed_state_means, color='k', alpha=0.5, label='KF')
            ax2.legend()
            ax2.set_title('distribution latent')
            ax3.plot(xtmp[:,0],'r-', label='real')
            ax3.plot(smoothed_state_means[:,0]*scal_kal+off_kal,"k--", label='KF')
            ax3.plot(Xfin[:,0]*scal_plds+off_plds,"b--", label='PLDS')
            ax3.legend()
            ax3.set_xlabel('time')
            ax3.set_ylabel('latent variable (rot*scal)')
            ax3.set_title('latent trajectory')
            
            ax4.plot(xtmp, Xfin*scal_plds/np.max(np.abs(Xfin*scal_plds)), '.b', label='PLDS')
            ax4.plot(xtmp, smoothed_state_means*scal_kal/np.max(np.abs(smoothed_state_means*scal_kal)), '.k', label='KF')
            ax4.legend()
            ax4.set_xlabel('real latent')
            ax4.set_ylabel('estimated latent (norm)')
            ax4.set_title('correlation')
            
            plt.show()
        else:
            fig, ax = plt.subplots(1, self.xdim, figsize = (20, 4))
            # plot latent dimension 1 &2
            if off:
                off_kal = (xtmp[0,:]-smoothed_state_means[0,:]*scal_kal[0])
                off_plds = (xtmp[0,:]-Xfin[0,:]*scal_plds[0])
            else:
                off_kal = np.array([0, 0])
                off_plds = np.array([0,0])
            for xx in range(self.xdim):
                ax[xx].plot(xtmp[:,xx],'r-', label='real')
                ax[xx].set_xlabel('time')
                ax[xx].set_ylabel('dim '+np.str(xx+1))
            for xx in range(smoothed_state_means.shape[1]):
                ax[xx].plot(smoothed_state_means[:,xx]*scal_kal[xx]+off_kal[xx],"k--", label='KF')
            for xx in range(Xfin.shape[1]):
                ax[xx].plot(Xfin[:,xx]*scal_plds[xx]+off_plds[xx], "b", label='PLDS')
            if otherX is not None:
                for xx in range(otherX.shape[1]):
                    ax[xx].plot(otherX[:,xx]*scal_otherX[xx], "g", label=otherXname)
            if otherX2 is not None:
                for xx in range(otherX2.shape[1]):
                    ax[xx].plot(otherX2[:,xx]*scal_otherX2[xx], "y", label=otherXname2)
            ax[0].legend()

            fig, ax = plt.subplots(1, 3, figsize=(20, 4))
            # plot correlation: KM and real
            for xx in range(smoothed_state_means.shape[1]):
                ax[0].plot(xtmp[:,xx], smoothed_state_means[:,xx], '.', color=plt.cm.coolwarm(xx/smoothed_state_means.shape[1]), label='dim '+np.str(xx))
            ax[0].set_xlabel('time')
            ax[0].set_title('decode output signal from spike trains')
            ax[0].legend()

            # plot correlation: PLDS and real
            for xx in range(Xfin.shape[1]):
                ax[1].plot(xtmp[:,xx], Xfin[:,xx], '.', color=plt.cm.coolwarm(xx/Xfin.shape[1]), label='dim '+np.str(xx))
            ax[1].legend()
            ax[0].set_ylabel('KF estimated variable')
            ax[0].set_xlabel('real latent variable')
            ax[1].set_ylabel('PLDS estimated variable')
            ax[1].set_xlabel('real latent variable')
            # # plot correlation: KM and PLDS
            ax[2].plot(Xfin, smoothed_state_means, '.g')
            ax[2].set_ylabel('KF estimated variable')
            ax[2].set_xlabel('PLDS estimated variable')
            if otherX is not None:
                fig, ax = plt.subplots(1, 3, figsize=(20, 4))
                # plot correlation: KM and real
                for xx in range(otherX.shape[1]):
                    ax[0].plot(xtmp[:, xx], otherX[:, xx], '.',
                               color=plt.cm.coolwarm(xx / otherX.shape[1]), label='dim ' + np.str(xx))
                ax[0].set_xlabel('time')
                ax[0].set_title('decode output signal from spike trains')
                ax[0].legend()

                # plot correlation: PLDS and real
                for xx in range(otherX2.shape[1]):
                    ax[1].plot(xtmp[:, xx], otherX2[:, xx], '.', color=plt.cm.coolwarm(xx / otherX2.shape[1]),
                               label='dim ' + np.str(xx))
                ax[1].legend()
                ax[0].set_ylabel(otherXname+' estimated variable')
                ax[0].set_xlabel('real latent variable')
                ax[1].set_ylabel(otherXname2+' estimated variable')
                ax[1].set_xlabel('real latent variable')
                # # plot correlation: KM and PLDS
                ax[2].plot(otherX, otherX2, '.g')
                ax[2].set_ylabel(otherXname+' estimated variable')
                ax[2].set_xlabel(otherXname2+' estimated variable')

        return cc_kl, cc_plds

            
        
#######################################################################
########################### learning ##################################
#######################################################################


    # update A
    def Mts(self, SIGest, tt, ss, ttrial=0):
        # tt indicates one time point
        # ss indicates another time point
        if self.xdim>1:
            tmp = SIGest[(tt*self.xdim):((tt+1)*self.xdim), (ss*self.xdim):((ss+1)*self.xdim)] + np.outer(self.estx[tt,:, ttrial],(self.estx[ss,:, ttrial]))
        else:
            tmp = SIGest[(tt*self.xdim):((tt+1)*self.xdim), (ss*self.xdim):((ss+1)*self.xdim)][0] + self.estx[tt,0, ttrial]*(self.estx[ss,0, ttrial])
        return tmp

    def upA(self, SIGest, ttrial = 0):
        Aest1 = 0
        Aest2 = 0
        n_step = self.n_step[ttrial]
        for tt in range(1,n_step):
            Aest1 = Aest1 + self.Mts(SIGest, tt, tt-1) 
            Aest2 = Aest2 + self.Mts(SIGest, tt-1, tt-1) 
        Aest = Aest1.dot(np.linalg.inv(Aest2))
        return Aest

    # update Q
    def upQ(self, SIGest):
        Qest = 0
        for ttrial in range(self.Ttrials):
            for tt in range(1, self.n_step[ttrial]):
                if self.xdim>1:
                    Qest += (self.Mts(SIGest[:,:,ttrial], tt, tt, ttrial)\
                        + self.estA.dot(self.Mts(SIGest[:,:,ttrial], tt-1, tt-1, ttrial)).dot(self.estA.T)\
                        - self.estA.dot(self.Mts(SIGest[:,:,ttrial], tt-1, tt, ttrial))\
                        - (self.Mts(SIGest[:,:,ttrial], tt, tt-1, ttrial)).dot(self.estA.T))#/(self.n_step[ttrial]-1)
                else:
                    Qest += (self.Mts(SIGest[:,:,ttrial], tt, tt, ttrial)\
                        + self.estA*self.Mts(SIGest[:,:,ttrial], tt-1, tt-1, ttrial)*self.estA.T\
                        - self.estA*self.Mts(SIGest[:,:,ttrial], tt-1, tt, ttrial)\
                        - self.Mts(SIGest[:,:,ttrial], tt, tt-1, ttrial)*self.estA.T)#/(self.n_step[ttrial]-1)
            #Qest = np.divide(Qest,(np.sum(self.n_step)-self.Ttrials))
            Qest = np.divide(Qest, (self.Ttrials))
            # changed here to fit the different number of presentation per trial
        return Qest


    def dC(self, xtmp, Ctmp, SIGest, dtmp = None): # gradient for C
        if dtmp is None:
            dtmp = self.d
        eq = np.ones(self.ydim)
        dc = np.zeros([self.ydim, self.xdim])
        for ttrial in range(self.Ttrials):  
            for tt in range(self.n_step[ttrial]):
                CS = Ctmp.dot(SIGest[(tt*self.xdim):((tt+1)*self.xdim), (tt*self.xdim):((tt+1)*self.xdim), ttrial])

                yest = self.bounded_exp(Ctmp.dot(xtmp[tt,:, ttrial])+dtmp[tt,:, ttrial]+0.5*np.diag(CS.dot(Ctmp.T)))

                dc= dc+(np.outer(self.y[tt,:, ttrial], xtmp[tt,:, ttrial])-np.diag(yest).dot(np.outer(eq,xtmp[tt,:, ttrial])+CS))
    
        return dc

    def LC(self, xtmp, Ctmp, SIGest, dtmp = None):
        # likelihood for the observational-parameter part of the lower bound to P(y|parameters)
        eq = np.ones(self.ydim)
        lik = 0
        if dtmp is None:
            dtmp = self.d
        for ttrial in range(self.Ttrials):  
            ctmp = np.copy(Ctmp)# [:self.n_step[ttrial],:self.n_step[ttrial]]
            for tt in range(self.n_step[ttrial]):
                CS = ctmp.dot(SIGest[(tt*self.xdim):((tt+1)*self.xdim), (tt*self.xdim):((tt+1)*self.xdim), ttrial])
                yest = self.bounded_exp(ctmp.dot(xtmp[tt,:, ttrial])+dtmp[tt,:, ttrial]+0.5*np.diag(CS.dot(ctmp.T)))
                lik += self.y[tt,:, ttrial].T.dot(ctmp.dot(xtmp[tt,:, ttrial])+dtmp[tt,:, ttrial])-eq.T.dot(yest)
        return lik



    def GD_C(self, xtmp, SIGest, alpha0=1, iimax = 100000, lr = 0.01, lim=0.00001, maxtim=60, estCo = None,\
             xstack=None, Sest=None, dtil=None):
        # gradient descent
        ESTC = np.zeros([self.ydim, self.xdim, iimax+1]) 
        if estCo is None:
            estCo = np.random.randn(self.ydim*self.xdim).reshape(self.ydim, self.xdim)
        ESTC[:,:,0] = estCo
        ii = 0
        conv = False
        start = time.time()
        while (ii<iimax) & (conv==False):
            dc = self.dC(xtmp, estCo, SIGest, dtmp = self.estd)
            alpha = alpha0 # reset alpha
            estC = estCo + alpha * lr*dc
            ii += 1
            # possibly improve this criteria (see code for alpha-beta gradient descent or something like that)
            while (self.LC(xtmp, estCo, SIGest)
                   >self.LC(xtmp, estC, SIGest))\
                    & (alpha>lim):
                alpha = alpha/10
                estC = estCo + alpha*lr*dc
                if np.sum(estC==np.nan)>0:
                    break
            if np.sum(estC==np.nan)>0:
                break
            if alpha<=lim:
                conv=True
                ii = ii-1
            else:
                estCo = estC
                ESTC[:,:,ii] = estC
            end = time.time()
            if (end-start)>maxtim: 
                print('gradient descent C: max time')
                break
        if ii==iimax: print('max iterations reached')
        return ESTC[:,:,ii], ESTC, ii, alpha


    # alternatively use scipy implementation for optimization:
    def wrap_dc(self, Ctmp, xtmp, SIGest, dtmp=None):
        Ctmp = Ctmp.reshape(self.ydim, self.xdim)
        der = -self.dC(xtmp, Ctmp, SIGest, dtmp)
        return der.reshape(self.xdim * self.ydim)

    def wrap_lc(self, Ctmp, xtmp, SIGest, dtmp=None):
        Ctmp = Ctmp.reshape(self.ydim, self.xdim)
        lik = -self.LC(xtmp, Ctmp, SIGest, dtmp)
        return lik

    def wrap_dLB(self, Btmp, R, xtmp, Ctmp, SIGest, X):
        Btmp = Btmp.reshape(self.ydim, R)
        dlik = -self.dB(R, xtmp, Ctmp, SIGest, X, Btmp)
        return dlik.reshape(R * self.ydim)

    def wrap_LB(self, Btmp, R, xtmp, Ctmp, SIGest, X):
        Btmp = Btmp.reshape(self.ydim, R)
        dtmp = np.zeros(self.estd.shape) * np.nan
        for tt in range(self.Ttrials):
            dtmp[:self.n_step[tt], :, tt] = X[:self.n_step[tt], :, tt].dot(Btmp.T)
        lik = -self.LC(xtmp, Ctmp, SIGest, dtmp)
        return lik

    # if R=1 and d is just an offset --> closed form solution

    def upd_closed(self, SIGest, X):
        dopt = (np.sum(np.sum(self.y, axis=0), axis=1))
        dopt[dopt==0] = .00000001
        dopt = np.log(dopt)
        tmpsum = np.zeros([self.ydim])
        for ttrial in range(self.Ttrials):
            for tt in range(self.n_step[ttrial]):
                CS = self.estC.dot(
                    SIGest[(tt * self.xdim):((tt + 1) * self.xdim), (tt * self.xdim):((tt + 1) * self.xdim), ttrial])
                tmpsum += np.exp(self.estC.dot(self.estx[tt, :, ttrial]) + .5 * np.diag(CS.dot(self.estC.T)))
        # print(self.estB[:,0]-dopt + np.log(tmpsum))
        self.estB[:,0] = dopt - np.log(tmpsum)
        self.estd = self.update_d(X, estB=self.estB)

    # if stimulus and d depends on those - need GD
        
    def dB(self, R, xtmp, Ctmp, SIGest, X, Btmp):
        eq = np.ones(self.ydim)
        db = np.zeros([self.ydim, R])# np.zeros([self.ydim, self.xdim])np.zeros([self.ydim, self.xdim, self.n_step])

        for ttrial in range(self.Ttrials):
            for tt in range(self.n_step[ttrial]):
                CS = Ctmp.dot(SIGest[(tt*self.xdim):((tt+1)*self.xdim), (tt*self.xdim):((tt+1)*self.xdim),ttrial])
                yest = self.bounded_exp(Ctmp.dot(xtmp[tt,:,ttrial])+Btmp.dot(X[tt,:,ttrial].T)+0.5*np.diag(CS.dot(Ctmp.T)))
                db = db + (np.outer(self.y[tt,:,ttrial], X[tt,:,ttrial])-np.diag(yest).dot(np.outer(eq,X[tt,:,ttrial])))
        return db

    def update_d(self, X, estB):
        dnew = np.zeros([self.maxn_step, self.ydim, self.Ttrials])*np.nan
        for ttrial in range(self.Ttrials):
            dnew[:self.n_step[ttrial],:,ttrial] = X[:self.n_step[ttrial],:,ttrial].dot(estB.T)
        return dnew


    def GD_B(self, xtmp, R, X, Ctmp, SIGest, \
             estBo = None, alpha0=1, iimax = 100000, lr = 0.01, lim=0.00001, maxtim=100000, printit=False):
        # xstack, Sest, Ctilest, \
        
        if printit: print('GD-B: start')

        ESTB = np.zeros([self.ydim, R, iimax+1]) 
        if estBo is None:
            estBo = np.random.randn(self.ydim*R).reshape(self.ydim, R)
            self.estd= self.update_d(X, estBo)       
        
        ESTB[:,:,0] = estBo
        ii = 0
        conv = False
        start = time.time()
        while (ii<iimax) & (conv==False):
            # compute derivative
            db = self.dB(R, xtmp, Ctmp, SIGest, X, estBo)
            # reset alpha
            alpha = alpha0
            # update B
            estB = estBo + alpha * lr*db
            ii += 1
            # update d
            dnew = self.update_d(X, estB)

            # if step is too big, reduce stepsize
            while (self.LC(xtmp, Ctmp, SIGest, dtmp=self.estd)\
                   >self.LC(xtmp, Ctmp, SIGest, dtmp=dnew)) & (alpha>lim):
                alpha = alpha/10
                # update B
                estB = estBo + alpha*lr*db
                # update d
                dnew = self.update_d(X, estB) # dnew = X.dot(estB.T) # (estB.dot(X.T)).T
    
    
                if np.sum(estB==np.nan)>0:
                    break
            # what happened in while loop:
            # 1. error: possibly explosion in value
            if np.sum(estB==np.nan)>0:
                break
            # 2. alpha could not get small enough --> converged to optimum
            if alpha<=lim:
                conv=True
                ii = ii-1
            else:
                # update B
                estBo = estB
                ESTB[:,:,ii] = estB
                # update d
                dnew = self.update_d(X, estB) # dnew = X.dot(estB.T) #     (estB.dot(X.T)).T
                self.estd = dnew
            end = time.time()
            # 3. reached maxtime --> stop, waited enough
            if (end-start)>maxtim: 
                print('gradient descent B: max time')
                print('iteration ', ii)
                break
        # 4. maximum number of iterations reached
        if ii==iimax: print('max iterations reached')
        if printit: print('GD-B: end')
        return ESTB[:,:,ii], ESTB, ii, alpha # , SIGest, dtil
        
    
    
    # time tracking
    def timtrack(self, tracktime, var, iterii, start, message, maxtim):
        # time tracking
        end = time.time()
        tracktime[var][iterii] = (end-start)
        if (end-start)>maxtim: 
            print (message)
            breakit = True
        else:
            breakit = False
            # break
        return tracktime, breakit

    def Estep(self, C_est=True, estA=True, estQ=True, estQ0=True, B_est=True, estx0=True, printit=False):
        # if S_est is True then all latent parameters are estimated (A, Q, Q0, xo)
        # estimate new x:
        xnew = np.zeros(self.x.shape) * np.nan
        muest = self.expand_mu(estA=estA, estx0=estx0)
        conv = np.zeros(self.Ttrials)*np.nan
        for ttrial in range(self.Ttrials):
            Ctilest = self.expand_C(est=C_est, ttrial=ttrial)
            Sest = self.expand_S(estA=estA, estQ=estQ, estQ0=estQ0, ttrial=ttrial)
            _, ystack = self.expand_xy(ttrial=ttrial)
            dtil = self.expand_d(est=B_est, ttrial=ttrial)
            if np.sum(self.estx) == 0:
                xtmp0 = muest[:(self.xdim * self.n_step[ttrial])] + np.random.randn((self.xdim * self.n_step[ttrial]))
            else:
                xtmp0 = self.estx[:self.n_step[ttrial], :, ttrial].reshape(self.n_step[ttrial] * self.xdim)
            resx = minimize(self.wrap_Lik, x0=xtmp0,
                            method='Newton-CG', jac=self.wrap_dLik, hess=self.wrap_HLik,
                            options={'disp': printit},
                            args=(ystack, Sest, muest, Ctilest, dtil, ttrial))  # , tol=1e-10)
            xnew[:self.n_step[ttrial], :, ttrial] = resx.x.reshape(self.n_step[ttrial], self.xdim)
            conv[ttrial] = np.copy(resx.success)
        return xnew, conv

    def compute_SIG(self, C_est=True, estA=True, estQ=True, estQ0=True, B_est=True, x_est=True, xtmp = None):
        allSIGesttmp = np.zeros([self.xdim * self.maxn_step, self.xdim * self.maxn_step, self.Ttrials]) * np.nan
        for ttrial in range(self.Ttrials):
            Ctilesttmp = self.expand_C(est=C_est, ttrial=ttrial)
            Sesttmp = self.expand_S(estA=estA, estQ=estQ, estQ0=estQ0, ttrial=ttrial)
            xstackesttmp, _ = self.expand_xy(est=x_est, ttrial=ttrial, xtmp=xtmp)
            dtiltmp = self.expand_d(est=B_est, ttrial=ttrial)
            allSIGesttmp[:(self.xdim * self.n_step[ttrial]), :(self.xdim * self.n_step[ttrial]), ttrial] = \
                np.linalg.inv(-self.HLik(xstackesttmp, Sesttmp, Ctilesttmp, dtil=dtiltmp, ttrial=ttrial))
        return allSIGesttmp
    # EM iterations

    #######################################################################
    ########################### EM ########################################
    #######################################################################

    def my_EM(self, ITER=1, printit=False, trackit=False,
             upx0=True, upQ0 = True, upA=True, upQ = True, upC = True, upB=False,\
             maxtimNew = 1000, maxtim = 1000, Adiag=False,\
             X = None, backtrack=False, backtrack_diff=.1, regA=False):
        # only uses the estimated values!

        # maxtim ... maximum time that is considered acceptabel for each parameter update/latent variable update
        # maxtimNew ... maximum time for Newton optimization
        # maxtimC ... maximum time for gradient descent update of C
        # if a parameter should not be updated, make sure that a reasonable (true?) value
        # is stored in the self.est... vector

        if X is None:
            X = self.stand_X()
        R = X.shape[1]

        if trackit:
            ESTx0 = np.zeros([self.xdim, ITER])*np.nan
            ESTA = np.zeros([self.xdim,self.xdim, ITER])*np.nan
            ESTQ0 = np.zeros([self.xdim,self.xdim, ITER])*np.nan
            ESTQ = np.zeros([self.xdim,self.xdim, ITER])*np.nan
            ESTC = np.zeros([self.ydim,self.xdim, ITER])*np.nan
            ESTx = np.zeros([self.maxn_step,self.xdim, ITER])*np.nan
        else:
            ESTx0 = 0
            ESTA = 0
            ESTQ0 = 0
            ESTQ = 0
            ESTC = 0
            ESTx = 0
            
        tracktime = pd.DataFrame(data = np.zeros([ITER,7]), columns=['x', 'x0', 'Q0', 'A', 'Q', 'B', 'C'])

        for iterii in range(ITER):

            ########################################
            ############# E-step ###################
            ########################################

            ### estimate x ###
            self.estx, conv = self.Estep(C_est=True, estA=True, estQ=True, estQ0=True, B_est=True, estx0=True)

            # compute Hessian:
            allSIGest = self.compute_SIG(C_est=True, estA=True, estQ=True, estQ0=True, B_est=True, x_est=True)

            if upC | upB:
                # compute old-parameter lower bound on P(y|parameters)
                curLold = self.LC(self.estx, self.estC, allSIGest, dtmp=self.estd)
            else:
                # compute old likelihood (from old parameters)
                curLold = self.yLik(B_est=True, x_est=True, C_est=True)

            # save old parameters
            if backtrack:

                if upx0:
                    oldx0 = np.copy(self.estx0)
                if upQ0:
                    oldQ0 = np.copy(self.estQ0)
                if upA:
                    oldA = np.copy(self.estA)
                if upQ:
                    oldQ = np.copy(self.estQ)
                if upC:
                    oldC = np.copy(self.estC)
                if upB:
                    oldB = np.copy(self.estB)

            ########################################
            ############# M-step ###################
            ########################################
            # update x0
            if upx0:
                start = time.time()
                if self.xdim>1:
                    self.estx0 = np.mean(self.estx[0,:,:], axis=1)
                else:
                    self.estx0 = np.mean(self.estx[0,:,:])
                end = time.time()
                tracktime['x0'][iterii] = (end-start)
                if (end-start)>maxtim:
                    print ('x0, too long (>maxtim)')
                    break


            # up Q0:
            if upQ0:
                if printit: print('updating Q0')
                Q0cov = np.zeros([self.xdim, self.xdim])
                for ttrial in range(self.Ttrials):
                    Q0cov = Q0cov + np.outer(self.estx0 - self.estx[0, :, ttrial],
                                             (self.estx0 - self.estx[0, :, ttrial]))

                start = time.time()
                self.estQ0 = np.nanmean(allSIGest[0:self.xdim, 0:self.xdim, :], axis=2) + Q0cov / self.Ttrials
                #self.estQ0 = (np.nanmean(allSIGest[0:self.xdim, 0:self.xdim,:], axis=2) + Q0cov)/self.Ttrials
                end = time.time()
                tracktime['Q0'][iterii] = (end-start)
                if (end-start)>maxtim:
                    print ('Q0, too long (>maxtim)')
                    break

            if upA:
                if printit: print('updating A')
                start = time.time()
                sumAest1 = 0
                sumAest2 = 0
                for ttrial in range(self.Ttrials):
                    # sum over M(t, t-1) and M(t-1, t-1)
                    for tt in range(1,self.n_step[ttrial]):
                        sumAest1 = sumAest1 + self.Mts(allSIGest[:,:,ttrial], tt, tt-1, ttrial=ttrial)
                        sumAest2 = sumAest2 + self.Mts(allSIGest[:,:,ttrial], tt-1, tt-1, ttrial=ttrial)
                if Adiag:
                    self.estA =  np.diag(np.diag(sumAest1.dot(np.linalg.inv(sumAest2))))
                else:
                    # update A
                    if self.xdim>1:
                        self.estA =  sumAest1.dot(np.linalg.inv(sumAest2)) # divide by K=Ttrials?
                    else:
                        self.estA = sumAest1/sumAest2
                end = time.time()
                tracktime['A'][iterii] = (end-start)
                if (end-start)>maxtim:
                    print ('A, too long (>'+np.str(maxtim)+')')
                    break
                if regA:
                    if self.xdim==1:
                        self.estA[self.estA>=1]=.9999
                    else:
                        tmp = np.linalg.eig(self.estA)
                        tmpeval = tmp[0]
                        tmpeval[tmpeval>=1] = .9999
                        self.estA = np.real(tmp[1].dot(np.diag(tmpeval)).dot(np.linalg.inv(tmp[1])))

            if upQ:
                if printit: print('updating Q')
                start = time.time()
                self.estQ = self.upQ(allSIGest)
                end = time.time()
                tracktime['Q'][iterii] = (end-start)
                if (end-start)>maxtim:
                    print ('Q, too long (>'+np.str(maxtim)+')')
                    break

            if upB:
                if printit: print('updating B')
                if R==1:
                    self.upd_closed(SIGest=allSIGest, X=X)
                else:
                    start = time.time()
                    #estb, _, _, _ = self.GD_B(self.estx, R, X, self.estC, SIGest=allSIGest,\
                    #                               estBo = self.estB, maxtim=300) # xstackest, Sest, Ctilest,
                    #self.estB = estb
                    resC = minimize(self.wrap_LB, x0=self.estB,
                                    method='BFGS', jac=self.wrap_dLB, options={'disp': False},
                                    args=(R, self.estx, self.estC, allSIGest, X))

                    self.estB = resC.x.reshape(self.ydim, R)
                    end = time.time()
                    tracktime['B'][iterii] = (end-start)
                    #if (end-start)>maxtim:
                    #    print('B, too long (>'+np.str(maxtim)+')')

            if upC:
                if printit: print('updating C')
                # update C
                start = time.time()
                resC = minimize(self.wrap_lc, x0=self.estC.reshape(self.xdim*self.ydim),
                       method='BFGS', jac=self.wrap_dc, options={'disp': False},
                      args=(self.estx, allSIGest, self.estd))

                self.estC = resC.x.reshape(self.ydim, self.xdim)
                end = time.time()
                tracktime['C'][iterii] = (end-start)
                #if (end-start)>maxtim:
                #    print ('C, too long (>'+np.str(maxtim)+')')
                #    break

        # potential backtracking if necessary        
        # compute new likelihood
        ### estimate x ###

        xnew, _ = self.Estep(C_est=True, estA=True, estQ=True, estQ0=True, B_est=True, estx0=True)

        # compute Hessian:
        allSIGesttmp = self.compute_SIG(C_est=True, estA=True, estQ=True, estQ0=True, B_est=True, x_est=True, xtmp=xnew)

        # compute new likelihood (with new parameters and fitted x)
        if upC | upB:
            # compute old-parameter lower bound on P(y|parameters)
            curLnew = self.LC(xnew, self.estC, allSIGesttmp, dtmp=self.estd)
        else:
            # compute old likelihood (from old parameters)
            curLnew = self.yLik(B_est=True, x_est=True, C_est=True, xtmp=xnew)

        if backtrack & ((curLold-curLnew)>backtrack_diff):
        # if new likelihood-old likelihood is smaller than the defined threshold,
        # backtracking kicks in and restores the old parameters
            # use old parameters
            if upx0:
                self.estx0 = np.copy(oldx0)
            if upQ0:
                self.estQ0 = np.copy(oldQ0)
            if upA:
                self.estA = np.copy(oldA)
            if upQ:
                self.estQ = np.copy(oldQ)
            if upC:
                self.estC = np.copy(oldC)
            if upB:
                self.estB = np.copy(oldB)
                self.estd = self.update_d(X, oldB)
            curLnew = np.copy(curLold)# self.yLik(B_est=True, x_est=True, C_est=True)
            print('backtracked, restored: ', curLnew)
        else:
            self.estx = np.copy(xnew)
        return ESTx0, ESTA, ESTQ0, ESTQ, ESTC, ESTx, conv, iterii, tracktime, curLnew, curLold, allSIGest
    
    
    # likelihood of data y
    
    def yLik(self, B_est=False, x_est = False, norm=False, C_est=False, xtmp = None):
        yL = np.array([0])
        for ttrial in range(self.Ttrials):
            Ctil = self.expand_C(est=C_est, ttrial=ttrial)
            dtil = self.expand_d(est=B_est, ttrial=ttrial)
            xtmpexp, ytmp = self.expand_xy(est=x_est, ttrial=ttrial, xtmp=xtmp)

            e = np.ones(self.ydim*self.n_step[ttrial])
            yL = yL + np.dot(ytmp.T, Ctil.dot(xtmpexp)+\
                                  dtil)-np.dot(e.T, self.bounded_exp(Ctil.dot(xtmpexp)+\
                                      dtil))
        if norm:
            if (np.sum(ytmp>170)==0):
                for tt in range(ytmp.shape[0]): # goes over dimensons and time because ytmp is ystack (see Macke 2015)
                    yL -= math.log(np.array([math.factorial(ytmp[tt])]))
            else:
                print('spike count>170')
        return yL

    # lower bound on likelihood: cost function for dynamics from linear Gaussian state space models




    def runEM(self, upA, upB, upQ, upQ0, upx0, upC, regA, Adiag=False, Xtrain=None,
              backtrack_diff=10, maxiter=50, maxtim=5000, fig = None, difflikthresh=.01,
              printit=False, backtracking=True, norm=False):
        if difflikthresh>=1:
            print('not running EM because difflikthresh>=1')
        if printit:
            print(self.ydim, ' neurons and average of ', np.mean(self.n_step), ' timepoints (per trial) and total of ', self.Ttrials,
                  ' trial(s)')
            print('fitting %.0f latent dimensions' % self.xdim)
            print(' ')
        iiwhile = 0
        conv = True
        DIFFLIK = []

        startEM = time.time()

        self.estx, _ = self.Estep(C_est=True, B_est=True, estA=True, estQ=True, estQ0=True, estx0=True)
        allSIG = self.compute_SIG(C_est=True, B_est=True, estA=True, estQ=True, estQ0=True, x_est=True)
        curLnew = self.LC(self.estx, self.estC, allSIG, dtmp=self.estd)
        percdiff = 1
        DIFFLIK.append(curLnew)
        self.EMconv = np.zeros(maxiter)*np.nan
        while (iiwhile < maxiter) & (np.max(conv * 1) == 1) & (percdiff > difflikthresh):  # (diffLik!=0):
            start = time.time()
            if printit:
                print('-----------------------------------------')
                print('iteration %.1f:' % (iiwhile + 1))
            _, _, _, _, _, _, conv, _, tracktime, newcurL, oldcurL, allSIG = \
                self.my_EM(ITER=1, trackit=False, maxtimNew=maxtim,
                          maxtim=maxtim,
                          upQ=upQ, upQ0=upQ0, upx0=upx0, upC=upC, upA=upA, Adiag=Adiag, upB=upB, X=Xtrain,
                          backtrack=backtracking, backtrack_diff=backtrack_diff, regA=regA)
            self.EMconv[iiwhile] = np.mean(conv)
            end = time.time()
            iiwhile += 1
            diffLik = (newcurL - oldcurL)
            percdiff = (diffLik / oldcurL)
            if percdiff<0:
                percdiff= 1
            DIFFLIK.append(newcurL)

            if printit:
                print(tracktime)
                print(' ')
                if upC | upB:
                    print('improvement in lower bound on lik for y: %.3f' % diffLik)
                else:
                    print('improvement in likelihood of data given estimated x and theta: %.3f' % diffLik)
                print('old ', oldcurL)
                print('new ', newcurL)
                print('relative ', (diffLik/oldcurL))
                if (diffLik/oldcurL) < 0:
                    print('adjusted relative ', percdiff)
                print(' ')
                print('time for EM iteration: %.3f' % (end - start))
                print('did converge:', np.mean(conv))
                icurL = self.yLik(B_est=True, x_est=True, C_est=True, norm=False)
                print('PLDS likelihood for y (unnormed): %.3f' % icurL)
                print(' ')
                # look at parameters
            if printit:
                if upA:
                    if (self.xdim > 1):
                        print('estimated real eigenvalues A = ', np.round((np.sort(np.linalg.eig(self.estA)[0])), 4))
                    else:
                        print('estimated A: ', self.estA)
                    print(' ')
                if upQ:
                    if self.xdim>1:
                        print('estimated eigenvalues Q = ', np.round(np.sort(np.linalg.eig(self.estQ)[0]), 4))
                    else:
                        print('estimated Q: ', self.estQ)
                    print(' ')
                if upC:
                    print('L2 norm of all C %.3f' %(np.sqrt(np.sum(self.estC**2))))
                if upQ0:
                    if self.xdim>1:
                        print('estimated eigenvalues Q0 = ', np.round(np.sort(np.linalg.eig(self.estQ0)[0]), 4))
                    else:
                        print('estimated Q0: ', self.estQ0)
                    print(' ')

        endEM = time.time()
        self.EMiter = iiwhile
        self.EMtime = endEM - startEM
        self.EMlik = DIFFLIK
        self.EMconv = self.EMconv[:iiwhile]

        if norm:
            normLik = self.yLik(B_est=True, x_est=True, C_est=True, norm=True)
        else:
            normLik = None
        if printit:
            print('totel time of EM: %.2f' % (endEM - startEM))
            print('normalized likelihood of model: ', normLik)
        return normLik, fig, DIFFLIK, allSIG, iiwhile-1

    def vis_lik_C(self, fig=None, ax=None, cscal=1, x_est=True, SIGest=None, ylik=False, maxtim=100000):
        # plt.rcParams.update({'font.size': 15})
        ub = cscal + cscal / 2
        lb = -(cscal + cscal / 2)
        Crange = np.arange(lb, ub, cscal / 100)
        x, y = np.meshgrid(Crange,Crange)
        #Ctmp = np.copy(self.estC)
        if x_est:
            xtmp, _ = self.Estep(C_est=True, estA=True, estQ=True, estQ0=True, B_est=False, estx0=True)
            if ylik:
                xtmp = np.copy(self.estx)
                self.estx = np.copy(xtmp)
        else:
            xtmp = np.copy(self.x)
        if fig is None:
            fig, ax = plt.subplots(1, len(self.xdim), figsize=(18, 4))
        for xxdim in range(self.xdim):
            Ctmp = np.copy(self.C)
            tmp = np.zeros(x.shape) * np.nan
            for ii in range(len(Crange)):
                for jj in range(len(Crange)):
                    Ctmp[0, xxdim] = x[ii, jj]
                    Ctmp[1, xxdim] = y[ii, jj]
                    if ylik:
                        tmp[ii, jj] = self.yLik(B_est=False, x_est=x_est, norm=False, C_est=True)
                    else:
                        tmp[ii, jj] = -self.wrap_lc(Ctmp, xtmp, SIGest, self.d)
            lev = np.linspace(np.min(tmp) - 10, np.max(tmp) + 10, 50)
            im = ax[xxdim].contourf(x, y, tmp, levels=lev)
            tmpmax = np.unravel_index(np.argmax(tmp), np.array(tmp).shape)
            ax[xxdim].plot(Crange[tmpmax[1]], Crange[tmpmax[0]], 'xr')
            ax[xxdim].plot(self.C[0, xxdim], self.C[1, xxdim], 'or')
            ax[xxdim].set_title('likelihood depending on latent  ' + np.str(xxdim + 1))
            ax[xxdim].set_xlabel('observable dimension 1')
            ax[xxdim].set_ylabel('observable dimension 2')
            plt.colorbar(im, ax=ax[xxdim])
        if ylik:
            LIKNAM = 'P(y|x, theta)'
        else:
            LIKNAM = 'lower bound on likelihood (observational part)'
        if x_est:
            print(LIKNAM + ' surface plotted given correct B and estimated x')
        else:
            print(LIKNAM + ' surface plotted given correct B and true x')
        print('#####################################################')
        if ylik:
            self.estx = np.copy(xtmp)


def vis_est_latent(MOD, C_est=False, estA=False, estQ=False, estQ0=False, B_est=False, estx0=False):
    fig_der = plt.figure(figsize=(12, 3))
    axres = plt.subplot2grid((1, 2), (0, 0), rowspan=1, colspan=1)
    axHL = plt.subplot2grid((1, 2), (0, 1), rowspan=1, colspan=1)
    #axL = plt.subplot2grid((1, 3), (0, 2), rowspan=1, colspan=1)

    Ctil, S, _, xstack, _, dtil = MOD.expand(ttrial=0)
    axHL.imshow(MOD.HLik(xstack, S, Ctil, dtil))
    axHL.set_title('Hessian (trial 0)')

    Xfin, conv = MOD.Estep(C_est=C_est, estA=estA, estQ=estQ, estQ0=estQ0, B_est=B_est, estx0=estx0,
                           printit=False)

    for jj in range(MOD.xdim):
        axres.plot(MOD.x[:, jj, :], Xfin[:, jj, :], '.')
    axres.set_ylabel('estimated latent variable')
    axres.set_xlabel('real latent variable')
    #axres.set_aspect(1)
    #axL.plot(L)
    #axL.set_title('Likelihood over Newton iterations')
    #axL.set_xlabel('iterations')
    return Xfin