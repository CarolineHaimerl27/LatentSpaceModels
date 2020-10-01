# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 11:55:33 2018

@author: caroline
"""

# take a Macke_PLDS object and get leave-neuron-out predictive log likelihood 
# comparison in 3 ways
# (interacts with the Macke_PLDS class)
# 1. with respect to stimulus-response only model
# 2. with respect to dimensionality
# 3. with respect to random (unlearned) parameter model

import numpy as np
import matplotlib.pyplot as plt
#import time
from linear_regression import myL2norm, RFlin
from Macke_PLDS import PLDS
from sklearn.decomposition import FactorAnalysis
import pickle


def model_train(data, DtM, xdim, ee, fig=None, norm=False, printit=False, saveresults=False):
    if xdim < 1:
        print('error, latent dimension must be >=1')
    ###############################################################################################################
    ######################################### set up model: #######################################################
    ###############################################################################################################
    np.random.seed(ee)
    data_trial = data.data_trial[:, :, DtM.TRAINTRIALS[:,ee]]
    Xtrain = data.X[:, :, DtM.TRAINTRIALS[:,ee]]
    # create model
    MOD = PLDS()
    # initialize A and C
    if DtM.estC is not None:
        estC = DtM.estC
    else:
        if DtM.C_starting is not None:
            estC = DtM.C_starting[xdim-DtM.MINDIM][:,:,ee]
        else:
            estC=None
    if DtM.estA is not None:
        estA = DtM.estA
    else:
        if DtM.A_starting is not None:
            estA = DtM.A_starting[xdim-DtM.MINDIM]
        else:
            estA=None
    # initialize noise paramterers
    if DtM.estQ is None:
        estQ = np.eye(xdim)*DtM.scalQ
    else:
        estQ = np.copy(DtM.estQ)
    if DtM.estQ0 is None:
        estQ0 = np.eye(xdim)*DtM.scalQ0
    else:
        estQ0 = np.copy(DtM.estQ0)
    MOD.par(xdim, ydim=np.copy(data_trial.shape[1]), estx0=DtM.estx0,
            est=True, X=Xtrain, Ttrials=len(DtM.TRAINTRIALS[:,ee]), y=data_trial, n_step=data.counts0[DtM.TRAINTRIALS[:,ee]], seed=ee,
            estA=estA, estQ=estQ, estQ0=estQ0, estC=estC, estB=DtM.estB)

    ###############################################################################################################
    ######################################### fit model: ##########################################################
    ###############################################################################################################
    if printit: print('################### fit PLDS to data ########################')
    # initialize
    if (xdim>1)&(DtM.residuals)&(DtM.estC is not None):
        factor = FactorAnalysis(n_components=MOD.xdim, random_state=ee).fit(DtM.RESIDUALS[:,:,ee])
        MOD.estC = factor.components_.T/np.max(np.abs(factor.components_))

    normLik, fig, diffLik, allSIG, iiwhile = MOD.runEM(upA=DtM.upA, upB=DtM.upB, upQ=DtM.upQ, upQ0=DtM.upQ0, upx0=DtM.upx0, upC=DtM.upC, regA=DtM.regA,
                                                       Adiag=DtM.Adiag, Xtrain=Xtrain,
                                                       backtrack_diff=DtM.backtrack_diff, maxiter=DtM.maxiter,
                                                       maxtim=DtM.maxtim, fig=fig,
                                                       difflikthresh=DtM.difflikthresh, printit=printit,
                                                       backtracking=DtM.backtracking, norm=norm)

    if saveresults:
        pickle.dump(MOD, open(DtM.path + DtM.name + '_PLDS_ncross_' + np.str(ee) + '_xdim_' + np.str(MOD.xdim) + '.pk', 'wb'))
    else:
        print('not saving this')
    return MOD, normLik, fig, diffLik, allSIG, iiwhile


def lnu_test_expand_C(MOD, Ctmp, n_leftout, dtmp, ttrial=None):
    if ttrial is None:
        n_step = MOD.maxn_step
    else:
        n_step = MOD.n_step[ttrial]
    # Ctilde
    Ctil = np.zeros([n_step*n_leftout, n_step*MOD.xdim])
    for ii in range(n_step):
        Ctil[(ii*n_leftout):((ii+1)*n_leftout),(ii*MOD.xdim):((ii+1)*MOD.xdim)] = Ctmp
    # dtilde
    dtil = np.zeros([n_step*n_leftout])
    for ii in range(n_step):
        dtil[(ii*n_leftout):((ii+1)*n_leftout)] = dtmp[ii, :]
    return Ctil, dtil


def compute_latent(nnout, MODall, seedtest, X, counts0, data_trial, testtrials):
    mask = np.ones(MODall.ydim, dtype=bool)
    mask[nnout] = False
    # create variables without the left out neuron
    datacuttimtest_nn = data_trial[:, mask, :]
    # create testing model without left out neuron
    MOD_test_nn = PLDS()
    if len(testtrials)==1:
        n_step = np.array(counts0[testtrials])
    else:
        n_step = counts0[testtrials]
    MOD_test_nn.par(MODall.xdim, MODall.ydim - len(nnout), seed=seedtest, est=True,
                    y=datacuttimtest_nn[:, :, testtrials], Ttrials=len(testtrials), n_step=n_step,
                    X=X[:, :, testtrials],
                    C=MODall.C[mask, :], Q0=MODall.Q0, A=MODall.A, Q=MODall.Q, x0=MODall.x0, B=MODall.B[mask, :])

    # estimate the latent from the majority of neurons
    xfin, _ = MOD_test_nn.Estep(C_est=False, estA=False, estQ=False, estQ0=False, B_est=False,
                                estx0=False)
    return xfin, MOD_test_nn

def model_test_lno(MODall, testtrials, seedtest, data_trial, X, counts0, path=None, name=None, ee=None,
                   whichneuron=None,
                   rotate=False, cho_est=None, evecest=None, As=None, AvT=None, Au=None, saveresults=False,
                   pred=False):
    ###############################################################################################################
    # test model: 
    ###############################################################################################################
    if pred:
        PRED = np.zeros(MODall.y.shape)*np.nan
    if whichneuron is None:
        whichneuron = np.arange(MODall.ydim)
    if rotate:  # if true that error is estimated for every added dimension separatedly and dimensions are ordered
                # depending on their temporal component
        MSE = np.zeros([len(whichneuron), MODall.xdim, 2])*np.nan
        # LIK = np.zeros([len(whichneuron), MOD.xdim, 2]) * np.nan
    else:
        MSE = np.zeros([len(whichneuron)])*np.nan
        # LIK = np.zeros([len(whichneuron)]) * np.nan
    for nnoutii in range(len(whichneuron)):
        nnout = np.array([whichneuron[nnoutii]])
        # compute the latent on all but the nnoutii neuron
        xfin, MOD_test_nn = compute_latent(nnout, MODall, seedtest, X, counts0, data_trial, testtrials)

        # test prediction for activity of remaining neuron
        if rotate:
            mse_plds = np.zeros([MOD_test_nn.Ttrials, MODall.xdim])*np.nan
            mse_plds_cum = np.zeros([MOD_test_nn.Ttrials, MODall.xdim])*np.nan
        else:
            mse_plds = np.zeros([MOD_test_nn.Ttrials])*np.nan
            mse_plds_cum = None
        if rotate:
            for ttrial in range(MOD_test_nn.Ttrials):
                data_tt = data_trial[:MOD_test_nn.n_step[ttrial], nnout,testtrials[ttrial]]
                # rotate latent x and mapping matrix C
                estxdeg = (np.linalg.inv(cho_est).dot(evecest.T).dot(xfin[:MOD_test_nn.n_step[ttrial],:,ttrial].T)).T
                xdeg_arot = np.diag(np.sqrt(As)).dot(AvT).dot(estxdeg.T).T
                estCdeg = MODall.C.dot(evecest.dot(cho_est))
                Cdeg_arot = estCdeg.dot(Au).dot(np.diag(np.sqrt(As)))
                for xx in range(MODall.xdim):
                    pred_tt = np.exp(xdeg_arot[:, xx]*(Cdeg_arot[nnout, xx]) +
                                    MODall.d[:MOD_test_nn.n_step[ttrial], nnout, testtrials[ttrial]])
                    mse_plds[ttrial, xx] = np.sum((pred_tt -data_tt)**2) / MOD_test_nn.n_step[ttrial]
                    pred_tt_cum = np.exp(xdeg_arot[:, :(xx+1)].dot(Cdeg_arot[nnout, :(xx+1)].T) +
                                    MODall.d[:MOD_test_nn.n_step[ttrial], nnout, testtrials[ttrial]])
                    mse_plds_cum[ttrial, xx] = np.sum((pred_tt_cum - data_tt)**2) / MOD_test_nn.n_step[ttrial]
                if pred:
                    PRED[:MODall.n_step[testtrials[ttrial]], nnoutii, testtrials[ttrial]] = pred_tt_cum[:,0]

        else:
            for ttrial in range(MOD_test_nn.Ttrials):
                data_tt = data_trial[:MOD_test_nn.n_step[ttrial], nnout,testtrials[ttrial]]
                pred_tt = np.exp(xfin[:MOD_test_nn.n_step[ttrial],:,ttrial].dot(MODall.C[nnout,:])+
                                 MODall.d[:MOD_test_nn.n_step[ttrial], nnout, testtrials[ttrial]])
                mse_plds[ttrial] = np.sum((pred_tt-data_tt)**2) /\
                                   MOD_test_nn.n_step[ttrial]

                if pred:

                    PRED[:MODall.n_step[testtrials[ttrial]], nnoutii, testtrials[ttrial]] = pred_tt[:,0]


        if rotate:
            MSE[nnoutii, :, 0] = np.nanmean(mse_plds, axis=0) # error for each dimension by itself
            MSE[nnoutii, :, 1] = np.nanmean(mse_plds_cum, axis=0) # if an increasing number of dimensions is used
        else:
            MSE[nnoutii] = np.nanmean(mse_plds)
    if saveresults: pickle.dump(MSE, open(path + name + 'MSE_PLDS_ncross_' + np.str(ee) + '_xdim_' + np.str(MODall.xdim) + '.pk', 'wb'))
    if pred==False: PRED = None
    return MSE, PRED

def fit_to_all_trials(data_trial, MOD, counts0, X, seedtest):
    MODall = PLDS()
    MODall.par(xdim=MOD.xdim, ydim=MOD.ydim, seed = seedtest,
               est = True, y = data_trial, Ttrials=len(counts0),
                n_step=counts0,C = MOD.estC, Q0 = MOD.estQ0,A = MOD.estA,
               Q = MOD.estQ, x0 = MOD.estx0, B=MOD.estB, X=X)
    # estimate the latent
    MODall.estx, _ = MODall.Estep(C_est=False, estA=False, estQ=False, estQ0=False, B_est=False,
                                                estx0=False)
    return MODall


def PLDS_rotations(MODall, scal=1, plotit=False, printit=False):
    R = MODall.B.shape[1]
    # restructure so Q is identity
    if plotit:
        fig, ax = plt.subplots(1, 3, figsize=(18, 4))

    # remove degeneracy
    if MODall.xdim == 1:
        estCdeg = MODall.C * np.sqrt(MODall.Q)
        Cdeg_arot = np.copy(estCdeg)
        estxdeg = MODall.estx / np.sqrt(MODall.Q)
        estx0deg = MODall.x0 / np.sqrt(MODall.Q)
        if plotit:
            ax[0].plot(estCdeg)
            ax[0].set_title('C')
        estAdeg = np.copy(MODall.A)
        xdeg_arot = np.copy(estxdeg)
        As = np.copy(MODall.A)
        AvT = np.ones(1)
        Au = np.ones(1)
        cho_est = np.sqrt(MODall.Q)
        evecest = np.ones(1)
        if printit: print('A rotated for Q=I: ', estAdeg)
    else:
        evaluest, evecest = np.linalg.eig(MODall.Q)
        cho_est = np.diag(np.sqrt(evaluest / scal))
        estAdeg = np.diag(1 / np.sqrt(evaluest)).dot(evecest.T).dot(MODall.A).dot(evecest.dot(cho_est))

        estCdeg = MODall.C.dot(evecest.dot(cho_est))
        estxdeg = np.zeros([MODall.maxn_step, MODall.xdim, MODall.Ttrials]) * np.nan
        for ttrial in range(MODall.Ttrials):
            estxdeg[:(MODall.n_step[ttrial]), :, ttrial] = (np.linalg.inv(cho_est).dot(evecest.T).dot( \
                MODall.estx[:MODall.n_step[ttrial], :, ttrial].T)).T
        estx0deg = (np.diag(1 / np.sqrt(evaluest)).dot(evecest.T).dot(MODall.x0.T)).T

        # rotate parameters to correpond to the two A eigenspectra using svd
        Au, As, AvT = np.linalg.svd(estAdeg)
        Cdeg_arot = estCdeg.dot(Au)
        if printit: print('A singular values: ', As)
        if printit: print('A rotated eigenvalues: ', np.sort(np.linalg.eig(estAdeg)[0]))

        xdeg_arot = np.zeros([MODall.maxn_step, MODall.xdim, MODall.Ttrials]) * np.nan
        for ttrial in range(MODall.Ttrials):
            xdeg_arot[:(MODall.n_step[ttrial]), :, ttrial] = AvT.dot(estxdeg[:MODall.n_step[ttrial], :, ttrial].T).T

        if plotit:

            ax[0].plot(Cdeg_arot[:, 0], '.')
            ax[0].set_title('A-rotated-C1')
            ax[1].plot(Cdeg_arot[:, 1], '.')
            ax[1].set_title('A-rotated-C2')
            if MODall.xdim > 2:
                ax[3].plot(Cdeg_arot[:, 2], '.')
                ax[3].set_title('A-rotated-C3')

    if printit:
        print('x0 rotated for Q=I ', estx0deg)
        # print('est Q0:', MOD.estQ0)
        print('')

    if plotit:
        plt.figure(figsize=(17, 4))
        cmap = plt.cm.get_cmap('RdYlGn')
        ax[0] = plt.subplot2grid((1, 2), (0, 0))
        ax[1] = plt.subplot2grid((1, 2), (0, 1))
        ax[0].set_title('estimated stimulus response coefficients')
        for ii in range(R):
            ax[0].plot(MODall.B[:, ii], '--', color=cmap((ii / (R))), label=(ii + 1))
            ax[1].boxplot(MODall.B[:, ii], positions=np.array([ii + 1]), patch_artist=True, \
                          boxprops=dict(facecolor=cmap(ii / (
                              R))))  # , labels=['loc_1_or_1', 'loc_1_or_2', 'loc_1_or_3', 'loc_1_or_4','loc_2_or_1', 'loc_2_or_2', 'loc_2_or_3', 'loc_2_or_4']);
            ax[0].legend()
        ax[1].set_xlim(0, R + 1)
        ax[1].set_title('distribution of stimulus coefficients for each stimulus')
    return estCdeg, Cdeg_arot, estxdeg, estx0deg, estAdeg, xdeg_arot, \
           As, AvT, Au, cho_est, evecest


def MTcomp(data, MT, xdeg_arot, xdim, lam = 0.01, Ttrain=10):
    MSE = np.zeros([Ttrain,4])
    # SR
    X = np.zeros([data.X.shape[0]*data.Ttrials, data.R])
    for xx in range(data.R):
        X[:,xx] = data.X[:,xx,:].reshape(data.X.shape[0]*data.Ttrials)
    XTMP = X[np.isnan(X[:,0])==False,:]

    logMT = np.copy(MT)
    logMT[logMT==0] = .0000001
    logMT = np.log(logMT)

    # PLDS
    Xlat = np.zeros([xdeg_arot.shape[0]*data.Ttrials, data.R+xdim])
    for xx in range(data.R):
        Xlat[:,xx] = data.X[:xdeg_arot.shape[0],xx,:].reshape(xdeg_arot.shape[0]*data.Ttrials)
    for xx in range(xdim):
        Xlat[:,data.R+xx] = xdeg_arot[:,xx,:].reshape(xdeg_arot.shape[0]*data.Ttrials)
    XTMPlat = Xlat[np.isnan(Xlat[:,0])==False,:]
    BETA = np.zeros([Ttrain, data.R+xdim, 2])*np.nan

    for ss in range(Ttrain):
        np.random.seed(ss)
        train = np.random.choice(np.array(np.round(XTMP.shape[0]*.95), dtype='int'), XTMP.shape[0])
        mask = np.ones(XTMP.shape[0], dtype='bool')
        mask[train] = False
        # SR
        #beta_sr, beta0_sr = RFlin_beta(XTMP[train,:], logMT[train], lam=lam) # might need to optimize for lambda here!
        beta_sr, beta0_sr = np.polyfit(XTMP[train,:], logMT[train], deg=1)
        pred_sr = np.exp(RFlin(XTMP[mask], beta_sr, beta0_sr))
        beta_sr[-1] += beta0_sr
        BETA[ss, :, 0] = np.concatenate([np.array([beta_sr]).T, np.zeros([xdim,1])], axis=0)[:,0]

        MSE[ss, 0] = np.mean((MT[mask]-pred_sr)**2)
        MSE[ss, 2] = np.corrcoef(MT[mask], pred_sr)[0,1]
        # PLDS
        beta, beta0 = RFlin_beta(XTMPlat[train,:], logMT[train], lam=lam) # might need to optimize for lambda here!
        #beta, beta0 = np.polyfit(XTMP[train, :], logMT[train], deg=1)
        pred = np.exp(RFlin(XTMPlat[mask], beta, beta0))
        '''fig, ax = plt.subplots(1,2,figsize=(13,5))

        ax[0].plot(MT)
        ax[0].plot(pred_sr)
        ax[0].plot(pred)
        ax[1].plot(MT, pred_sr, '.')
        ax[1].plot(MT, pred, '.')
        print('MSE %.3f' %(np.mean((MT-pred)**2)))
        print('cc %.3f' %(np.corrcoef(MT, pred)[0,1]))'''
        MSE[ss, 1] = np.mean((MT[mask]-pred)**2)
        MSE[ss, 3] = np.corrcoef(MT[mask], pred)[0,1]
        beta[data.R-1] += beta0
        BETA[ss, :, 1] = beta

    return MSE, BETA, XTMPlat


def vis_SR(SPIKES, testtrials, nn, PRED, beta1, beta2, X, stim_dim, N_timebins, model=1, on_off=False, pred2=None):
    fig, ax = plt.subplots(1,1, figsize=(15,5))
    tmp = SPIKES[:,nn, testtrials].T.reshape(PRED.shape[0]*len(testtrials))
    tmp = tmp[np.isnan(tmp)==False]
    ax.plot([0, len(tmp)], [np.mean(tmp), np.mean(tmp)], '-', color='grey')
    ax.plot(tmp, 'k', label='data')
    tmp_pred = PRED[:,nn, testtrials].T.reshape(PRED.shape[0]*len(testtrials))
    ax.set_title('neuron %.0f' %nn)
    tmp_pred = tmp_pred[np.isnan(tmp_pred)==False]
    ax.plot([0, len(tmp_pred)], [np.mean(tmp_pred), np.mean(tmp_pred)], '--', color='orange')
    ax.plot(tmp_pred, '--r',label='model')
    if pred2 is not None:
        tmp_pred2 = pred2[:, nn, testtrials].T.reshape(PRED.shape[0] * len(testtrials))
        tmp_pred2 = tmp_pred2[np.isnan(tmp_pred2) == False]
        ax.plot(tmp_pred2, '--b', label='model 2')
    ax.legend()
    ax.set_xlabel('time bins')
    ax.set_ylabel('firing rate')
    if model==1:
        fig, ax = plt.subplots(1, 1, figsize=(15, 2))
        cmap=plt.cm.coolwarm
        for xx in range(X.shape[1]-1):
            xtmp = X[:,xx, testtrials].T.reshape(PRED.shape[0]*len(testtrials))
            ax.plot(xtmp[np.isnan(xtmp)==False], label=xx, color=cmap(xx/(X.shape[1]-1))) # color=cmap(np.floor(xx/4)/len(stim_dim)))
        ax.set_xlabel('time bins')
        ax.set_ylabel('stimulus on/off')
        names = ('orientation 1, contrast low', 'orientation 1, contrast high',
                 'orientation 2, contrast low', 'orientation 2, contrast high')
        # look at filter:
        fig2, ax2 = plt.subplots(1, 2, figsize=(10, 3))
        ax2[0].plot([0, N_timebins], [0, 0], '--k')
        ax2[1].plot([0, N_timebins], [0, 0], '--k')
        for rr in range(len(stim_dim)):
            ax2[0].plot(beta1[(rr * N_timebins):((rr + 1) * N_timebins), nn],
                        color=cmap(rr * N_timebins / (X.shape[1] - 1)))
            if beta2 is not None: ax2[1].plot(beta2[(rr * N_timebins):((rr + 1) * N_timebins), nn],
                                              color=cmap(rr * N_timebins / (X.shape[1] - 1)), label=names[rr])
        ax2[0].set_title('beta1')
        ax2[1].set_title('beta2')
        ax2[1].legend()
    if model==2:
        fig, ax = plt.subplots(1, 3, figsize=(15,4))
        tmp_x = np.zeros([PRED.shape[0] * len(testtrials), X.shape[1]])*np.nan
        for ss in range(X.shape[1]):
            tmp_x[:,ss] = X[:, ss, testtrials].T.reshape(PRED.shape[0] * len(testtrials))
        tmp_x = tmp_x[np.isnan(np.sum(tmp_x, axis=1))==False]
        for ss in range(2):
            e = np.random.randn(np.sum(tmp_x[:,0]==ss))*.1
            ax[ss].plot(tmp_x[tmp_x[:,0]==ss, 1:(-1-on_off)].dot(np.arange(N_timebins))+e, tmp[tmp_x[:,0]==ss], '.', label='data')
            ax[ss].plot(tmp_x[tmp_x[:,0]==ss, 1:(-1-on_off)].dot(np.arange(N_timebins))+e, tmp_pred[tmp_x[:,0]==ss], 'x', label='model')
            ax[ss].set_xlabel('time window')
            ax[ss].set_ylabel('FR')
        for ss in range(2):
            ax[2].plot(tmp_x[tmp_x[:, 0] == ss, 1:(-1-on_off)].dot(np.arange(N_timebins)) + np.random.randn(np.sum(tmp_x[:,0]==ss))*.1,
                       tmp[tmp_x[:, 0] == ss] - tmp_pred[tmp_x[:, 0] == ss], '.', label='contrast '+np.str(ss))
        ax[0].set_title('contrast condition 1')
        ax[0].legend()
        ax[2].legend()
        ax[1].set_title('contrast condition 2')
        ax[2].set_title('residuals')
        ax[0].set_ylim(0, np.max([np.max(tmp), np.max(tmp_pred)]))
        ax[1].set_ylim(0, np.max([np.max(tmp), np.max(tmp_pred)]))

    #for rr in range(data.R - 2):
    #    ax2[rr, 0].set_ylabel(names[rr])



def vis_SR_pop(SPIKES, testtrials, PRED, SR_MSE, D, nn=0, pred2=None):
    fig, ax = plt.subplots(1,1, figsize=(10,3))
    ax.plot(np.mean(np.nanmean(SPIKES, axis=0), axis=1), 'k', label='data')
    ax.plot(np.mean(np.nanmean(PRED, axis=0), axis=1),'--r', label='pred')
    if pred2 is not None:
        ax.plot(np.mean(np.nanmean(pred2, axis=0), axis=1), '--b', label='pred 2')
    #ax.plot(np.exp(beta[-1,:]), label='coef')
    ax.legend()
    ax.set_xlabel('neurons')
    ax.set_ylabel('mean firing')
    error = np.mean(np.nanmean(SPIKES, axis=0), axis=1) - np.mean(np.nanmean(PRED, axis=0), axis=1)

    if SR_MSE is not None:
        fig2, ax2 = plt.subplots(1,2, figsize=(10,3))
        ax2[0].plot(SPIKES[:,:, testtrials].reshape(PRED.shape[0]*len(testtrials)*D),
                  PRED[:,:, testtrials].reshape(PRED.shape[0]*len(testtrials)*D), 'ok')
        ax2[0].plot(SPIKES[:,nn, testtrials].T.reshape(PRED.shape[0]*len(testtrials)),
                  PRED[:,nn, testtrials].T.reshape(PRED.shape[0]*len(testtrials)), 'ro')
        ax2[0].set_aspect('equal')
        ax2[0].set_xlabel('true spikes')
        ax2[0].set_xlabel('predicted spikes')
        ax2[0].set_title('testtrials (red=example neuron')
        ax2[1].boxplot(SR_MSE[:,0]) # 0 because we only assume one crossvalidation
        ax2[1].set_ylabel('MSE')
    return fig, ax, error


# look at residuals
def distr_res(X, pred, SPIKES, neurons, joint=True):
    vals = np.unique(X)
    vals = vals[np.isnan(vals)==False]

    xres = (X[:,-(1+joint),:]).reshape(pred.PRED.shape[0]*pred.PRED.shape[2])
    xres = xres[np.isnan(xres)==False]
    xres = (xres==vals[1])

    fig, ax = plt.subplots(4,2,figsize=(17,16))


    resid_pois = pred.PRED_pois-SPIKES
    if pred.beta is not None:
        resid = pred.PRED - SPIKES
        maxb = np.nanmax(resid)
        minb = np.nanmin(resid)
    else:
        maxb = np.nanmax(resid_pois)
        minb = np.nanmin(resid_pois)
    bins=np.arange(minb-10,maxb+10, 20)
    bins2 = np.arange(-100, 400, 20)

    maxb = np.nanmax(resid_pois)
    minb = np.nanmin(resid_pois)
    bins3 = np.arange(minb - 10, maxb + 10, 20)
    ax[0,0].plot([0,0], [0,1], '--', color='grey')
    ax[0,1].plot([0,0], [0,1], '--', color='grey')
    ax[1,0].plot([0,0], [0,1], '--', color='grey')
    ax[1,1].plot([0,0], [0,1], '--', color='grey')
    ax[2,0].plot([0,0], [0,1], '--', color='grey')
    ax[2,1].plot([0,0], [0,1], '--', color='grey')
    ax[3,0].plot([0,0], [0,1], '--', color='grey')
    ax[3,1].plot([0,0], [0,1], '--', color='grey')

    for nn in range(len(neurons)):
        if pred.beta is not None:
            res = resid[:,neurons[nn],:].reshape(pred.PRED.shape[0]*pred.PRED.shape[2])
            res = res[np.isnan(res)==False]
            if np.sum(xres>0) > 0: # stim on
                htmp = np.histogram(res[xres>0], bins=bins)
                ax[0,0].plot(bins[:-1], htmp[0]/np.nansum(htmp[0]), '-', color=plt.cm.coolwarm(nn/len(neurons))) # (neurons[nn]-np.min(neurons))/np.max(neurons)))
            if np.sum(xres==0)>0: #stim off
                htmp = np.histogram(res[xres==0], bins=bins)
                ax[0,1].plot(bins[:-1], htmp[0]/np.nansum(htmp[0]), '-', color=plt.cm.coolwarm(nn/len(neurons))) # (neurons[nn]-np.min(neurons))/np.max(neurons)))

        res = resid_pois[:,neurons[nn],:].reshape(pred.PRED.shape[0]*pred.PRED.shape[2])
        res = res[np.isnan(res)==False]

        if np.sum(xres > 0) > 0:
            htmp = np.histogram(res[xres > 0], bins=bins3)
            ax[1,0].plot(bins3[:-1], htmp[0]/np.nansum(htmp[0]), '-', color=plt.cm.coolwarm(nn/len(neurons))) # (neurons[nn]-np.min(neurons))/np.max(neurons)))
        if np.sum(xres == 0) > 0:
            htmp = np.histogram(res[xres==0], bins=bins3)
            ax[1,1].plot(bins3[:-1], htmp[0]/np.nansum(htmp[0]), '-', color=plt.cm.coolwarm(nn/len(neurons)))

        spikes = SPIKES[:,neurons[nn],:].reshape(pred.PRED.shape[0]*pred.PRED.shape[2])
        spikes[spikes==0]=.000001
        if pred.beta is not None:
            res = pred.logPRED[:,neurons[nn],:].reshape(pred.PRED.shape[0]*pred.PRED.shape[2])-np.log(spikes)
            res = res[np.isnan(res)==False]

            if np.sum(xres > 0) > 0:
                htmp = np.histogram(res[xres>0], bins=bins2)
                ax[2,0].plot(bins2[:-1], htmp[0]/np.nansum(htmp[0]), '-', color=plt.cm.coolwarm(nn/len(neurons))) # (neurons[nn]-np.min(neurons))/np.max(neurons)))
            if np.sum(xres==0)>0:
                htmp = np.histogram(res[xres==0], bins=bins2)
                ax[2,1].plot(bins2[:-1], htmp[0]/np.nansum(htmp[0]), '-', color=plt.cm.coolwarm(nn/len(neurons))) # (neurons[nn]-np.min(neurons))/np.max(neurons)))

        res = pred.PRED_pois[:,neurons[nn],:].reshape(pred.PRED.shape[0]*pred.PRED.shape[2])-np.log(spikes)
        res = res[np.isnan(res)==False]

        if np.sum(xres > 0) > 0:
            htmp = np.histogram(res[xres > 0], bins=bins2)
            ax[3,0].plot(bins2[:-1], htmp[0]/np.nansum(htmp[0]), '-', color=plt.cm.coolwarm(nn/len(neurons))) # (neurons[nn]-np.min(neurons))/np.max(neurons)))
        if np.sum(xres == 0) > 0:
            htmp = np.histogram(res[xres==0], bins=bins2)
            ax[3,1].plot(bins2[:-1], htmp[0]/np.nansum(htmp[0]), '-', color=plt.cm.coolwarm(nn/len(neurons)))



    ax[0,0].set_ylabel('%')
    ax[0,0].set_title('stimulus on (linreg)')
    ax[0,1].set_title('stimulus off')
    ax[3,0].set_xlabel('residual')
    ax[1,0].set_ylabel('%')
    ax[1,0].set_title('stimulus on (pois)')
    ax[1,1].set_title('stimulus off')
    ax[2,0].set_ylabel('% (log linear)')
    ax[3,0].set_ylabel('% (pois)')