# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 2018

@author: caroline
"""

import numpy as np
import matplotlib.pyplot as plt
import math

# function to summarize
class spiking:
    def par(self, tmax, dt, time, n_step, N, nu, mu, lbd_d, lbd_v, decoderscaling, J=1):
        self.tmax = tmax
        self.dt = dt
        self.time = time
        self.n_step = n_step
        self.N = N
        self.nu=nu # linear Spike Cost
        self.mu=mu # quadratic cost: distributing activity
        self.lbd_d = lbd_d # Readout Decay Rate, time constant of the decoder
        self.lbd_v = lbd_v # Membrane Leak, time constant of the membrane
        self.decoderscaling = decoderscaling
        self.J = J
    
    def signal(self, X, optpar=1):
        X_dot = np.zeros((self.J,self.n_step))
        if X=='sine':
            X = np.zeros((self.J,self.n_step))
            X[0,:] = np.cos(0.7*(self.time-self.tmax/10)) # np.cos(self.time*np.pi*optpar)   
            X_dot[0,:] = -0.7*np.sin((0.7*(self.time-self.tmax/10))) # np.pi*np.sin(self.time*np.pi)
            if self.J==2:
                X[1,:] = np.sin(0.7*(self.time-self.tmax/10)) # np.sin(self.time*np.pi*optpar)
                X_dot[1,:] = 0.7*np.cos((0.7*(self.time-self.tmax/10))) # np.pi*np.cos(self.time*np.pi)
        elif X=='cons':
            X = np.zeros((self.J,self.n_step))+optpar
            X_dot[0,1:] = np.diff(X[0,:])
            if self.J==2:
                X[1,:] = np.zeros((self.J,self.n_step))+optpar
                X_dot[1,1:] = np.diff(X[0,:])                
        elif X=='LDS':
            X = np.zeros((self.J,self.n_step))
            if self.J==2:
                X[0,:] = optpar[0,:]
                X_dot[0,1:] = np.diff(X[0,:])
                X[1,:] = optpar[1,:]
                X_dot[1,1:] = np.diff(X[1,:])
            else:
                X[0,:] = optpar
                X_dot[0,1:] = np.diff(X[0,:])
        c = X_dot + self.lbd_d * X
        return X, c
    
    def connect(self, Dtyp):
        if Dtyp=='hetero':
            # heterogeneous
            if self.J==1:

                z = np.linspace(0,2 * np.pi,self.N + 1)
                angles = [np.cos(r) for r in z]
                del angles[-1]
                D = np.asarray(angles).T

                # scaling
                D *= self.decoderscaling / float(self.N)

                # Threshold
                T = (np.diag(np.outer(D, D)) + self.nu + self.mu) * .5

                # Recurrent Weights
                Omega = - np.outer(D, D) - self.mu * np.eye(self.N)
                
            else:
                z = np.linspace(0,2 * np.pi,self.N + 1)
                angles = [(np.cos(r),np.sin(r)) for r in z]
                del angles[-1]
                D = np.asarray(angles).T

                # scaling
                D *= self.decoderscaling / float(self.N)

                # Threshold
                T = np.reshape(np.sum(D ** 2,0) + self.nu + self.mu, (self.N,1)) * .5

                # Recurrent Weights
                Omega = - D.T.dot(D) - self.mu * np.eye(self.N)
                
        elif Dtyp=='random':
            if self.J==1:
                # random
                D = np.random.randn(self.N)

                # scaling
                D *= self.decoderscaling / float(self.N)

                # Threshold
                T = (np.diag(np.outer(D, D)) + self.nu + self.mu) * .5

                # Recurrent Weights
                Omega = - np.outer(D, D) - self.mu * np.eye(self.N)
            else:
                # random
                D = np.random.randn(self.N*2).reshape(2, self.N)

                # scaling
                D *= self.decoderscaling / float(self.N)

                # Threshold
                T = np.reshape(np.sum(D ** 2,0) + self.nu + self.mu, (self.N,1)) * .5

                # Recurrent Weights
                Omega = - D.T.dot(D) - self.mu * np.eye(self.N)
                
        elif Dtyp=='homo':
            if self.J==1:
                # homogeneous
                D = np.zeros(self.N) + 1

                # scaling
                D *= self.decoderscaling / float(self.N)

                # Threshold
                T = (np.diag(np.outer(D, D)) + self.nu + self.mu) * .5

                # Recurrent Weights
                Omega = - np.outer(D, D) - self.mu * np.eye(self.N)
            else:
                 # homogeneous
                D = np.zeros(self.N*2).reshape(2, self.N) + 1

                # scaling
                D *= self.decoderscaling / float(self.N)

                # Threshold
                T = np.reshape(np.sum(D ** 2,0) + self.nu + self.mu, (self.N,1)) * .5

                # Recurrent Weights
                Omega = - D.T.dot(D) - self.mu * np.eye(self.N)
                
        else:
            if self.J==1:
                D = Dtyp
                # scaling
                D *= self.decoderscaling / float(self.N)
                # Threshold
                T = (np.diag(np.outer(D, D)) + self.nu + self.mu) * .5

                # Recurrent Weights
                Omega = - np.outer(D, D) - self.mu * np.eye(self.N)
            else:
                D = Dtyp
                # Threshold
                T = np.reshape(np.sum(D ** 2,0) + self.nu + self.mu, (self.N,1)) * .5

                # Recurrent Weights
                Omega = - D.T.dot(D) - self.mu * np.eye(self.N)
        return Omega, T, D

    def mixednet(self, c, D, Omega, T, seed, thresh_nois_sig=0):
        np.random.seed(seed)
        O = np.zeros((self.N,self.n_step))               # Spike Train
        V = np.zeros((self.N,self.n_step))               # Membrane Potential
        R = np.zeros((self.N,self.n_step))               # Firing Rate
        
        if self.J==1:
            for t in range(self.n_step - 1):
                V[:, t + 1] = (1 - self.dt * self.lbd_v) * V[:, t]+\
                              self.dt * np.dot(D.T, c[0][t])+\
                              np.dot(Omega, O[:, t])
                test = V[:,t + 1] - np.ravel(T) +np.random.randn(V.shape[0])*thresh_nois_sig
                if np.max(test) >= 0:
                    spiking_neuron = np.random.choice(np.where(test==test.max())[0])
                    O[spiking_neuron, t + 1] = 1

                R[:,t + 1] = (1 - self.dt * self.lbd_d) * R[:, t] + O[:,t] # lbd_ readout decay rate
            
        else:
            for t in range(self.n_step - 1):        
                V[:, t + 1] = (1 - self.dt * self.lbd_v) * V[:, t] \
                                + self.dt * np.dot(D.T, c[:, t]) \
                                + np.dot(Omega, O[:, t])
                test = V[:,t + 1] - np.ravel(T) +np.random.randn(V.shape[0])*thresh_nois_sig
                if np.max(test) >= 0:
                    spiking_neuron = np.random.choice(np.where(test==test.max())[0])
                    O[spiking_neuron, t + 1] = 1

                R[:,t + 1] = (1 - self.dt * self.lbd_d) * R[:, t] + O[:,t] # lbd_ readout decay rate
            
        X_hat = np.dot(D,R)
        return O, V, R, X_hat

    def MA(self, xsig, win): # , scalsig=1):
        ma_z = np.copy(xsig)
        for ii in range(win, len(ma_z)-win):
            ma_z[ii] = np.mean(ma_z[ii-win:ii+win])
        ma_z[0:win] = np.mean(ma_z[0:win])
        ma_z[len(ma_z)-win:len(ma_z)] = np.mean(ma_z[len(ma_z)-win:len(ma_z)])
        # ma_z = ma_z/np.max(np.abs(ma_z)) * scalsig
        return ma_z

    def PlotDecoding(self, X, X_hat, O, D, win):
    
        figdec = plt.figure(figsize=(18,4))
        ax_dec = plt.subplot2grid((1,3), (0,0), rowspan=1, colspan=1)
        ax_sig = plt.subplot2grid((1,3), (0,1), rowspan=1, colspan=1)
        ax_spi = plt.subplot2grid((1,3), (0,2), rowspan=1, colspan=1)

        ax_dec.plot(D, 'x-')
        ax_dec.set_title('decoder')
        ax_dec.set_xlabel('neuron')
        ax_dec.set_ylabel('readout')
        
        smo_X_hat = 1
        
        if self.J==1:
            ax_sig.plot(self.time,X[0,:],'r', label='signal') # , label = u'$signal \, trajectory$')
            ax_sig.plot(self.time,X_hat,'--b', label='decoded')
            ax_sig.set_title('decoding')
            ax_sig.legend()
            ax_sig.set_xlabel('time')
            # ax_sig.set_ylabel(u'$X$')
            # plt.axis('equal')
            # sns.despine(offset=True)
            # binned version
            smo_X_hat = self.MA(X_hat, win)
            ax_sig.plot(self.time[:np.shape(X)[1]],smo_X_hat,'y')
        else:
            ax_sig.plot(X_hat[0,:],X_hat[1,:],'b')
            ax_sig.plot(X[0,:],X[1,:], 'r')
            ax_sig.set_title('encoded signal (true=red)')
            ax_sig.set_xlabel('1st latent dimension')
            ax_sig.set_ylabel('2nd latent dimension')
            ax_sig.axis('equal')
        time = self.time[:np.shape(X)[1]]
        for ii in range(self.N):
            ax_spi.plot(time[O[ii,:]>0], O[ii, O[ii,:]>0]*(ii+1)+1, '.')
            #ax_spi.plot(self.time[:np.shape(X)[1]], O[ii,:]*(ii+1), '.', label='neuron'+np.str(ii))
        ax_spi.set_title('spiking')
        ax_spi.set_xlabel('time')
        ax_spi.set_ylabel('neuron')
        plt.show()
        # plt.plot(X[0,:], 'r')
        # plt.plot(c[0,:], 'g')
        # plt.show()
        return smo_X_hat


def rasterplot(spikes, ax, tim):
    # spikes is a neuron by time matrix with zeros and ones

    tim = np.repeat(tim, 2)
    for nn in range(spikes.shape[0]):
        tmp = np.zeros(spikes.shape[1] * 2) * np.nan
        ind = np.arange(0, spikes.shape[1] * 2, 2)
        tmp[ind[spikes[nn, :] == 1]] = nn
        tmp[ind[spikes[nn, :] == 1] + 1] = nn + 1
        ax.plot(tim, tmp, '-', color=plt.cm.coolwarm(np.abs(nn-spikes.shape[0]/2)/(spikes.shape[0]/2)));
    ax.set_ylim([0, spikes.shape[0]+1])
    ax.set_xlim([np.min(tim), np.max(tim)])

def cart_to_pol(tmp):
    # tmp is a 2 column matrix with cartesian coordiantes
    r = np.sqrt(np.sum(tmp**2, axis=1))
    ttmp = tmp[:,0]
    ttmp[ttmp==0]=.0000000000000000001
    th = np.zeros(tmp.shape[0])*np.nan
    for nn in range(tmp.shape[0]):
        th[nn] = math.atan(tmp[nn,1]/ttmp[nn])
    r = r*np.sign(tmp[:,0])
    return r, th

def vis_fit_AC(MOD, estC, estA, kf, otherC=None, othername='other',
               otherC2=None, othername2='other'):
    # use MOD for teh real parameters
    r, th = cart_to_pol(MOD.C)
    rest, thest = cart_to_pol(estC)
    rkf, thkf = cart_to_pol(kf.observation_matrices)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(r * np.cos(th), r * np.sin(th), '-x', label='real')
    ax[0].plot(rest * np.cos(thest), rest * np.sin(thest), '-x', label='PLDS')
    ax[0].plot(rkf * np.cos(thkf), rkf * np.sin(thkf), '-x', label='KF')
    if otherC is not None:
        rest, thest = cart_to_pol(otherC)
        ax[0].plot(rest * np.cos(thest), rest * np.sin(thest), '-x', label=othername)
    if otherC2 is not None:
        rest, thest = cart_to_pol(otherC2)
        ax[0].plot(rest * np.cos(thest), rest * np.sin(thest), '-x', label=othername2)
    ax[0].legend()
    ax[0].set_aspect(1)

    evals, evec = np.linalg.eig(MOD.A)
    _, th = cart_to_pol(evec)
    evalsest, evecest = np.linalg.eig(estA)
    _, thest = cart_to_pol(evecest)
    evalsKF, evecKF = np.linalg.eig(kf.transition_matrices)
    _, thKF = cart_to_pol(evecKF)

    ax[1].plot(np.cos(np.arange(0, 2 * np.pi, .001)), np.sin(np.arange(0, 2 * np.pi, .001)), '--k')
    ax[1].plot(evals * np.cos(th), evals * np.sin(th), 'o', label='real')
    ax[1].plot(evalsest * np.cos(thest), evalsest * np.sin(thest), 'o', label='PLDS')
    ax[1].plot(evalsKF * np.cos(thKF), evalsKF * np.sin(thKF), 'o', label='KF')
    ax[1].legend()
    ax[1].set_aspect(1)

    ax[0].set_title('mapping C')
    ax[1].set_title('transition A')


def get_predict(loading, data, n_var=None, indLNO=None):
    mask = np.ones(data.shape[1], dtype='bool')
    if indLNO is not None:
        mask[indLNO] = False
    if n_var is None:
        n_var = np.zeros(data.shape[1])
    try:
        beta = loading[mask,:].T.dot(np.linalg.inv(loading[mask,:].dot(loading[mask,:].T)+np.diag(n_var[mask])))
    except np.linalg.LinAlgError:
        beta = loading[mask, :].T.dot(np.linalg.inv(loading[mask, :].dot(loading[mask, :].T) + np.diag(n_var[mask]+.1*np.random.randn(np.sum(mask)))))
    factor = beta.dot(data[:,mask].T)
    if indLNO is None:
        indLNO = mask
    pred = loading[indLNO,:].dot(factor).T
    return pred