# created by Pedro Herrero-Vidal

import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.integrate import odeint

class firing_rate_network:
    # parameter initialization
    def __init__(self, a=270., b=108., d=.154, gamma=.64, tau_s=.1, tau0=.002,
                 g_E=.2609, g_I=-.0497, I_o=.3255, g_ext=.00052):
        self.a = a              # in Hz/nA
        self.b = b              # Hz
        self.d = d              # synaptic time constant in s
        self.gamma = gamma      # saturation factor for gating variable
        self.tau_s = tau_s      # in s
        self.tau0 = tau0        # in s
        self.g_E = g_E          # excitatory strength in nA
        self.g_I = g_I          # cross-inhibition strength in nA
        self.I_o = I_o          # background current in nA
        self.g_ext = g_ext      # stimulus input strength [nA/Hz]
        
    # compute network firing rate
    def rate(self, I):
        return (self.a*I - self.b) / (1. - np.exp(-self.d*(self.a*I - self.b)))
    
    # firing rate network model simulations using Euler approximation
    def simulation(self, I_1, I_2, I_12=0, I_22=0, t=2, 
                   t1=.2, t12=.5, t2=.2, t22 = 0,
                   onset_jitter = 0,
                   pulse_t1=0., pulse_t12=0., pulse_t2=0., pulse_t22=0.,
                   sigma_I1 =0, sigma_I2 =0,
                   s1=.1, s2=.1, sigma=.02, phi=1., n_trial=10, dt=0.5/1000, seed=1):
#         dt = 0.5/1000 #self.tau_s/10
        
        np.random.seed(seed)
        T = np.arange(0, t, dt)
        
        stim1 = []
        stim2 = []
        
        for jj in range(n_trial):
            on_jit = np.random.normal(0, onset_jitter)
            stim11 = (T>(t1+on_jit))   * (T<(t1+pulse_t1))   * I_1  * self.g_ext
            stim12 = (T>(t12+on_jit))  * (T<(t12+pulse_t12)) * I_12 * self.g_ext
            stim1sigma1 = (T>(t1+on_jit))  * (T<(t1+pulse_t1))   * np.random.normal(0, sigma_I1, len(T)) * self.g_ext
            stim1sigma2 = (T>(t12+on_jit)) * (T<(t12+pulse_t12)) * np.random.normal(0, sigma_I1, len(T)) * self.g_ext
            stim1.append(stim11 + stim12 + stim1sigma1 + stim1sigma2)
            stim21  = (T>(t2+on_jit))  * (T<(t2+pulse_t2))   * I_2  * self.g_ext
            stim22  = (T>(t22+on_jit)) * (T<(t22+pulse_t22)) * I_22 * self.g_ext
            stim2sigma1 = (T>(t2+on_jit))  * (T<(t2+pulse_t2))   * np.random.normal(0, sigma_I2, len(T)) * self.g_ext
            stim2sigma2 = (T>(t22+on_jit)) * (T<(t22+pulse_t22)) * np.random.normal(0, sigma_I2, len(T)) * self.g_ext
            stim2.append(stim21 + stim22 + stim2sigma1 + stim2sigma2)
        stim1 = np.array(stim1)
        stim2 = np.array(stim2) 

        S1 = []
        S2 = []
        R1 = []
        R2 = []
        s1 = 0.1*np.ones(n_trial)
        s2 = 0.1*np.ones(n_trial)
        
        Ieta1 = np.zeros(n_trial)
        Ieta2 = np.zeros(n_trial)
        
        for t in range(len(T)):
            Isyn1 = self.g_E*s1 + self.g_I*s2 + stim1[:,t] + Ieta1
            Isyn2 = self.g_E*s2 + self.g_I*s1 + stim2[:,t] + Ieta2
            
            r1 = self.rate(Isyn1)
            r2 = self.rate(Isyn2)
            
            s1_next = s1 + phi*(r1*self.gamma*(1 - s1) - (s1/self.tau_s))*dt
            s2_next = s2 + phi*(r2*self.gamma*(1 - s2) - (s2/self.tau_s))*dt
            
            Ieta1_next = Ieta1 + (dt/self.tau0)*(self.I_o-Ieta1) + np.sqrt(dt/self.tau0)*sigma*np.random.randn(n_trial)
            Ieta2_next = Ieta2 + (dt/self.tau0)*(self.I_o-Ieta2) + np.sqrt(dt/self.tau0)*sigma*np.random.randn(n_trial)
            
            s1 = s1_next
            s2 = s2_next
            Ieta1 = Ieta1_next
            Ieta2 = Ieta2_next

            S1.append(s1)
            S2.append(s2)
            R1.append(r1)
            R2.append(r2)
        return S1, S2, R1, R2, stim1[0,:], stim2[0,:], T
    
def plot_NetworkAct_Input(T, I1, I2, R1, R2, S1, S2):
    fig = plt.figure(figsize=(12,14), facecolor='w')
    ax = fig.add_axes([0, 0.35, 0.35, 0.1])
    plt.plot(T, I1, 'blue', lw=2)
    plt.plot(T, I2, 'red')
    plt.ylabel('input (nA)')
    
    plt.plot([], [], 'o', c='limegreen')
    plt.plot([], [], 'o', c='purple')
    plt.legend(['neural population 1', 'neural population 2', 'init. cond.', 'final cond.'], 
               bbox_to_anchor=(3, 1))

    ax = fig.add_axes([0, 0.1, 0.35, 0.2])
    plt.plot(T, R1, 'blue', alpha=.2)
    plt.plot(T, R2, 'red' , alpha=.2)
    plt.xlabel('time (s)')
    plt.ylabel('activity (Hz)')
    
    ax = fig.add_axes([0.45, 0.3, 0.35, 0.15])
    plt.plot(T, S1, 'blue', alpha=.2)
    plt.plot(T, S2, 'red' , alpha=.2)
    plt.xlabel('time (s)')
    plt.ylabel('synaptic drive')
    
    ax = fig.add_axes([0.45, 0.1, 0.15, 0.15])
    plt.plot(S1, S2, 'k', alpha=.2)
    plt.plot(S1[0], S2[0], 'o', c='limegreen')
    plt.plot(S1[-1], S2[-1], 'o', c='purple')
    plt.xlabel('synaptic drive pop. 1')
    plt.ylabel('synaptic drive pop. 2')
    plt.xlim((0, 0.7))
    plt.ylim((0, 0.7))
    
    ax = fig.add_axes([0.65, 0.1, 0.15, 0.15])
    plt.plot(R1, R2, 'k', alpha=.2)
    plt.plot(R1[0], R2[0], 'o', c='limegreen')
    plt.plot(R1[-1], R2[-1], 'o', c='purple')
    plt.xlabel('act. pop. 1 (Hz)')
    plt.ylabel('act. pop. 2 (Hz)')
    plt.xlim((-2, 45))
    plt.ylim((-2, 45))
    plt.show()