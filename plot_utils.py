import numpy as np
import matplotlib.pyplot as plt


def plotNetwork_inputOutput(O, ma_x, X_hat, tmax, xdim, tt=0, ix=None):
    spksT = np.arange(O.shape[1])
    spks = [spksT[O[i, :]>0] for i in range(O.shape[0])]

    fig = plt.figure(facecolor='w', figsize=(12, 3.5))
    ax = fig.add_axes([0, 0, 0.2, 1])
    plt.plot(ma_x[0,:,tt], ma_x[1,:,tt], lw=3)
    plt.plot(X_hat[0, :],  X_hat[1, :], alpha=.7)
    plt.legend(['signal', 'estimate', 'input']); plt.xlabel('LD-1'); plt.ylabel('LD-2');

    ax = fig.add_axes([0.26, 0, 0.5, 1])
    if ix is not None: 
        plt.plot(np.zeros(ix.shape)-300, ix, 'or', markersize=3)
    plt.eventplot(spks, colors='k');
    plt.xticks(np.linspace(0, len(spksT), 5), [str(i) for i in np.linspace(0, tmax, 5)])
    plt.xlabel('Time (sec)'); plt.ylabel('Neuron #')

    ax = fig.add_axes([0.81, 0, 0.2, 1])
    plt.plot(np.sum(O, axis=1)/tmax , range(O.shape[0]), c='k');
    plt.xlabel('Firing rate (Hz)'); plt.ylabel('Neuron #')
    plt.show()
    
def plotLSMtrajectories(sub_x, oZ_PCA, oZ_FA, oZ_LDS, oZ_GPFA, mask, cc_off=0, legnd = []):
    csZ_subX  = []
    csZ_PCA  = []
    csZ_FA   = []
    csZ_LDS  = []
    csZ_GPFA = []

    csZ_subX.append(sub_x[:,   :, mask])
    csZ_PCA.append(oZ_PCA[:,   :, mask])
    csZ_FA.append(oZ_FA[:,     :, mask])
    csZ_LDS.append(oZ_LDS[:,   :, mask])
    csZ_GPFA.append(oZ_GPFA[:, :, mask])

    csZ_subX.append(sub_x[:,   :, ~mask])
    csZ_PCA.append(oZ_PCA[:,   :, ~mask])
    csZ_FA.append(oZ_FA[:,     :, ~mask])
    csZ_LDS.append(oZ_LDS[:,   :, ~mask])
    csZ_GPFA.append(oZ_GPFA[:, :, ~mask])
    
    cmap = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf']
    fig, ax = plt.subplots(1, 5, figsize=(17,4), facecolor='w')
    trial_idxs = np.random.choice(8, 1, replace=False)

    '''for nn in range(2):
        ax[0].plot(np.mean(csZ_subX[nn][0,:,:],axis=1),np.mean(csZ_subX[nn][1,:,:],axis=1),'-', c=cmap[nn+cc_off], lw=4)
        ax[1].plot(np.mean(csZ_PCA[nn][0,:,:],axis=1), np.mean(csZ_PCA[nn][1,:,:],axis=1), '-', c=cmap[nn+cc_off], lw=4)
        ax[2].plot(np.mean(csZ_FA[nn][0,:,:],axis=1),  np.mean(csZ_FA[nn][1,:,:],axis=1),  '-', c=cmap[nn+cc_off], lw=4)
        ax[3].plot(np.mean(csZ_LDS[nn][0,:,:],axis=1), np.mean(csZ_LDS[nn][1,:,:],axis=1), '-', c=cmap[nn+cc_off], lw=4)
        ax[4].plot(np.mean(csZ_GPFA[nn][0,:,:],axis=1),np.mean(csZ_GPFA[nn][1,:,:],axis=1),'-', c=cmap[nn+cc_off], lw=4)
    '''
    for nn in range(2):
        for mm in trial_idxs:
            ax[0].plot(csZ_subX[nn][0,:,mm],csZ_subX[nn][1,:,mm],'--', c=cmap[nn+cc_off], lw=1)
            ax[1].plot(csZ_PCA[nn][0,:,mm], csZ_PCA[nn][1,:,mm], '--', c=cmap[nn+cc_off], lw=1)
            ax[2].plot(csZ_FA[nn][0,:,mm],  csZ_FA[nn][1,:,mm],  '--', c=cmap[nn+cc_off], lw=1)
            ax[3].plot(csZ_LDS[nn][0,:,mm], csZ_LDS[nn][1,:,mm], '--', c=cmap[nn+cc_off], lw=1)
            ax[4].plot(csZ_GPFA[nn][0,:,mm],csZ_GPFA[nn][1,:,mm],'--', c=cmap[nn+cc_off], lw=1)

    ax[0].set_title('Latent dynamics')
    ax[1].set_title('PCA')
    ax[2].set_title('FA')
    ax[3].set_title('LDS')
    ax[4].set_title('GPFA')

    for ax_idx in range(5):
    #     ax[ax_idx].set_aspect('equal')
        ax[ax_idx].set_xticks([])
        ax[ax_idx].set_yticks([])

    # for dim in range(n_dim_state):
    #     for stim_toPlt in range(len(conditions)):
    #         ax[dim].plot(x, Z_KF[mouse_IDs[m]][stim_toPlt, trial_idxs, :, dim].T, '--', c=cmap[stim_toPlt], alpha=.4)

    ax[0].set_xlabel('LD-1')
    ax[0].set_ylabel('LD-2');
    ax[0].legend(legnd);

    
    
    
    
    