import matplotlib.pyplot as plt

def myplot(rows, columns, heightratio=None, widthratio=None, figsize=None,
           removeticks=False,
           xticklabelsize=20, yticklabelsize=20, fontsize=14, titlesize=16, labelsize=14, xlabels=None, ylabels=None):
    plt.rc('xtick', labelsize=xticklabelsize)
    plt.rc('ytick', labelsize=yticklabelsize)
    plt.rc('font', size=fontsize)  # controls default text sizes
    plt.rc('axes', titlesize=titlesize)  # fontsize of the axes title
    plt.rc('axes', labelsize=labelsize)  # fontsize of the x and y labels
    if (heightratio is not None):
        fig, ax = plt.subplots(rows, columns,
                               gridspec_kw={'width_ratios': widthratio,
                                            'height_ratios': heightratio}, squeeze=False,
                               figsize=figsize)
    else:
        fig, ax = plt.subplots(rows, columns,squeeze=False,
                               figsize=figsize)
    if removeticks:
        for ii in range(len(ax)):
            for jj in range(len(ax[ii])):
                ax[ii][jj].set_xticks([], [])
                ax[ii][jj].set_yticks([], [])
    if xlabels is not None:
        for ii in range(len(ax)):
            for jj in range(len(ax[ii])):
                ax[ii][jj].set_xlabel(xlabels[ii][jj])
    if ylabels is not None:
        for ii in range(len(ax)):
            for jj in range(len(ax[ii])):
                ax[ii][jj].set_ylabel(ylabels[ii][jj])
    return fig, ax
