# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 14:01:06 2024

@author: simon.kern
"""
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

def plot_sequenceness(seq_fwd, seq_bkw, sfreq=100, ax=None, title=None,
                      color=None, which=['fwd-bkw', 'fwd', 'bkw'], clear=True, 
                      plotmax=True, plot95=True, rescale=True, despine=True):
    """Plot forward, backward and differential sequenceness with conf. interv.
    
    Given forward and backwards sequenceness permutation result matrices, plot
    curves. 2D matrices are interpreted as a single participant, 3D matrices
    as (participant, timelags, n_shuffles) with the first index of the shuffle
    being the non-shuffled version.

    Parameters
    ----------
    seq_fwd : np.ndarray
        DESCRIPTION.
    seq_bkw : np.ndarray
        DESCRIPTION.
    sfreq : int, optional
        Sample frequency of the data points. The default is 100.
    ax : matplotlib.pyplot.Axes, optional
        axis to put the plot into. Will create if None. The default is None.
    title : str, optional
        Title for the plot. The default is None.
    color : str, optional
        Color label to use for all plots. The default is None.
    which : list of str, optional
        Selects which plots to create fwd = forward sequenceness, 
        bkw = backward sequenceness, fwd-bkw = differential sequenceness. 
        The default is ['fwd-bkw', 'fwd', 'bkw'].
    clear : bool, optional
        Whether to clear the axis before plotting. The default is True.

    plotmax : bool, optional
        Plot the significance intervals as defined in Liu et al (2020), which
        is the maximum across all time lags. The default is True.
    plot95 : bool, optional
        Additionally to the maximum across shuffles, plot the 95% of shuffle 
        maximas. The default is True.
    rescale : bool, optional
        rescale both plots such that the significance intervals is at 1.
        Else intervals will be different for fwd and bkw. The default is True.      
    despine : bool, optional
        call sns.despine() after plotting. The default is True.

    Returns
    -------
    ax : TYPE
        DESCRIPTION.

    """

    # TODO: scaling with two axis?
    def shadedErrorBar(x, y, err, ax=None, **kwargs):
        ax.plot(x, y, **kwargs)
        ax.fill_between(x, y-err, y+err, alpha=0.35, label='_nolegend_' , **kwargs)

    sf = np.array(seq_fwd, copy=True)
    sb = np.array(seq_bkw, copy=True)
    if sf.ndim==2:
        sf = sf.reshape([1, *sf.shape])
    if sb.ndim==2:
        sb = sb.reshape([1, *sb.shape])

    sf = np.nan_to_num(sf)
    sb = np.nan_to_num(sb)

    factor = 1000/100
    times = np.arange(0, sf.shape[-1]*factor, factor) #just assume sampling frequency

    if ax is None:
        fig = plt.figure()
        ax = plt.gca()
    else:
        fig = ax.figure
        
    if despine:
        sns.despine()
        
    if clear:
        ax.clear()
        
    palette = sns.color_palette()

    # First plot fwd-bkw
    div = 1
    if 'fwd-bkw' in which:
        c = palette[0] if color is None else color
        npThresh = np.max(abs(np.mean(sf[:,1:,1:]-sb[:,1:,1:], 0)), -1)
        npThreshMax = max(npThresh);
        div = npThreshMax if rescale else 1
        npThresh95 = np.quantile(npThresh, 0.95)/div
        dtp = (sf[:,0,:]-sb[:,0,:])/div;
        shadedErrorBar(times, dtp.mean(0), np.std(dtp, 0)/np.sqrt(len(sf)), ax=ax, color=c)

    # Now plot fwd
    if 'fwd' in which:
        c = palette[1] if color is None else color
        npThresh = np.max(abs(np.mean(sf[:,1:,1:],0)) , -1);
        npThreshMax = max(npThresh);
        div = npThreshMax if rescale else 1
        npThresh95 = np.quantile(npThresh, 0.95)/div
        dtp = sf[:,0,:]/div
        shadedErrorBar(times, dtp.mean(0), np.std(dtp, 0)/np.sqrt(len(sf)), ax=ax, color=c)

    # now plot bkw
    if 'bkw' in which:
        c = palette[2] if color is None else color
        npThresh = np.max(abs(np.mean(sb[:,1:,1:],0)) , -1);
        npThreshMax = max(npThresh);
        div = npThreshMax if rescale else 1
        npThresh95 = np.quantile(npThresh, 0.95)/div
        dtp = sb[:,0,:]/div
        shadedErrorBar(times, dtp.mean(0), np.std(dtp, 0)/np.sqrt(len(sb)), ax=ax, color=c)

    div = 1 if rescale else npThreshMax
    if plotmax:
        ax.hlines(-div, times[0], times[-1], linestyles='--', color='black', linewidth=1.5, alpha=0.6)
        ax.hlines(div, times[0], times[-1], linestyles='--', color='black', linewidth=1.5, alpha=0.6)
    if plot95:
        ax.hlines(npThresh95, times[0], times[-1], linestyles='--', color='black', linewidth=1, alpha=0.4)
        ax.hlines(-npThresh95, times[0], times[-1], linestyles='--', color='black', linewidth=1, alpha=0.4)
    ax.legend(which, loc='upper right')
    ax.set_xlabel('lag (ms)')
    ax.set_ylabel('sequenceness')
    ax.set_ylim(-div*1.5, div*1.5)
    if title is not None:  ax.set_title(title)
    ax.set_xticks(times[::5])
    ax.set_xticks(times[::5], minor=True)
    ax.grid(axis='x', linewidth=1, which='both', alpha=0.3)
    fig.tight_layout()
    return ax

