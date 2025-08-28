# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 14:01:06 2024

@author: simon.kern
"""
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)


def plot_sequenceness(seq_fwd, seq_bkw, sfreq=100, ax=None, title=None,
                      color=None, which=['fwd-bkw', 'fwd', 'bkw'], clear=True,
                      plotmax=True, plot95=True, rescale=True, despine=True,
                      **kwargs):
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
        kwargs.update(dict(label='_nolegend_' ))
        ax.fill_between(x, y-err, y+err, alpha=0.35, **kwargs)

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
        shadedErrorBar(times, dtp.mean(0), np.std(dtp, 0)/np.sqrt(len(sf)), ax=ax, color=c, label='fwd-bkw')
        div = 1 if rescale else npThreshMax
        if plotmax:
            ax.hlines(-div, times[0], times[-1], linestyles='--', color=c, linewidth=1.5, alpha=0.6, label='_')
            ax.hlines(div, times[0], times[-1], linestyles='--', color=c, linewidth=1.5, alpha=0.6, label='_')
        if plot95:
            ax.hlines(npThresh95, times[0], times[-1], linestyles='--', color=c, linewidth=1, alpha=0.4, label='_')
            ax.hlines(-npThresh95, times[0], times[-1], linestyles='--', color=c, linewidth=1, alpha=0.4, label='_')

    # now plot bkw
    if 'bkw' in which:
        c = palette[2] if color is None else color
        npThresh = np.max(abs(np.mean(sb[:,1:,1:],0)) , -1);
        npThreshMax = max(npThresh);
        div = npThreshMax if rescale else 1
        npThresh95 = np.quantile(npThresh, 0.95)/div
        dtp = sb[:,0,:]/div
        shadedErrorBar(times, dtp.mean(0), np.std(dtp, 0)/np.sqrt(len(sb)), ax=ax, color=c, label='bkw')
        div = 1 if rescale else npThreshMax
        if plotmax:
            ax.hlines(-div, times[0], times[-1], linestyles='--', color=c, linewidth=1.5, alpha=0.6, label='_')
            ax.hlines(div, times[0], times[-1], linestyles='--', color=c, linewidth=1.5, alpha=0.6, label='_')
        if plot95:
            ax.hlines(npThresh95, times[0], times[-1], linestyles='--', color=c, linewidth=1, alpha=0.4, label='_')
            ax.hlines(-npThresh95, times[0], times[-1], linestyles='--', color=c, linewidth=1, alpha=0.4, label='_')


    # Now plot fwd
    if 'fwd' in which:
        c = palette[1] if color is None else color
        npThresh = np.max(abs(np.mean(sf[:,1:,1:],0)) , -1);
        npThreshMax = max(npThresh);
        div = npThreshMax if rescale else 1
        npThresh95 = np.quantile(npThresh, 0.95)/div
        dtp = sf[:,0,:]/div
        shadedErrorBar(times, dtp.mean(0), np.std(dtp, 0)/np.sqrt(len(sf)), ax=ax, color=c, label='fwd')


        div = 1 if rescale else npThreshMax
        if plotmax:
            ax.hlines(-div, times[0], times[-1], linestyles='--', color=c, linewidth=1.5, alpha=0.6, label='_')
            ax.hlines(div, times[0], times[-1], linestyles='--', color=c, linewidth=1.5, alpha=0.6, label='_')
        if plot95:
            ax.hlines(npThresh95, times[0], times[-1], linestyles='--', color=c, linewidth=1, alpha=0.4, label='_')
            ax.hlines(-npThresh95, times[0], times[-1], linestyles='--', color=c, linewidth=1, alpha=0.4, label='_')

    ax.legend(loc='upper right')
    ax.set_xlabel('lag (ms)')
    ax.set_ylabel('sequenceness')
    ax.set_ylim(-div*1.5, div*1.5)
    if title is not None:  ax.set_title(title)
    ax.set_xticks(times[::5])
    ax.set_xticks(times[::5], minor=True)
    ax.grid(axis='x', linewidth=1, which='both', alpha=0.3)
    fig.tight_layout()
    return ax




def plot_permutation_distribution(sx, ax=None, title=None, **kwargs):
    """plots the means of a TDLM permutation results.

    plots a histogram of mean permutation sequenceness values across the
    time lags. Indicates the base value (permutation index 0) in red.
    Calculates the p value.


    Parameters
    ----------
    sx : np.ndarray
        either forward sequenceness matrix or backward sequencenes matrix.
        usually of 3d shape (n_subj, n_perm, n_lags)
    ax : plt.Axis, optional
        which axis to plot into. The default is None.
    title : str, optional
        title for the distribution. The default is None.
    **kwargs :
        keyword arguments to forward to seaborn.histplot.

    Returns
    -------
    ax : plt.Axis
        the axis that has been plotted into.
    p : np.float
        p value of the permutation distribution test.
    """
    # Calculate mean across values for each subject
    mean_subjects = np.nanmean(sx, axis=(2,))

    # Calculate mean of means
    mean_of_means = np.nanmean(mean_subjects, axis=0)

    # Plot histogram
    bins = np.histogram_bin_edges(mean_of_means, bins=50)

    ax = sns.histplot(
        mean_of_means, bins=bins, alpha=kwargs.pop('alpha', 0.5), ax=ax,
        stat="count", **kwargs
    )

    # Find the bin that the first permutation falls into
    bin_index = np.searchsorted(bins, mean_of_means[0])

    # Highlight the bin with red color
    axmin = bins[bin_index]
    axmax = bins[bin_index + 1] if (bin_index+1)<len(bins) else axmin*1.02
    ax.axvspan(axmin, axmax, color="red", alpha=0.5)
    p = np.mean(np.abs(mean_of_means[0]) < np.abs(mean_of_means[1:]))

    # Add labels and title
    ax.set_xlabel("Mean sequenceness of permutation")
    ax.set_ylabel("Count")
    ax.set_title(title)
    # ax.text(ax.get_xlim()[1]*0.97, ax.get_ylim()[1]*0.95, f'{p=:.3f}', horizontalalignment='right')
    ax.legend([f"observed\n{p=:.3f}"], fontsize=12, loc="upper left")
    return ax, p
