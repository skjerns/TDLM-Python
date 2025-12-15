# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 14:01:06 2024

@author: simon.kern
"""
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from tdlm.core import signflit_test


def plot_sequenceness(seq_fwd, seq_bkw, sfreq=100, ax=None, title=None,
                      color=None, which=['fwd-bkw', 'fwd', 'bkw'], clear=True,
                      plotsignflip=False, plotmax=True, plot95=False,
                      min_lag=0, max_lag=None, rescale=True, despine=True,
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
    min_lag: int, optional
        Minimum time lag that has been analysed. The default is 0, which
        corresponds to the first entry of the sequenceness array specifying
        the 0-time lag (often NaN)
    max_lag: int, optional
        Maximum time lag that has been looked at, e.g. 300 for timelags up to
        300 ms. The default is None, in which case a sample frequency of
        100 Hz is assumed and the length is taken from the shape of the
        seq_fwd and seq_bkw

    plotmax : bool, optional
        Plot the (old) significance intervals as defined in Liu et al (2020),
        which is the maximum across all time lags. The default is True.
    plotsignflip : bool, optional
            Plot the (new) signflip permutation threshold for p<0.05. This new
            threshold is recommended over the old maximum, as it is more valid
            to compare random effects vs fixed effects. Per default, 1000 perms
            are being run, which can be set by settings the parameter as int.
            The default is True.
    plot95 : bool, optional
        Additionally to the maximum across shuffles, plot the 95% of shuffle
        maximas. The default is False.
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

    assert sf.shape==sb.shape, f'{sf.shape=} must be {sb.shape=} but is not'

    sf = np.nan_to_num(sf)
    sb = np.nan_to_num(sb)

    if max_lag is None:
        max_lag = sf.shape[-1]*10 - min_lag - 10

    times = np.linspace(min_lag, max_lag, sf.shape[-1]).astype(int)

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
    sxs = [sf-sb, sf, sb]
    for i, direction in enumerate(['fwd-bkw', 'fwd', 'bkw']):
        if direction not in which:
            continue
        sx = sxs[i]
        print(direction)
        c = palette[i] if color is None else color
        perm_maxes = np.max(abs(np.mean(sx[:,1:,1:], 0)), -1)
        thresh_max = max(perm_maxes);

        div = thresh_max if rescale else 1

        dtp = (sx[:,0,:])/div;
        shadedErrorBar(times, dtp.mean(0), np.std(dtp, 0)/np.sqrt(len(sx)), ax=ax, color=c, label=direction)

        if plotmax:
            thresh_max = 1 if rescale else thresh_max
            ax.hlines(-thresh_max, times[0], times[-1], linestyles='--', color=c, linewidth=1.5, alpha=0.6, label='_')
            ax.hlines(thresh_max, times[0], times[-1], linestyles='--', color=c, linewidth=1.5, alpha=0.6, label='_')
        if plotsignflip:
            p, _, _, thresholds = signflit_test(sx[:, 0, :], n_perms=10000)
            thresh_signflip = thresholds/div
            ax.plot(times, -thresh_signflip, linestyle='--', color=c, linewidth=1, alpha=0.6, label='_')
            ax.plot(times, thresh_signflip, linestyle='--', color=c, linewidth=1, alpha=0.6, label='_')
        if plot95:
            thresh_95 = np.quantile(perm_maxes, 0.95)/div
            ax.hlines(thresh_95, times[0], times[-1], linestyles='-.', color=c, linewidth=1, alpha=0.4, label='_')
            ax.hlines(-thresh_95, times[0], times[-1], linestyles='-.', color=c, linewidth=1, alpha=0.4, label='_')

    # just for legend creation, not visible
    if plotmax:
        ax.plot([0, 0.000001], [0, .000000001], color='gray', linestyle='--', linewidth=1.5, label='perm. max')
    if plotsignflip:
        ax.plot([0, 0.000001], [0, .000000001], color='gray', linestyle=':', linewidth=1.5, label='signflip<0.05')
    if plot95:
        ax.plot([0, 0.000001], [0, .000000001], color='gray', linestyle='-.', linewidth=1, label='95% perm.')

    ax.legend(loc='upper right', fontsize=12)
    ax.set_xlabel('lag (ms)')
    ax.set_ylabel('sequenceness')

    # only set ylim congruent for rescaled graphs
    if rescale:
        ax.set_ylim(-1.5, 1.5)

    if title is not None:  ax.set_title(title)
    # ax.set_xticks(times[::5], minor=True
    ax.grid(axis='x', linewidth=1, which='both', alpha=0.3)
    fig.tight_layout()
    return ax


def plot_tval_distribution(t_true, t_maxes, bins=100,
                           title='tvalue distribution', ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    ax = sns.histplot(t_maxes, bins=bins, ax=ax, stat="percent")

    # Highlight the bin with red color
    p = (t_true<t_maxes).mean()
    p05 = np.quantile(t_maxes, 0.95)
    p001 = np.quantile(t_maxes, 0.999)
    ylims = ax.get_ylim()
    ax.vlines(t_true, *ylims, label=f"observed\n{p=:.5f}", color='red')
    ax.vlines(p05, *ylims, label='p=0.05', linestyle='--', color='black')
    ax.vlines(p001, *ylims, label='p=0.001', linestyle=':', color='black')
    # Add labels and title
    ax.set_title(title)
    ax.set_xlabel("tvalue distribution")
    ax.set_ylabel("Percentage")
    # ax.text(ax.get_xlim()[1]*0.97, ax.get_ylim()[1]*0.95, f'{p=:.3f}', horizontalalignment='right')
    ax.legend( fontsize=12, loc="upper left")

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
