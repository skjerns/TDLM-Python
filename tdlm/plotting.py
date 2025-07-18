# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 14:01:06 2024

@author: simon.kern
"""
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)


def plot_sequenceness(seq_fwd, seq_bkw, *, sfreq=100,
                      ax=None, title=None, color=None,
                      which=('fwd-bkw', 'fwd', 'bkw'),
                      clear=True, plotmax=True, plot95=True,
                      rescale=True, despine=True, **kwargs):
    """
    Plot forward, backward and differential sequenceness with confidence bands.

    Parameters
    ----------
    sfreq : float | int, optional
        Sampling frequency in *Hz* (samples / second). Determines the
        millisecond spacing on the x-axis.  Default = 100 Hz.
    """
    def shaded_error(ax, x, y, err, **kwargs):
        """Helper that draws a mean line plus a symmetric shaded error band."""
        ax.plot(x, y, **kwargs)
        ax.fill_between(x, y - err, y + err, alpha=0.35,
                        label='_nolegend_', **{k: v for k, v in kwargs.items()
                                               if k != 'label'})
    # ----------- preliminaries ------------------------------------------------
    sns.despine(ax=ax) if despine else None
    if ax is None:
        _, ax = plt.subplots()
    if clear:
        ax.clear()

    # bring data to shape (n_subj, lags, n_perm)
    sf, sb = map(np.asarray, (seq_fwd, seq_bkw))
    if sf.ndim == 2:
        sf = sf[None]
    if sb.ndim == 2:
        sb = sb[None]
    sf, sb = map(np.nan_to_num, (sf, sb))

    # time axis in **milliseconds**
    step_ms = 1000. / sfreq                 # 1 sample → this many ms
    times   = np.arange(sf.shape[-1]) * step_ms

    # colour palette
    pal = sns.color_palette()
    clr = dict(fwd=pal[1] if color is None else color,
               bkw=pal[2] if color is None else color,
               diff=pal[0] if color is None else color)

    # ----------- plotting helpers --------------------------------------------
    def _plot(d, thresh, label, c):
        div   = thresh if rescale else 1.
        band  = np.quantile(thresh, .95) / div
        curve = d[:, 0, :] / div
        shaded_error(ax, times, curve.mean(0),
                     curve.std(0) / np.sqrt(len(curve)),
                     color=c, label=label, **kwargs)
        if plotmax:
            for s in (-1, 1):
                ax.hlines(s * (thresh if rescale else 1.),
                          times[0], times[-1],
                          linestyles='--', color=c, alpha=.6, linewidth=1.5)
        if plot95:
            for s in (-1, 1):
                ax.hlines(s * band,
                          times[0], times[-1],
                          linestyles='--', color=c, alpha=.4, linewidth=1.)

    # ----------- compute & draw ----------------------------------------------
    if 'fwd-bkw' in which:
        diff  = sf - sb
        thr   = np.max(np.abs(np.mean(diff[:, 1:, 1:], axis=0)), axis=-1)
        _plot(diff, thr.max(), 'fwd-bkw', clr['diff'])

    if 'fwd' in which:
        thr = np.max(np.abs(np.mean(sf[:, 1:, 1:], axis=0)), axis=-1)
        _plot(sf, thr.max(), 'fwd', clr['fwd'])

    if 'bkw' in which:
        thr = np.max(np.abs(np.mean(sb[:, 1:, 1:], axis=0)), axis=-1)
        _plot(sb, thr.max(), 'bkw', clr['bkw'])

    # ----------- axis cosmetics ----------------------------------------------
    ax.set_xlabel('lag (ms)')
    ax.set_ylabel('sequenceness')
    ax.set_title(title or '')
    ax.legend(loc='upper right', frameon=False)

    # y-limits symmetrical around zero
    ylim = np.max(np.abs(ax.get_ylim()))
    ax.set_ylim(-ylim, ylim)

    # --- x-tick strategy -----------------------------------------------------
    # aim for ~10 labelled ticks irrespective of sfreq & series length
    n_major = 10
    tick_spacing = step_ms * max(1, (len(times) // n_major))
    ax.xaxis.set_major_locator(MultipleLocator(tick_spacing))
    ax.xaxis.set_minor_locator(AutoMinorLocator())

    ax.grid(axis='x', which='both', alpha=.3)
    plt.tight_layout()
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
