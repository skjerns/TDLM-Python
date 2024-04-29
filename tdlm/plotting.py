# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 14:01:06 2024

@author: simon.kern
"""
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

def plot_sequenceness(seq_fwd, seq_bkw, cTime=None, ax=None, title=None,
                      color=None, which=['fwd-bkw', 'fwd', 'bkw'], clear=True, 
                      rescale=True, plot95=True, plotmax=True, despine=True):


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

    if cTime is None:
        cTime = np.arange(0, sf.shape[-1]*10, 10) #just assume sampling frequency

    if ax is None:
        fig = plt.figure()
        ax = plt.gca()
        
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
        shadedErrorBar(cTime, dtp.mean(0), np.std(dtp, 0)/np.sqrt(len(sf)), ax=ax, color=c)

    # Now plot fwd
    if 'fwd' in which:
        c = palette[1] if color is None else color
        npThresh = np.max(abs(np.mean(sf[:,1:,1:],0)) , -1);
        npThreshMax = max(npThresh);
        div = npThreshMax if rescale else 1
        npThresh95 = np.quantile(npThresh, 0.95)/div
        dtp = sf[:,0,:]/div
        shadedErrorBar(cTime, dtp.mean(0), np.std(dtp, 0)/np.sqrt(len(sf)), ax=ax, color=c)

    # now plot bkw
    if 'bkw' in which:
        c = palette[2] if color is None else color
        npThresh = np.max(abs(np.mean(sb[:,1:,1:],0)) , -1);
        npThreshMax = max(npThresh);
        div = npThreshMax if rescale else 1
        npThresh95 = np.quantile(npThresh, 0.95)/div
        dtp = sb[:,0,:]/div
        shadedErrorBar(cTime, dtp.mean(0), np.std(dtp, 0)/np.sqrt(len(sb)), ax=ax, color=c)

    div = 1 if rescale else npThreshMax
    if plotmax:
        ax.hlines(-div, cTime[0], cTime[-1], linestyles='--', color='black', linewidth=1.5, alpha=0.6)
        ax.hlines(div, cTime[0], cTime[-1], linestyles='--', color='black', linewidth=1.5, alpha=0.6)
    if plot95:
        ax.hlines(npThresh95, cTime[0], cTime[-1], linestyles='--', color='black', linewidth=1, alpha=0.4)
        ax.hlines(-npThresh95, cTime[0], cTime[-1], linestyles='--', color='black', linewidth=1, alpha=0.4)
    ax.legend(which, loc='upper right')
    ax.set_xlabel('lag (ms)')
    ax.set_ylabel('sequenceness')
    ax.set_ylim(-div*1.5, div*1.5)
    if title is not None:  ax.set_title(title)
    ax.set_xticks(cTime[::5])
    ax.set_xticks(cTime[::5], minor=True)
    ax.grid(axis='x', linewidth=1, which='both', alpha=0.3)
    fig.tight_layout()
    return ax