# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 10:14:05 2024

@author: simon.kern
"""

import os
import unittest
import numpy as np
from scipy import io
from tqdm import tqdm
import tdlm
import matplotlib.pyplot as plt
import mat73


if __name__=='__main__':
    basedir = 'C:/Users/simon.kern/Nextcloud/ZI/2020.1 Pilotstudie/Experiment/analysis/TDLM'
    preds = mat73.loadmat(f'{basedir}/preds.mat')['preds']
    X = io.loadmat(f'{basedir}/X.mat')
    workspace = io.loadmat(f'{basedir}/workspace.mat')

    seq = 'ABCDE'
    tf = tdlm.utils.seq2tf(seq, n_states=8)
    tb = tf.T
    np.random.seed(0)
    res = tdlm.compute_1step(preds, tf, tb=tb, n_shuf=100, max_lag=50, 
                             cross_corr=True)
    seq_fwd, seq_bkw, seq_fwd_corr, seq_bkw_corr = res
    tdlm.plotting.plot_sequenceness(seq_fwd, seq_bkw, rescale=False)
    tdlm.plotting.plot_sequenceness(seq_fwd_corr, seq_bkw_corr, rescale=False)
