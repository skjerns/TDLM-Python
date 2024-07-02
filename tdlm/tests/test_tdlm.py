# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 10:14:05 2024

@author: simon.kern
"""

import os
import unittest
import numpy as np
# from scipy import io
# from tqdm import tqdm
import tdlm
# import matplotlib.pyplot as plt
import mat73



class TestMatlab(unittest.TestCase):
    def test_cross_correlation_matlab(self):
        # load some sample data
        basedir = os.path.abspath(os.path.dirname(__file__))
        preds = mat73.loadmat(f'{basedir}/data/preds.mat')['preds']
        # X = io.loadmat(f'{basedir}/X.mat')
        # workspace = io.loadmat(f'{basedir}/workspace.mat')
    
        seq = 'ABCDE'
        tf = tdlm.utils.seq2tf(seq, n_states=8)
        tb = tf.T
        np.random.seed(0)
        seq_fwd, seq_bkw  = tdlm.compute_1step(preds, tf, tb=tb, n_shuf=100, max_lag=50, 
                                 cross_corr=True)
        
        peak =  np.nanargmax(seq_fwd[0])
        
        assert peak==4  # at 40 ms
        assert seq_fwd[0, peak] > np.nanmax(seq_fwd[1:])  # fwd is higher
        assert seq_bkw[0, peak] < np.nanmax(seq_bkw[1:])  # bkw is lower
    
        tdlm.plotting.plot_sequenceness(seq_fwd, seq_bkw, rescale=False)
        # tdlm.plotting.plot_sequenceness(seq_fwd_corr, seq_bkw_corr, rescale=False)

if __name__=='__main__':
    unittest.main()
    
    
    