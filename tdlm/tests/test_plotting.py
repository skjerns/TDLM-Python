# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 12:04:06 2024

@author: Simon
"""

# -*- coding: utf-8 -*-

import os
import unittest
import numpy as np
import scipy
from scipy import io
from tqdm import tqdm
from tdlm import plotting
import matplotlib.pyplot as plt


class TestPlotting(unittest.TestCase):

    def test_uperms(self):

        # dummy curve, should be below threshold
        seq_fwd = (np.arange(30*3) + 5).reshape([1, 3, 30]).astype(float)
        seq_fwd[:, :, 0] = np.nan
        plotting.plot_sequenceness(seq_fwd, None, which = ['fwd'])

        # dummy curve, should be above threshold
        seq_fwd = (np.arange(30*3) + 5)[::-1].reshape([1, 3, 30]).astype(float)
        seq_fwd[:, :, 0] = np.nan
        plotting.plot_sequenceness(seq_fwd, None, which = ['fwd'])

if __name__=='__main__':
    unittest.main()
