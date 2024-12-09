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
import tdlm

class TestTDLM(unittest.TestCase):

    # def test_simple_sequence(self):
    #     # create a very simple fake probability vector with ones and zeros
    #     tf = np.roll(np.eye(5), 1, axis=1)
    #     sfreq = 100
    #     length = 60
    #     proba = np.random.rand(sfreq*length)
    #     # induce time lag of 3 time steps
    #     for lag in [1, 2, 3, 4, 5]:
    #         probas = np.zeros([proba.shape[0], 5])
    #         for i in range(5):
    #             probas[:, i] = np.roll(proba, i*lag)

    #         sf, sb = tdlm.compute_1step(probas, tf, n_shuf=10)
    #         fig, ax = plt.subplots(figsize=[8, 6])
    #         plotting.plot_sequenceness(sf, sb, which =['fwd', 'bkw'], ax=ax)

    #         #  check that plotted lines have the peak at the same point
    #         data = ax.lines[0].get_ydata()
    #         sb = np.nan_to_num(sb, -np.inf)
    #         assert np.argmax(data) == np.argmax(sb[0])

    #         data = ax.lines[1].get_ydata()
    #         sf = np.nan_to_num(sf, -np.inf)
    #         assert np.argmax(data) == np.argmax(sf[0]) == lag
    #         plt.close(fig)


    def test_alpha_supression(self):
        pass
if True:
        # create a very simple fake probability vector with ones and zeros
        tf = np.roll(np.eye(5), 1, axis=1)
        sfreq = 100
        length = 60
        proba = np.random.rand(sfreq*length)
        # induce time lag of 3 time steps
        lag = 10
        probas = np.zeros([proba.shape[0], 5])
        for i in range(5):
            probas[:, i] = np.roll(proba, i*lag)

        # sf, sb = tdlm.compute_1step(probas, tf, n_shuf=10)
        # sf_a, sb_a = tdlm.compute_1step(probas, tf, n_shuf=10, alpha_freq=10)
        sf_a2, sb_a2 = tdlm.compute_1step(probas, tf, n_shuf=10, alpha_freq=9)


        fig, ax = plt.subplots(figsize=[8, 6])
        plotting.plot_sequenceness(sf, sb, which =['fwd'], ax=ax)
        plotting.plot_sequenceness(sf_a, sb_a, which =['fwd'], ax=ax, color='red', clear=False)
        plotting.plot_sequenceness(sf_a2, sb_a2, which =['fwd'], ax=ax, color='green', clear=False)

if __name__=='__main__':
    unittest.main()
