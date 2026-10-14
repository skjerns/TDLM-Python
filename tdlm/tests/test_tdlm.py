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

    def test_simple_sequence_plotting(self):
        """create a time lagged probability series by shifting array rows

        check if the plot is correct"""
        # create a very simple fake probability vector with ones and zeros
        tf = np.roll(np.eye(5), 1, axis=1)
        sfreq = 100
        length = 60
        proba = np.random.rand(sfreq*length)
        # induce time lag of 3 time steps
        for lag in [2, 3, 4, 5]:
            probas = np.zeros([proba.shape[0], 5])
            for i in range(5):
                probas[:, i] = np.roll(proba, i*lag)

            sf, sb = tdlm.compute_1step(probas, tf, n_shuf=10)
            fig, ax = plt.subplots(figsize=[8, 6])
            plotting.plot_sequenceness(sf, sb, which =['fwd', 'bkw'], ax=ax)

            # check that plotted lines have the peak where the data has it
            data = ax.lines[0].get_ydata()
            sf = np.nan_to_num(sf, -np.inf)
            assert np.argmax(data) == np.argmax(sf[0]) == lag

            data = ax.lines[1].get_ydata()
            sb = np.nan_to_num(sb, -np.inf)
            assert np.argmax(data)== np.argmax(sb[0])
            plt.close(fig)


    def test_simple_2step_sequence_plotting(self):
        # create a very simple fake probability vector with ones and zeros
        tf = np.roll(np.eye(5), 1, axis=1)
        sfreq = 100
        length = 60
        proba = np.random.rand(sfreq*length)
        # induce time lag of 3 time steps
        for lag in [1, 2, 3, 4, 5]:
            probas = np.zeros([proba.shape[0], 5])
            for i in range(5):
                probas[:, i] = np.roll(proba, i*lag)

            sf, sb = tdlm.compute_2step(probas, tf, n_shuf=10)
            fig, ax = plt.subplots(figsize=[8, 6])
            plotting.plot_sequenceness(sf, sb, which =['fwd', 'bkw'], ax=ax)

            #  check that plotted lines have the peak at the same point
            data = ax.lines[0].get_ydata()
            sf = np.nan_to_num(sf, -np.inf)
            assert np.argmax(data) == np.argmax(sf[0]) == lag
            plt.close(fig)


            data = ax.lines[1].get_ydata()
            sb = np.nan_to_num(sb, -np.inf)
            assert np.argmax(data) == np.argmax(sb[0])

    def test_2step_sequence_exclusive(self):
        """sanity check that no 3-step are detected if they are not present"""

        # create a very simple fake probability vector with ones and zeros,
        # where we only have two step
        tf = np.roll(np.eye(4), 1, axis=1)
        tf[3, 0] = 0
        sfreq = 100
        length = 60
        proba = np.random.rand(sfreq*length)
        # induce time lag of 3 time steps
        lag = 4
        # should only contain A->B and C->D, so no triplets.
        probas = np.zeros([proba.shape[0], 4])
        probas[:, 0] =  proba
        probas[:, 1] =  np.roll(proba, lag)
        probas[:, 2] =  np.roll(proba, 80)
        probas[:, 3] =  np.roll(proba, 80+lag)

        sf1, sb1 = tdlm.compute_1step(probas, tf, max_lag=30, n_shuf=10)
        sf2, sb2 = tdlm.compute_2step(probas, tf, max_lag=30, n_shuf=10)
        plotting.plot_sequenceness(sf1, sb1, which =['fwd', 'bkw'])
        plotting.plot_sequenceness(sf2, sb2, which =['fwd', 'bkw'])

        assert sf1[0, lag] >  sf2[0, lag]*2, \
            '2step is not significantly smaller than 1step!'

    def test_alpha_freq_state_relabeling(self):
        """with alpha control, relabeling states must not change sequenceness,
        also if max_lag is not a multiple of alpha_freq"""
        rng = np.random.default_rng(0)
        probas = rng.random([1000, 5])
        tf = tdlm.seq2tf('ABCDE')
        perm = rng.permutation(5)
        for max_lag in [30, 35]:
            sf, sb = tdlm.compute_1step(probas, tf, n_shuf=1, max_lag=max_lag, alpha_freq=10)
            sf_p, sb_p = tdlm.compute_1step(probas[:, perm], tf[np.ix_(perm, perm)],
                                            n_shuf=1, max_lag=max_lag, alpha_freq=10)
            np.testing.assert_allclose(sf, sf_p)
            np.testing.assert_allclose(sb, sb_p)

# if True:
#         # create a very simple fake probability vector with ones and zeros
#         tf = np.roll(np.eye(5), 1, axis=1)
#         sfreq = 100
#         length = 60
#         proba = np.random.rand(sfreq*length)
#         # induce time lag of 3 time steps
#         lag = 10
#         probas = np.zeros([proba.shape[0], 5])
#         for i in range(5):
#             probas[:, i] = np.roll(proba, i*lag)

#         # sf, sb = tdlm.compute_1step(probas, tf, n_shuf=10)
#         # sf_a, sb_a = tdlm.compute_1step(probas, tf, n_shuf=10, alpha_freq=10)
#         sf_a2, sb_a2 = tdlm.compute_1step(probas, tf, n_shuf=10, alpha_freq=9)


#         fig, ax = plt.subplots(figsize=[8, 6])
#         plotting.plot_sequenceness(sf, sb, which =['fwd'], ax=ax)
#         plotting.plot_sequenceness(sf_a, sb_a, which =['fwd'], ax=ax, color='red', clear=False)
#         plotting.plot_sequenceness(sf_a2, sb_a2, which =['fwd'], ax=ax, color='green', clear=False)


def _oscillation(period, n_states=5, n_samples=6000, amp=1.0, phase_lag=2):
    """sinusoid of a given period (in samples) for each state, each state
    delayed by phase_lag samples relative to the previous one, such that the
    oscillation itself creates (spurious) forward sequenceness"""
    t = np.arange(n_samples)[:, None]
    phases = -2 * np.pi * phase_lag * np.arange(n_states) / period
    return amp * np.sin(2 * np.pi * t / period + phases)


def _sequence(lag, n_states=5, n_samples=6000, rng=None):
    """noise probabilities with a true A->B->C->... sequence at a given lag"""
    rng = np.random.default_rng(rng)
    base = rng.random(n_samples)
    probas = rng.random([n_samples, n_states]) * 0.5
    for i in range(n_states):
        probas[:, i] += np.roll(base, i * lag)
    return probas


class TestAlphaControl(unittest.TestCase):
    """check that alpha_freq controls for oscillations of that period, also
    if max_lag is not a multiple of alpha_freq (see #15)"""

    tf = tdlm.seq2tf('ABCDE')
    periods = [7, 8, 10, 12]
    max_lags = [30, 35, 40, 41]

    def test_alpha_divides_phase_lagged_oscillation(self):
        """an oscillation in the probabilities causes oscillating
        sequenceness. Lags ilag, ilag+alpha_freq, ... end up in the same GLM
        and are collinear for an oscillation of period alpha_freq, so the
        oscillation is divided across the k lags of that GLM: each lag should
        be ~1/k of the uncontrolled sequenceness. It cannot be removed
        completely, as the data cannot tell which of the lags is the true one"""
        for period in self.periods:
            for max_lag in self.max_lags:
                with self.subTest(period=period, max_lag=max_lag):
                    rng = np.random.default_rng(period * 100 + max_lag)
                    probas = rng.random([6000, 5]) + _oscillation(period)
                    sf, _ = tdlm.compute_1step(probas, self.tf, n_shuf=1,
                                               max_lag=max_lag)
                    sf_a, _ = tdlm.compute_1step(probas, self.tf, n_shuf=1,
                                                 max_lag=max_lag,
                                                 alpha_freq=period)
                    sf, sf_a = sf[0, 1:], sf_a[0, 1:]

                    # plt.plot(np.transpose([sf, sf_a]))

                    # number of lags in the GLM that each lag belongs to
                    lags = np.arange(max_lag)
                    k = np.array([len(range(l % period, max_lag, period))
                                  for l in lags])
                    expected = sf / k

                    np.testing.assert_allclose(sf_a, expected, rtol=0,
                                               atol=0.1 * np.abs(sf).max())
                    assert np.corrcoef(sf_a, expected)[0, 1] > 0.98
                    # overall oscillation amplitude goes down roughly by
                    # the smallest number of lags in a GLM
                    assert np.abs(sf_a).max() < np.abs(sf).max() / k.min() * 1.5

    def test_alpha_removes_echoes(self):
        """a true sequence at seq_lag where the underlying signal carries a
        rhythm of period P creates echoes at seq_lag + P, seq_lag + 2P, ...
        alpha control should remove these echoes and keep the true peak"""
        n_samples = 6000
        t = np.arange(n_samples)
        for period in self.periods:
            for max_lag in self.max_lags:
                for seq_lag in [2, 3]:  # not P/2, else no echoes arise
                    with self.subTest(period=period, max_lag=max_lag, seq_lag=seq_lag):
                        rng = np.random.default_rng(period * 100 + max_lag + seq_lag)
                        base = rng.random(n_samples) + 0.8 * np.sin(2 * np.pi * t / period)
                        probas = rng.random([n_samples, 5]) * 0.5
                        for i in range(5):
                            probas[:, i] += np.roll(base, i * seq_lag)

                        sf, _ = tdlm.compute_1step(probas, self.tf, n_shuf=1,
                                                   max_lag=max_lag)
                        sf_a, _ = tdlm.compute_1step(probas, self.tf, n_shuf=1,
                                                     max_lag=max_lag,
                                                     alpha_freq=period)
                        echoes = np.arange(seq_lag + period, max_lag + 1, period)
                        peak, peak_a = sf[0, seq_lag], sf_a[0, seq_lag]

                        # without control there are clear echoes
                        assert np.abs(sf[0, echoes]).min() > 0.25 * peak
                        # with control they are gone, true peak is preserved
                        assert np.abs(sf_a[0, echoes]).max() < 0.1 * peak_a
                        assert peak_a > 0.8 * peak
                        self.assertEqual(np.nanargmax(sf_a[0]), seq_lag)


if __name__=='__main__':
    unittest.main()
