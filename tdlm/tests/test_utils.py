# -*- coding: utf-8 -*-

import os
import unittest
import numpy as np
import scipy
import math
from scipy import io
from tqdm import tqdm
from tdlm.utils import create_travelling_wave, unique_permutations
from tdlm.utils import _unique_permutations_MATLAB

from tdlm.utils import simulate_meeg, _trans_overlap
import matplotlib.pyplot as plt


class TestUtils(unittest.TestCase):
    def setUp(self):
        self.hz = 10  # 10 Hz frequency
        self.sfreq = 100  # 1000 Hz sampling rate
        self.size = 1000  # 10 second of data
        self.seconds = 10
        self.sensor_pos = [
            (0, 0),  # Center
            (1, 0),  # 1 cm right
            (-1, 0),  # 1 cm left
            (0, 1),  # 1 cm up
            (0, -1)  # 1 cm down
        ]
        self.speed = 50  # 50 cm/second

    def test_output_shape(self):
        wave = create_travelling_wave(hz = self.hz,
                                      seconds=self.seconds,
                                      sfreq = self.sfreq,
                                      chs_pos=self.sensor_pos,
                                      speed=self.speed)
        self.assertEqual(wave.shape, (self.size, len(self.sensor_pos)))

    def test_frequency(self):
        wave = create_travelling_wave(hz = self.hz,
                                      seconds=self.seconds,
                                      sfreq = self.sfreq,
                                      chs_pos=self.sensor_pos,
                                      speed=self.speed)
        # Check frequency of the center sensor (should be exactly as specified)
        fft = np.fft.rfft(wave[:, 0])
        freq = np.fft.rfftfreq(self.size, 1/self.sfreq)
        peak_freq = freq[np.argmax(np.abs(fft))]
        self.assertAlmostEqual(peak_freq, self.hz, places=1)

    def test_amplitude(self):
        wave = create_travelling_wave(hz = self.hz,
                                      seconds=self.seconds,
                                      sfreq = self.sfreq,
                                      chs_pos=self.sensor_pos,
                                      speed=self.speed)
        # Check if amplitude is consistent across all sensors
        amplitudes = np.max(wave, axis=0) - np.min(wave, axis=0)
        self.assertTrue(np.allclose(amplitudes, amplitudes[0], rtol=1e-5))

    def test_uperms_MATLAB(self):
        n_states = 5
        X = np.arange(n_states)+10

        for k in range(1, 120):
            (nPerms, pInds, Perms) = _unique_permutations_MATLAB(X, k)
            assert nPerms==math.factorial(n_states)
            assert len(np.unique(pInds, axis=0))==k
            assert (pInds[0]==[0, 1, 2, 3, 4]).all()

        # too many permutations requested, not possible
        with self.assertRaises(ValueError):
            (nPerms, pInds, Perms) = _unique_permutations_MATLAB(X, 121)

        (nPerms, pInds, Perms) = _unique_permutations_MATLAB(X, 120)

    def test_travelling_waves(self):
        pos = [(0, 0),       # Center
                (1, 0),       # 1 cm right
                (-1, 0),      # 1 cm left
                (0, 1),       # 1 cm up
                (0, -1),      # 1 cm down
                (2, 0),       # 2 cm right
                (-2, 0),      # 2 cm left
                (0, 2),       # 2 cm up
                (0, -2),      # 2 cm down
                (3, 0),       # 3 cm right
                (-3, 0),      # 3 cm left
                (0, 3),       # 3 cm up
                (0, -3),      # 3 cm down
                (4, 0),       # 4 cm right
                (-4, 0),      # 4 cm left
                (0, 4),       # 4 cm up
                (0, -4),      # 4 cm down
                ]
        for sfreq in np.arange(50, 500, 29):
            for div in np.arange(10):
                seconds = 500/sfreq
                wave = create_travelling_wave(7, seconds, sfreq, pos, speed=sfreq/div)
                for i in range(1,17):
                    off = (i-1)//4+1
                    np.testing.assert_array_almost_equal(wave[:200, 0], wave[div*off:200+div*off, i])

    def test_simulate_meeg(self):
        data = simulate_meeg(60, 100)


    def test_uperms(self):
        """test whether new implementation of unique_permutations works as intended"""
        X = np.arange(5)
        np.random.seed(0)

        for _ in range(10):  # repeat 10 times for repeatability
            for i in range(5):
                perms = unique_permutations(X, max_true_trans=i)
                for perm in perms[1:]:
                    assert _trans_overlap(X, perm)<=i
                assert len(perms)<=120

            for i in range(5):
                perms = unique_permutations(X, k=119, max_true_trans=i)
                for perm in perms[1:]:
                    assert _trans_overlap(X, perm)<=i
                assert len(perms)<=120-4+i

            for i in range(5):
                perms = unique_permutations(X, k=40, max_true_trans=i)
                for perm in perms[1:]:
                    assert _trans_overlap(X, perm)<=i
                assert len(perms)==40

            for i in range(5):
                perms = unique_permutations(X, k=54, max_true_trans=i)
                for perm in perms[1:]:
                    assert _trans_overlap(X, perm)<=i
                assert len(perms)==54

            perms = unique_permutations(X, max_true_trans=6)
            assert len(perms)==120

            with self.assertRaises(ValueError):
                perms = unique_permutations(X, k=121)



if __name__=='__main__':
    unittest.main()
