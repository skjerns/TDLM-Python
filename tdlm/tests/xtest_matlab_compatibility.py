# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 15:24:08 2024

test if MATLAB original code gives same results as current implementation

TEST IS DISABLED:
MATLAB enginge for Python does not run on GitHub Actions yet.

@author: simon.kern
"""

import os
import sys

try:
    script_dir = os.path.dirname(__file__)
    sys.path.append(os.path.abspath(f'{script_dir}/matlab_code'))
except:
    pass

import unittest
import mat73
import numpy as np
from unittest.mock import patch
from scipy import io
from tqdm import tqdm
from joblib.parallel import Parallel, delayed
from matlab_funcs import get_matlab_engine, autoconvert
import matlab

import tdlm
from tdlm.utils import _unique_permutations_MATLAB as uperms
from tdlm.core import _cross_correlation

def uperms_matlab(*args, **kwargs):
    ml = get_matlab_engine()
    uperms_ml = autoconvert(ml.uperms)
    n_perm, p_inds, perms = uperms_ml(*args, nargout=3, **kwargs)
    p_inds -= 1  # matlab index starts at 1, python at 0
    perms -= 1  # matlab index starts at 1, python at 0
    return n_perm, p_inds, perms


class TestMatlab(unittest.TestCase):

    def test_uperms(self):
        """"call uperms function from MATLAB for testing"""
        print('Starting matlab engine')

        ml = get_matlab_engine()
        if not 'matlab_code' in ml.cd():
            ml.cd(f'{script_dir}/matlab_code')

        repetitions = 15  # no k set
        with patch('numpy.random.permutation', lambda x: np.array(ml.randperm(x)).squeeze()-1):
            for i in tqdm(list(range(1, repetitions)), desc='Running tests 1/2'):
                X = np.random.randint(0, 100, 4)
                X_ml = matlab.int64(X.tolist())

                # monkey-patch permutation function to MATLAB.randperm to get same random results
                permutation = lambda x: np.array(ml.randperm(x), dtype=int).squeeze() - 1
                ml.rng(i)
                nPerms_py, pInds_py, Perms_py = uperms(X)
                ml.rng(i)
                nPerms_ml, pInds_ml, Perms_ml = ml.uperms(X_ml, nargout=3)

                pInds_ml = np.array(pInds_ml) - 1
                pInds_ml.sort(0)
                pInds_py.sort(0)

                Perms_ml = np.array(Perms_ml)
                Perms_ml.sort(0)
                Perms_py.sort(0)

                np.testing.assert_almost_equal(nPerms_py, nPerms_ml)
                np.testing.assert_almost_equal(pInds_py, pInds_ml)
                np.testing.assert_almost_equal(Perms_py, Perms_ml)

        repetitions = 15
        for i in tqdm(list(range(repetitions)), desc='Running tests 2/2'):
            n, m = np.random.randint(2, 7, [2])
            X = np.random.randint(0, 100, [np.random.randint(2, 6), np.random.randint(2, 6)])
            X_ml = matlab.int64(X.tolist())
            k = np.random.randint(1, len(X))

            # monkey-patch to get same random results
            permutation = lambda x: np.array(ml.randperm(x), dtype=int).squeeze() - 1
            ml.rng(i)
            nPerms_py, pInds_py, Perms_py = uperms(X, None)
            ml.rng(i)
            nPerms_ml, pInds_ml, Perms_ml = ml.uperms(X_ml, nargout=3)
            pInds_ml = np.array(pInds_ml) - 1
            pInds_ml.sort(0)
            pInds_py.sort(0)

            Perms_ml = np.array(Perms_ml)
            Perms_ml.sort(0)
            Perms_py.sort(0)

            np.testing.assert_almost_equal(nPerms_py, nPerms_ml, decimal=12)
            np.testing.assert_almost_equal(pInds_py, pInds_ml, decimal=12)
            np.testing.assert_almost_equal(Perms_py, Perms_ml, decimal=12)

        # # Make sure uperms gives consistent results
        res = []
        X = np.random.randint(0, 100, [25, 5])
        for i in range(10):
            np.random.seed(0)
            res.append(uperms(X, 25))

        for i in range(9):
            x1, y1, z1 = res[i]
            x2, y2, z2 = res[i + 1]
            np.testing.assert_array_equal(y1, y2)
            np.testing.assert_array_equal(z1, z2)

        X = np.random.randint(0, 100, [25, 5])

        # test parallel to check that parallel calls produce different results
        for backend in ['sequential', 'threading', 'multiprocessing', 'loky']:
            # Make sure uperms also works together with Parallel
            res = Parallel(n_jobs=2, backend=backend)(delayed(uperms)(X, 30) for i in range(10))
            for i in range(8):
                x1, y1, z1 = res[i]
                x2, y2, z2 = res[i + 1]
                np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, y1, y2)
                np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, z1, z2)



    def test_cross_correlation_matlab(self):
        # the script will simultaneously call the matlab function as well as
        # the python function and compare the results.

        # load testing data
        # the testfile can be created by inserting the line
        # `save('sequenceness_crosscorr_params.mat', 'rd', 'T', 'T2' )`
        # into the matlab script sequenceness_crosscorr.m at line 6

        params = io.loadmat(f'{script_dir}/matlab_code/sequenceness_crosscorr_params.mat')
        tf = params['T']
        T2 = []
        rd = params['rd']

        ml = get_matlab_engine()
        if not 'matlab_code' in ml.cd():
            ml.cd(f'{script_dir}/matlab_code')
        rd_ml = matlab.double(rd.tolist())
        tf_ml = matlab.double(tf.tolist())
        tb_ml = matlab.double(tf.T.tolist())

        sf_ml = [ml.sequenceness_Crosscorr(rd_ml, tf_ml, [], lag) for lag in range(30)]
        sb_ml = [ml.sequenceness_Crosscorr(rd_ml, tb_ml, [], lag) for lag in range(30)]
        sf_py, sb_py = _cross_correlation(preds=rd, tf=tf, tb=tf.T, max_lag=30)

        diff_ml = np.array(sf_ml)-np.array(sb_ml)
        diff_py = sf_py-sb_py

        # plt.plot(diff_ml)
        # plt.plot(diff_py)
        # results are only equivalent to some decimal place
        # my original code was bitwise equivalent, but toby wise's code
        # has some slight deviation inside. no idea where that comes from.
        decimal = 3
        np.testing.assert_almost_equal(diff_ml, diff_py, decimal=decimal)
        print(f'Algorithms gave equivalent results up to {decimal=}.')


    def test_glm_matlab_vanilla(self):
        """test whether the results of Simulate_Replay.m are the same

        this only tests the actual sequenceness calculation. it makes
        no sense to compare the predictions themselves, as matlab and python
        implement Lasso regression differently. However, uperms is slightly
        differently implemented, so we need to monkey patch that.
        """
        data = mat73.loadmat(f'{script_dir}/matlab_code/simulate_replay_results.mat')
        preds = data['preds']
        tf = data['TF']
        sf_matlab = data['sf'].squeeze()
        sb_matlab = data['sb'].squeeze()
        uniqueperms = data['uniquePerms']-1

        # monkey patch uperms, to give equivalent results to MATLAB
        with patch('tdlm.core.unique_permutations', lambda *x: (0, uniqueperms, 0)):
            # print(tdlm.utils.unique_permutations([1,2,3]))
            sf, sb = tdlm.compute_1step(preds, tf, max_lag=60, n_shuf=100)

        # first test if actual sequenceness results are the same
        np.testing.assert_allclose(sf_matlab[0, :], sf[0, :])
        np.testing.assert_allclose(sb_matlab[0, :], sb[0, :])

        # next test if shuffles are the same.
        # this should depend on uperms being implemented the same
        np.testing.assert_allclose(sf_matlab[1:, :], sf[1:, :])
        np.testing.assert_allclose(sb_matlab[1:, :], sb[1:, :])

    def test_glm_matlab_alpha_correction(self):
        """test if alpha correction also gives the same results as MATLAB"""
        data = mat73.loadmat(f'{script_dir}/matlab_code/simulate_replay_withalpha_results.mat')
        preds = data['preds']
        tf = data['TF']
        sf_matlab = data['sf'].squeeze()
        sb_matlab = data['sb'].squeeze()
        uniqueperms = data['uniquePerms']-1

        # monkey patch uperms, to give equivalent results to MATLAB
        with patch('tdlm.core.unique_permutations', lambda *x: (0, uniqueperms, 0)):
            # print(tdlm.utils.unique_permutations([1,2,3]))
            sf, sb = tdlm.compute_1step(preds, tf, max_lag=60, n_shuf=100,
                                        alpha_freq=10)

        # first test if actual sequenceness results are the same
        np.testing.assert_allclose(sf_matlab[0, :], sf[0, :])
        np.testing.assert_allclose(sb_matlab[0, :], sb[0, :])

        # next test if shuffles are the same.
        # this should depend on uperms being implemented the same
        np.testing.assert_allclose(sf_matlab[1:, :], sf[1:, :])
        np.testing.assert_allclose(sb_matlab[1:, :], sb[1:, :])

    # def test_glm_matlab_multistep(self):
        # raise NotImplementedError()


    def test_glm_matlab_multistep(self):
        """test if multistep (2step) also gives the same results as MATLAB"""
        data = mat73.loadmat(f'{script_dir}/matlab_code/simulate_replay_longerlength_results.mat')
        preds = data['preds']
        tf = data['TF']
        sf_matlab = data['sf1'].squeeze()
        sb_matlab = data['sb1'].squeeze()
        max_lag = int(data['maxLag'])
        n_shuf = int(data['nShuf'])
        unique_perms = data['uniquePerms']-1

        # monkey patch uperms, to give equivalent results to MATLAB
        with patch('tdlm.core.unique_permutations', lambda *x: (0, unique_perms, 0)):
            # print(tdlm.utils.unique_permutations([1,2,3]))
            sf, sb = tdlm.compute_2step(preds, tf, max_lag=max_lag,
                                        n_shuf=n_shuf)

        # first test if actual sequenceness results are the same
        np.testing.assert_allclose(sf_matlab[0, :], sf[0, :])
        np.testing.assert_allclose(sb_matlab[0, :], sb[0, :])

        # next test if shuffles are the same.
        # this should depend on uperms being implemented the same
        np.testing.assert_allclose(sf_matlab[1:, :], sf[1:, :])
        np.testing.assert_allclose(sb_matlab[1:, :], sb[1:, :])




if __name__ == '__main__':
    unittest.main()
