# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 15:24:08 2024

test if MATLAB original code gives same results as current implementation

@author: simon.kern
"""
import os
import sys; sys.path.append(os.path.abspath('./matlab_code'))
import unittest
import numpy as np
from scipy import io
from tqdm import tqdm
from joblib.parallel import Parallel, delayed
import matlab
from matlab_funcs import get_matlab_engine, autoconvert, MATLABLasso
from sklearn.datasets import load_iris

from tdlm.tools import uperms


# from tdlm.core import sequencess_crosscorr

def uperms_matlab(*args, **kwargs):
    ml = get_matlab_engine()
    uperms_ml = autoconvert(ml.uperms)
    n_perm, p_inds, perms = uperms_ml(*args, nargout=3, **kwargs)
    p_inds -= 1  # matlab index starts at 1, python at 0
    perms -= 1  # matlab index starts at 1, python at 0
    return n_perm, p_inds, perms


class TestMatlab(unittest.TestCase):

    def test_compare_to_matlab(self):
        # the script will simultaneously call the matlab function as well as
        # the python function and compare the results.

        # load testing data
        if os.path.exists('./matlab_code/sequenceness_crosscorr_params.mat'):
            # the testfile can be created by inserting the line
            # `save('sequenceness_crosscorr_params.mat', 'rd', 'T', 'T2' )`
            # into the matlab script sequenceness_crosscorr.m
            params = io.loadmat('sequenceness_crosscorr_params.mat')
            T = params['T']
            T2 = []
            rd = params['rd']
        else:
            # if file isn't present just make up some data
            T1 = np.ones([8, 8])
            T2 = T1.copy()
            for i in np.random.randint(0, 8, [6]): T1[i] = True
            for i in np.random.randint(0, 8, [6]): T2[i] = True
            rd = np.random.randn(6000, 8)

        print('Starting matlab engine')
        ml = get_matlab_engine()

        rd_ml = matlab.double(rd.tolist())
        T_ml = matlab.double(T.tolist())
        T2_ml = matlab.double(T2)
        for lag in tqdm(list(range(1, 60)), desc='Comparing algorithms'):
            res1_ml = ml.sequenceness_crosscorr(rd_ml, T_ml, [], lag)
            res2_ml = ml.sequenceness_crosscorr(rd_ml, T_ml, T2_ml, lag)
            res1_py = sequenceness_crosscorr(rd, T, None, lag)
            res2_py = sequenceness_crosscorr(rd, T, T2, lag)

            np.testing.assert_almost_equal(res1_ml, res1_py, decimal=12)
            np.testing.assert_almost_equal(res2_ml, res2_py, decimal=12)
        print('Algorithms gave equivalent results.')

    def test_uperms(selfs):
        """"call uperms function from MATLAB for testing"""
        print('Starting matlab engine')

        _np_permute = np.random.permutation
        ml = get_matlab_engine()
        ml.cd('./matlab_code')

        repetitions = 15
        for i in tqdm(list(range(repetitions)), desc='Running tests 1/3'):
            X = np.random.randint(0, 100, 4)
            X_ml = matlab.int64(X.tolist())
            k = np.random.randint(1, len(X))
            # monkey-patch permutation function to MATLAB.randperm to get same random results
            permutation = lambda x: np.array(ml.randperm(x), dtype=int).squeeze() - 1
            ml.rng(i)
            nPerms_py, pInds_py, Perms_py = uperms(X, k)
            ml.rng(i)
            nPerms_ml, pInds_ml, Perms_ml = ml.uperms(X_ml, k, nargout=3)

            pInds_ml = np.array(pInds_ml) - 1
            pInds_ml.sort(0)
            pInds_py.sort(0)

            Perms_ml = np.array(Perms_ml)
            Perms_ml.sort(0)
            Perms_py.sort(0)

            np.testing.assert_almost_equal(nPerms_py, nPerms_ml, decimal=12)
            np.testing.assert_almost_equal(pInds_py, pInds_ml, decimal=12)
            np.testing.assert_almost_equal(Perms_py, Perms_ml, decimal=12)

        repetitions = 15  # no k set
        for i in tqdm(list(range(repetitions)), desc='Running tests 2/3'):
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

            np.testing.assert_almost_equal(nPerms_py, nPerms_ml, decimal=12)
            np.testing.assert_almost_equal(pInds_py, pInds_ml, decimal=12)
            np.testing.assert_almost_equal(Perms_py, Perms_ml, decimal=12)

        repetitions = 15
        for i in tqdm(list(range(repetitions)), desc='Running tests 3/3'):
            n, m = np.random.randint(2, 7, [2])
            X = np.random.randint(0, 100, [np.random.randint(2, 6), np.random.randint(2, 6)])
            X_ml = matlab.int64(X.tolist())
            k = np.random.randint(1, len(X))

            permutation = lambda x: np.array(ml.randperm(x),
                                             dtype=int).squeeze() - 1  # monkey-patch to get same random results
            ml.rng(i)
            nPerms_py, pInds_py, Perms_py = uperms(X, None)
            permutation = _np_permute
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
            permutation = _np_permute

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

        for backend in ['sequential', 'threading', 'multiprocessing', 'loky']:
            # Make sure uperms also works together with Parallel
            res = Parallel(n_jobs=2, backend=backend)(delayed(uperms)(X, 30) for i in range(10))
            for i in range(8):
                x1, y1, z1 = res[i]
                x2, y2, z2 = res[i + 1]
                np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, y1, y2)
                np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, z1, z2)

    def test_classifier(self):
    
        
        # get test dataset
        X, y = load_iris(return_X_y=True)

        # binominal case
        clf = MATLABLasso()
        clf.fit(X[y<2], y[y<2])
        preds = clf.predict(X[y<2])
        assert np.sum(preds<0)==50
        
        # multinominal case
        clf = MATLABLasso()
        clf.fit(X, y)
        preds = clf.predict(X).argmax(1)
        assert np.sum(preds==0)==50
        assert np.sum(preds==1)==51
        assert np.sum(preds==2)==49
        
if __name__ == '__main__':
    unittest.main()
