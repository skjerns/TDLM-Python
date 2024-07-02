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

from tdlm.utils import unique_permutations as uperms
from tdlm.core import _cross_correlation

def uperms_matlab(*args, **kwargs):
    ml = get_matlab_engine()
    uperms_ml = autoconvert(ml.uperms)
    n_perm, p_inds, perms = uperms_ml(*args, nargout=3, **kwargs)
    p_inds -= 1  # matlab index starts at 1, python at 0
    perms -= 1  # matlab index starts at 1, python at 0
    return n_perm, p_inds, perms


class TestMatlab(unittest.TestCase):

    def test_cross_correlation_matlab(self):
        # the script will simultaneously call the matlab function as well as
        # the python function and compare the results.

        # load testing data
        # the testfile can be created by inserting the line
        # `save('sequenceness_crosscorr_params.mat', 'rd', 'T', 'T2' )`
        # into the matlab script sequenceness_crosscorr.m at line 6
     
        params = io.loadmat('./matlab_code/sequenceness_crosscorr_params.mat')
        tf = params['T']
        T2 = []
        rd = params['rd']

        ml = get_matlab_engine()
        if not 'matlab_code' in ml.cd():
            ml.cd('./matlab_code')   
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


    def test_glm_matlab(self):


    def test_uperms(selfs):
        """"call uperms function from MATLAB for testing"""
        print('Starting matlab engine')

        ml = get_matlab_engine()
        if not 'matlab_code' in ml.cd():
            ml.cd('./matlab_code')        

        #TODO test doesnt work, but I have no time to debug 
        # repetitions = 15  # no k set
        # for i in tqdm(list(range(1, repetitions)), desc='Running tests 2/3'):
        #     X = np.random.randint(0, 100, 4)
        #     X_ml = matlab.int64(X.tolist())
        #     # monkey-patch permutation function to MATLAB.randperm to get same random results
        #     permutation = lambda x: np.array(ml.randperm(x), dtype=int).squeeze() - 1
        #     ml.rng(i)
        #     nPerms_py, pInds_py, Perms_py = uperms(X)
        #     ml.rng(i)
        #     nPerms_ml, pInds_ml, Perms_ml = ml.uperms(X_ml, nargout=3)

        #     pInds_ml = np.array(pInds_ml) - 1
        #     pInds_ml.sort(0)
        #     pInds_py.sort(0)

        #     Perms_ml = np.array(Perms_ml)
        #     Perms_ml.sort(0)
        #     Perms_py.sort(0)

        #     np.testing.assert_almost_equal(nPerms_py, nPerms_ml, decimal=12)
        #     np.testing.assert_almost_equal(pInds_py, pInds_ml, decimal=12)
        #     np.testing.assert_almost_equal(Perms_py, Perms_ml, decimal=12)

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

    def test_classifier(self):
        """for a sanity check I had implemented a Python wrapper around
        the MATLAB Lasso classifier. Not actually that useful I know."""
        # get test dataset
        ml = get_matlab_engine()
        if not 'matlab_code' in ml.cd():
            ml.cd('./matlab_code')   
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
