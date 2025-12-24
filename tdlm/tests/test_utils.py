# -*- coding: utf-8 -*-

import os
import unittest
import numpy as np
import math
from tdlm.utils import  unique_permutations
from tdlm.utils import _unique_permutations_MATLAB
from tdlm.utils import _trans_overlap


class TestUtils(unittest.TestCase):

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
