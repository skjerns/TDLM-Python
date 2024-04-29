# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 15:10:07 2024

core functions for Temporally Delayed Linear Modelling

@author: simon.kern
"""

import math
import numpy as np

def unique_permutations(X, k=None):
    """
    #uperms: unique permutations of an input vector or rows of an input matrix
    # Usage:  nPerms              = uperms(X)
    #        [nPerms pInds]       = uperms(X, k)
    #        [nPerms pInds Perms] = uperms(X, k)
    #
    # Determines number of unique permutations (nPerms) for vector or matrix X.
    # Optionally, all permutations' indices (pInds) are returned. If requested,
    # permutations of the original input (Perms) are also returned.
    #
    # If k < nPerms, a random (but still unique) subset of k of permutations is
    # returned. The original/identity permutation will be the first of these.
    #
    # Row or column vector X results in Perms being a [k length(X)] array,
    # consistent with MATLAB's built-in perms. pInds is also [k length(X)].
    #
    # Matrix X results in output Perms being a [size(X, 1) size(X, 2) k]
    # three-dimensional array (this is inconsistent with the vector case above,
    # but is more helpful if one wants to easily access the permuted matrices).
    # pInds is a [k size(X, 1)] array for matrix X.
    #
    # Note that permutations are not guaranteed in any particular order, even
    # if all nPerms of them are requested, though the identity is always first.
    #
    # Other functions can be much faster in the special cases where they apply,
    # as shown in the second block of examples below, which uses perms_m.
    #
    # Examples:
    #  uperms(1:7),       factorial(7)        # verify counts in simple cases,
    #  uperms('aaabbbb'), nchoosek(7, 3)      # or equivalently nchoosek(7, 4).
    #  [n pInds Perms] = uperms('aaabbbb', 5) # 5 of the 35 unique permutations
    #  [n pInds Perms] = uperms(eye(3))       # all 6 3x3 permutation matrices
    #
    #  # A comparison of timings in a special case (i.e. all elements unique)
    #  tic; [nPerms P1] = uperms(1:20, 5000); T1 = toc
    #  tic; N = factorial(20); S = sample_no_repl(N, 5000);
    #  P2 = zeros(5000, 20);
    #  for n = 1:5000, P2(n, :) = perms_m(20, S(n)); end
    #  T2 = toc # quicker (note P1 and P2 are not the same random subsets!)
    #  # For me, on one run, T1 was 7.8 seconds, T2 was 1.3 seconds.
    #
    #  # A more complicated example, related to statistical permutation testing
    #  X = kron(eye(3), ones(4, 1));  # or similar statistical design matrix
    #  [nPerms pInds Xs] = uperms(X, 5000); # unique random permutations of X
    #  # Verify correctness (in this case)
    #  G = nan(12,5000); for n = 1:5000; G(:, n) = Xs(:,:,n)*(1:3)'; end
    #  size(unique(G', 'rows'), 1)    # 5000 as requested.
    #
    # See also: randperm, perms, perms_m, signs_m, nchoosek_m, sample_no_repl
    # and http://www.fmrib.ox.ac.uk/fsl/randomise/index.html#theory

    # Copyright 2010 Ged Ridgway
    # http://www.mathworks.com/matlabcentral/fileexchange/authors/27434
    """
    # Count number of repetitions of each unique row, and get representative x
    X = np.array(X).squeeze()
    assert len(X) > 1

    if X.ndim == 1:
        uniques, uind, c = np.unique(X, return_index=True, return_counts=True)
    else:
        # [u uind x] = unique(X, 'rows'); % x codes unique rows with integers
        uniques, uind, c = np.unique(X, axis=0, return_index=True, return_counts=True)

    uniques = uniques.tolist()
    x = np.array([uniques.index(i) for i in X.tolist()])

    c = sorted(c)
    nPerms = np.prod(np.arange(c[-1] + 1, np.sum(c) + 1)) / np.prod([math.factorial(x) for x in c[:-1]])
    nPerms = int(nPerms)
    # % computation of permutation
    # Basics
    n = len(X);
    if k is None or k > nPerms:
        k = nPerms;  # default to computing all unique permutations

    # % Identity permutation always included first:
    pInds = np.zeros([int(k), n]).astype(np.uint32)
    Perms = pInds.copy();
    pInds[0, :] = np.arange(0, n);
    Perms[0, :] = x;

    # Add permutations that are unique
    u = 0;  # to start with
    while u < k - 1:
        pInd = np.random.permutation(int(n));
        pInd = np.array(pInd).astype(int)  # just in case MATLAB permutation was monkey patched
        if x[pInd].tolist() not in Perms.tolist():
            u += 1
            pInds[u, :] = pInd
            Perms[u, :] = x[pInd]
    # %
    # Construct permutations of input
    if X.ndim == 1:
        Perms = np.repeat(np.atleast_2d(X), k, 0)
        for n in np.arange(1, k):
            Perms[n, :] = X[pInds[n, :]]
    else:
        Perms = np.repeat(np.atleast_3d(X), k, axis=2);
        for n in np.arange(1, k):
            Perms[:, :, n] = X[pInds[n, :], :]
    return (nPerms, pInds, Perms)


def char2num(seq):
    """convert list of chars to integers eg ABC=>012"""
    if isinstance(seq, str):
        seq = list(seq)
    assert ord('A')-65 == 0
    nums = [ord(c.upper())-65 for c in seq]
    assert all([0<=n<=90 for n in nums])
    return nums


def num2char(arr):
    """convert list of ints to alphabetical chars eg 012=>ABC"""
    if isinstance(arr, int):
        return chr(arr+65)
    arr = np.array(arr, dtype=int)
    return np.array([chr(x+65) for x in arr.ravel()]).reshape(*arr.shape)


def tf2seq(TF):
    """from transition matrix to alphanumerical sequence"""
    raise NotImplementedError
    seq = ''
    for i, row in enumerate(TF):
        for j, col in enumerate(row):
            np.where()
    return seq

def seq2tf(sequence, n_states=None):
    """
    create a transition matrix from a sequence string,
    e.g. ABCDEFG
    Please note that sequences will not be wrapping automatically,
    i.e. a wrapping sequence should be denoted by appending the first state.

    :param sequence: sequence in format "ABCD..."
    :param seqlen: if not all states are part of the sequence,
                   the number of states can be specified
                   e.g. if the sequence is ABE, but there are also states F,G
                   n_states would be 7

    """

    seq = char2num(sequence)
    if n_states is None:
        n_states = max(seq)+1
    # assert max(seq)+1==n_states, 'not all positions have a transition'
    TF = np.zeros([n_states, n_states], dtype=int)
    for i, p1 in enumerate(seq):
        if i+1>=len(seq): continue
        p2 = seq[(i+1) % len(seq)]
        TF[p1, p2] = 1
    return TF.astype(float)

def seq2TF_2step(seq, n_states=None):
    """create a transition matrix with all 2 steps from a sequence string,
    e.g. ABCDEFGE. """
    import pandas as pd
    triplets = []
    if n_states is None:
        n_states = max(char2num(seq))+1
    TF2 = np.zeros([n_states**2, n_states], dtype=int)
    for i, p1 in enumerate(seq):
        if i+2>=len(seq): continue
        triplet = seq[i] + seq[(i+1) % len(seq)] + seq[(i+2)% len(seq)]
        i = char2num(triplet[0])[0] * n_states + char2num(triplet[1])[0]
        j = char2num(triplet[2])
        TF2[i, j] = 1
        triplets.append(triplet)

    seq_set = num2char(np.arange(n_states))
    # for visualiziation purposes
    df = pd.DataFrame({c:TF2.T[i] for i,c in enumerate(seq_set)})
    df['index'] = [f'{y}{x}' for y in seq_set for x in seq_set]
    df = df.set_index('index')
    TF2 = df.loc[~(df==0).all(axis=1)]
    return TF2