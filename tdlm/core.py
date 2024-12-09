# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 15:10:07 2024

core functions for Temporally Delayed Linear Modelling

@author: simon.kern
"""
import os
import numpy as np
import logging
from tdlm.utils import unique_permutations, seq2tf
from scipy.linalg import toeplitz
from numba import njit
from scipy.stats import zscore as zscore_func  # with alias to prevent clash
from joblib import Parallel, delayed
from numpy.linalg import pinv

# try:
#     from jax.numpy.linalg import pinv
# except ModuleNotFoundError:
#     logging.warning('jaxlib not installed, can massively speed up with GPU')

# some helper functions to make matlab work like python
ones = lambda *args, **kwargs: np.ones(shape=args, **kwargs)
zeros = lambda *args, **kwargs: np.zeros(shape=args, **kwargs)
nan = lambda *args: np.full(shape=args, fill_value=np.nan)
squash = lambda arr: np.ravel(arr, 'F')  # MATLAB uses Fortran style reshaping



# @profile

def _find_betas(preds: np.ndarray, n_states: int, max_lag: int, alpha_freq=None):
    """for prediction matrix X (states x time), get transitions up to max_lag.
    Similar to cross-correlation, i.e. shift rows of matrix iteratively

    paralellizeable version
    """
    n_bins = max_lag + 1;

    # design matrix is now a matrix of nsamples X (n_states*max_lag)
    # with each column a shifted version of the state vector (shape=nsamples)
    dm = np.hstack([toeplitz(preds[:, kk], [zeros(n_bins, 1)])[:, 1:] for kk in range(n_states)])

    betas = nan(n_states * max_lag, n_states);

    ## GLM: state regression, with other lags
    #TODO: Check if this does what is expected
    bins = alpha_freq if alpha_freq else max_lag

    for ilag in list(range(bins)):
        # create individual GLMs for each time lagged version
        ilag_idx = np.arange(0, n_states * max_lag, bins) + ilag +1;
        # add a vector of ones for controlling the regression
        ilag_X = np.pad(dm[:, ilag_idx], [[0, 0], [0, 1]], constant_values=1)

        # add control for certain time lags to reduce alpha
        # Now find coefficients that solve the linear regression for this timelag
        # this a the second stage regression
        # print(ilag_X.shape)
        ilag_betas = pinv(ilag_X) @ preds;  # if SVD fails, use slow, exact solution
        betas[ilag_idx, :] = ilag_betas[0:-1, :];

    return betas

@njit
def _numba_roll(X, shift):
    """
    numba optimized np.roll function
    taken from https://github.com/tobywise/online-aversive-learning
    """
    # Rolls along 1st axis
    new_X = np.zeros_like(X)
    for i in range(X.shape[1]):
        new_X[:, i] = np.roll(X[:, i], shift)
    return new_X


# @njit
def _cross_correlation(preds, tf, tb, max_lag=40, min_lag=0):
    """
    Computes sequenceness by cross-correlation

    taken from https://github.com/tobywise/online-aversive-learning
    """
    preds_f = preds @ tf
    preds_b = preds @ tb

    ff = np.zeros(max_lag - min_lag)
    fb = np.zeros(max_lag - min_lag)

    for lag in range(min_lag, max_lag):

        r = np.corrcoef(preds[lag:, :].T, _numba_roll(preds_f, lag)[lag:, :].T)
        r = np.diag(r, k=tf.shape[0])
        forward_mean_corr = np.nanmean(r)

        r = np.corrcoef(preds[lag:, :].T, _numba_roll(preds_b, lag)[lag:, :].T)
        r = np.diag(r, k=tb.shape[0])
        backward_mean_corr = np.nanmean(r)

        ff[lag - min_lag] = forward_mean_corr
        fb[lag - min_lag] = backward_mean_corr

    return ff, fb

def sequenceness_crosscorr(preds, tf, tb=None, n_shuf=1000, min_lag=0, max_lag=50,
                           alpha_freq=None):

    n_states = preds.shape[-1]
    # unique permutations
    _, unique_perms, _ = unique_permutations(np.arange(1, n_states + 1), n_shuf)


    if tb is None:
        # backwards is transpose of forwards
        tb = tf.T

    seq_fwd_corr = nan(n_shuf, max_lag + 1)  # forward cross-correlation
    seq_bkw_corr = nan(n_shuf, max_lag + 1)  # backward cross-correlation

    for i in range(n_shuf):
        # select next unique permutation of transitions
        # index 0 is the non-shuffled original transition matrix
        rp = unique_perms[i, :]
        tf_perm = tf[rp, :][:, rp]
        tb_perm = tb[rp, :][:, rp]
        seq_fwd_corr[i, :-1], seq_bkw_corr[i, :-1] = _cross_correlation(preds,
                                                                        tf_perm,
                                                                        tb_perm,
                                                                        max_lag=max_lag,
                                                                        min_lag=min_lag)
    return seq_fwd_corr, seq_bkw_corr

# @profile
def compute_1step(preds, tf, tb=None, n_shuf=1000, min_lag=0, max_lag=50,
                  alpha_freq=None, seed=None):
    """
    Calculate 1-step-sequenceness for probability estimates and transitions.

    Parameters
    ----------
    preds : np.ndarray
        2d matrix with predictions, shape= (n_states, times), where each
        timestep contains n_states prediction values for states at that time
    tf : np.ndarray
        transition matrix with expected transitions for the underlying states.
    tb : np.ndarray
        backward transition matrix expected transitions for the underlying
        states. In case transitions are non-directional, the backwards matrix
        is simply set to be the transpose of tf. Default tb = tf.T
    n_shuf : int
        number of random shuffles to be done for permutation testing.
    max_lag : int
        maximum time lag to calculate. Time dimension is measured in sample
        steps of the preds time dimension.
    alpha_freq : int, optional
        Alpha oscillation frequency to control for. Time shifted copies of the
        signal are added in this frequency to the GLM, acting as a confounds.
        Warning: Must be supplied in sample points, not in Hertz!
        The default is None.
    n_steps : int, optional
        number of transition steps to look for. Not implemented yet.
        The default is 1.


    Returns
    -------
    sf : np.ndarray
        forward sequencess for all time lags and shuffles. Row 0 is the
        non-shuffled version. First lag is NAN as it is undefined for lag = 0
    sb : np.ndarray
        backward sequencess for all time lags and shuffles. Row 0 is the
        non-shuffled version. First lag is NAN as it is undefined for lag = 0

    """
    if seed is not None:
        np.random.seed(seed)
    n_states = preds.shape[-1]
    # unique permutations
    _, unique_perms, _ = unique_permutations(np.arange(1, n_states + 1), n_shuf)

    seq_fwd = nan(n_shuf, max_lag + 1)  # forward sequenceness
    seq_bkw = nan(n_shuf, max_lag + 1)  # backward sequencenes

    if tb is None:
        # backwards is transpose of forwards
        tb = tf.T

    ## GLM: state regression, with other lags

    betas = _find_betas(preds, n_states, max_lag, alpha_freq=alpha_freq)
    # betas = find_betas_optimized(X, n_states, max_lag, alpha_freq=alpha_freq)
    # np.testing.assert_array_almost_equal(betas, betas2, decimal= 12)

    # reshape the coeffs for regression to be in the order of ilag x (n_states x n_states)
    betasn_ilag_stage = np.reshape(betas, [max_lag, n_states ** 2], order='F');

    for i in range(n_shuf):
        rp = unique_perms[i, :]  # select next unique permutation of transitions
        tf_perm = tf[rp, :][:, rp]
        tb_perm = tb[rp, :][:, rp]
        t_auto = np.eye(n_states)  # control for auto correlations
        t_const = np.ones([n_states, n_states])  # keep betas in same range

        # create our design matrix for the second step analysis
        dm = np.vstack([squash(tf_perm), squash(tb_perm), squash(t_auto), squash(t_const)]).T
        # print(dm.shape)
        # now calculate regression coefs for use with transition matrix
        bbb = pinv(dm) @ (betasn_ilag_stage.T)  #%squash(ones(n_states))

        seq_fwd[i, 1:] = bbb[0, :]  # forward coeffs
        seq_bkw[i, 1:] = bbb[1, :]  # backward coeffs

    return seq_fwd, seq_bkw


# # @profile
# def compute_1step_parallel(preds, tf, tb=None, n_shuf=1000, min_lag=0, max_lag=50,
#                   alpha_freq=None,  cross_corr=False, n_jobs=-1):
#     """
#     Calculate 1-step-sequenceness for probability estimates and transitions.

#     Parameters
#     ----------
#     preds : np.ndarray
#         2d matrix with predictions, shape= (n_states, times), where each
#         timestep contains n_states prediction values for states at that time
#     tf : np.ndarray
#         transition matrix with expected transitions for the underlying states.
#     tb : np.ndarray
#         backward transition matrix expected transitions for the underlying
#         states. In case transitions are non-directional, the backwards matrix
#         is simply set to be the transpose of tf. Default tb = tf.T
#     n_shuf : int
#         number of random shuffles to be done for permutation testing.
#     max_lag : int
#         maximum time lag to calculate. Time dimension is measured in sample
#         steps of the preds time dimension.
#     alpha_freq : int, optional
#         Alpha oscillation frequency to control for. Time shifted copies of the
#         signal are added in this frequency to the GLM, acting as a confounds.
#         Warning: Must be supplied in sample points, not in Hertz!
#         The default is None.
#     n_steps : int, optional
#         number of transition steps to look for. Not implemented yet.
#         The default is 1.
#     cross_corr : bool, optional
#         Additionally to GLM analysis, perform cross correlation.
#         The default is False.
#     n_jobs : int, optional
#         Number of parallel cores to use for some of the sub-analysis.
#         The default is 1.

#     Returns
#     -------
#     sf : np.ndarray
#         forward sequencess for all time lags and shuffles. Row 0 is the
#         non-shuffled version
#     sb : np.ndarray
#         backward sequencess for all time lags and shuffles. Row 0 is the
#         non-shuffled version

#     """

#     n_states = preds.shape[-1]
#     # unique permutations
#     _, unique_perms, _ = unique_permutations(np.arange(1, n_states + 1), n_shuf)

#     seq_fwd = nan(n_shuf, max_lag + 1)  # forward sequenceness
#     seq_bkw = nan(n_shuf, max_lag + 1)  # backward sequencenes

#     if tb is None:
#         # backwards is transpose of forwards
#         tb = tf.T

#     ## GLM: state regression, with other lags

#     betas = _find_betas(preds, n_states, max_lag, alpha_freq=alpha_freq)
#     # betas = find_betas_optimized(X, n_states, max_lag, alpha_freq=alpha_freq)
#     # np.testing.assert_array_almost_equal(betas, betas2, decimal= 12)

#     # reshape the coeffs for regression to be in the order of ilag x (n_states x n_states)
#     betasn_ilag_stage = np.reshape(betas, [max_lag, n_states ** 2], order='F');

#     bbbs = Parallel(n_jobs)(delayed(_compute_1step)(betasn_ilag_stage,
#                                                     tf[unique_perms[i, :], :][:, unique_perms[i, :]],
#                                                     tb[unique_perms[i, :], :][:, unique_perms[i, :]],
#                                                     n_states) for i in range(n_shuf))
#     for i in range(n_shuf):
#         seq_fwd[i, 1:] = bbbs[i][0, :]  # forward coeffs
#         seq_bkw[i, 1:] = bbbs[i][1, :]  # backward coeffs

#     return seq_fwd, seq_bkw

# def _compute_1step(betasn_ilag_stage, tf_perm, tb_perm, n_states):
#     t_auto = np.eye(n_states)  # control for auto correlations
#     t_const = np.ones([n_states, n_states])  # keep betas in same range

#     # create our design matrix for the second step analysis
#     dm = np.vstack([squash(tf_perm), squash(tb_perm), squash(t_auto), squash(t_const)]).T

#     # now calculate regression coefs for use with transition matrix
#     bbb = _pinv(dm) @ (betasn_ilag_stage.T)  #%squash(ones(n_states))
#     return bbb
#%% __main__
if __name__=='__main__':
    import stimer
    # with stimer:
    #     a1, b1 = compute_1step(np.random.rand(20000, 10), tf=np.eye(10))

    preds = np.random.rand(8, 10000).T
    n_states = preds.shape[1]
    alpha_freq = None
    max_lag = 50
    with stimer('unoptimized'):
        for i in range(5):
            x1 = _find_betas(preds, n_states=n_states, max_lag=max_lag, alpha_freq=10)


# def sequenceness(preds, tf, tb=None, n_shuf=1000, max_lag=30, alpha_freq=None,
#                  n_steps=1, cross_corr=False, n_jobs=1, zscore=False):
#     """
#     Calculate sequenceness for given probability estimates and transitions.

#     Sequenceness can be either calculated using a GLM or cross correlation.
#     Permutation testing is done using unique permutations

#     Parameters
#     ----------
#     preds : np.ndarray
#         2d matrix with predictions, shape= (n_states, times), where each
#         timestep contains n_states prediction values for states at that time
#     tf : np.ndarray
#         forward transition matrix with expected transitions for the underlying
#         states.
#     tb : np.ndarray
#         backward transition matrix expected transitions for the underlying
#         states. In case transitions are non-directional, the backwards matrix
#         is simply set to be the transpose of tf. Default tb = tf.T
#     n_shuf : int
#         number of random shuffles to be done for permutation testing.
#     max_lag : int
#         maximum time lag to calculate. Time dimension is measured in sample
#         steps of the preds time dimension.
#     alpha_freq : int, optional
#         Alpha oscillation frequency to control for. Time shifted copies of the
#         signal are added in this frequency to the GLM, acting as a confounds.
#         The default is None.
#     n_steps : int, optional
#         number of transition steps to look for. Not implemented yet.
#         The default is 1.
#     cross_corr : bool, optional
#         Additionally to GLM analysis, perform cross correlation.
#         The default is False.
#     n_jobs : int, optional
#         Number of parallel cores to use for some of the sub-analysis.
#         The default is 1.
#     zscore : bool, optional
#         zscore the resulting probability values. This has the advantage of
#         normalizing the sequenceness values across participants but the
#         disadvantage of making the absolute values no longer interpretable.
#         This was e.g. done in Wimmer et al. 2020. Another disadvantage is that
#         the summed differential sequenceness will by definition always be zero
#         making a time windowed approach as in Wise et al. (2021) inapplicable.
#         The default is False.

#     Raises
#     ------
#     NotImplementedError
#         2 step TDLM isn't implemented yet.

#     Returns
#     -------
#     sf : np.ndarray
#         forward sequencess for all time lags and shuffles. Row 0 is the
#         non-shuffled version
#     sb : np.ndarray
#         backward sequencess for all time lags and shuffles. Row 0 is the
#         non-shuffled version
#     sf_corr : np.ndarray
#         forward cross-correlation for all time lags and shuffles. Row 0 is the
#         non-shuffled version. None if no correlation is done
#     sb_corr : np.ndarray
#         backward cross-correlation for all time lags and shuffles. Row 0 is the
#         non-shuffled version. None if no correlation is done

#     """
#     assert preds.ndim == 2, 'predictions must be 2d'
#     assert preds.shape[0] > preds.shape[1], f'preds shape[1] should be time dimension but{preds.shape=}'

#     if n_steps == 1:
#         sf, sb, sf_corr, sb_corr = compute(preds=preds,
#                                            tf=tf,
#                                            n_shuf=n_shuf,
#                                            max_lag= max_lag,
#                                            alpha_freq=alpha_freq,
#                                            cross_corr=cross_corr)
#     elif n_steps == 2:
#         raise NotImplementedError('not yet implemented')
#         sf, sb, sf_corr, sb_corr = compute_2step(preds, seq, n_shuf, max_lag, uniquePerms,
#                                                  alpha_freq=alpha_freq, cross_corr=cross_corr,
#                                                  n_jobs=n_jobs)
#     if zscore:
#         sf = zscore_func(sf, -1, nan_policy='omit')
#         sb = zscore_func(sb, -1, nan_policy='omit')
#         sf_corr = zscore_func(sf_corr, -1, nan_policy='omit')
#         sb_corr = zscore_func(sb_corr, -1, nan_policy='omit')

#     return sf, sb, sf_corr, sb_corr
