# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 15:10:07 2024

core functions for Temporally Delayed Linear Modelling

@author: simon.kern
"""
import os
import numpy as np
from tdlm.utils import unique_permutations, seq2tf
from scipy.linalg import toeplitz
from numba import njit
from scipy.stats import zscore as zscore_func  # with alias to prevent clash

# some helper functions to make matlab work like python
ones = lambda *args, **kwargs: np.ones(shape=args, **kwargs)
zeros = lambda *args, **kwargs: np.zeros(shape=args, **kwargs)
nan = lambda *args: np.full(shape=args, fill_value=np.nan)
squash = lambda arr: np.ravel(arr, 'F')  # MATLAB uses Fortran style reshaping


# @profile
def pinv_(arr):
    """pseudoinverse, using optimization, up to 50% faster using JAX"""
    try:
        from absl import logging
        # jax is 50% faster, but only available on UNIX
        os.environ['JAX_PLAtfORMS'] = 'cpu'
        import jax
        logging.set_verbosity(logging.WARNING)
        jax.config.update('jax_platform_name', 'cpu')
        pinv = jax.numpy.linalg.pinv
        logging.set_verbosity(logging.INFO)

    except Exception:
        pinv = np.linalg.pinv
    return np.array(pinv(arr))


def find_betas(preds: np.ndarray, nstates: int, max_lag: int, alpha_freq=None):
    """for prediction matrix X (states x time), get transition matrix up to max_lag"""

    nbins = max_lag + 1;

    # design matrix is now a matrix of nsamples X (nstates*max_lag)
    # with each column a shifted version of the state vector (shape=nsamples)
    dm = np.hstack([toeplitz(preds[:, kk], [zeros(nbins, 1)])[:, 1:] for kk in range(nstates)])

    Y = preds;
    betas = nan(nstates * max_lag, nstates);

    ## GLM: state regression, with other lags
    bins = alpha_freq if alpha_freq else max_lag

    for ilag in list(range(bins)):
        # create individual GLMs for each time lagged version
        ilag_idx = np.arange(0, nstates * max_lag, bins) + ilag;
        # add a vector of ones for controlling the regression
        ilag_X = np.pad(dm[:, ilag_idx], [[0, 0], [0, 1]], constant_values=1)

        # add control for certain time lags to reduce alpha
        # Now find coefficients that solve the linear regression for this timelag
        # this a the second stage regression

        ilag_betas = pinv_(ilag_X) @ preds;  # if SVD fails, use slow, exact solution
        betas[ilag_idx, :] = ilag_betas[0:-1, :];

    return betas

@njit
def numba_roll(X, shift):
    """
    numba optimized np.roll function
    taken from https://github.com/tobywise/online-aversive-learning
    """
    # Rolls along 1st axis
    new_X = np.zeros_like(X)
    for i in range(X.shape[1]):
        new_X[:, i] = np.roll(X[:, i], shift)
    return new_X
    

@njit
def cross_correlation(X_data, transition_matrix, max_lag=40, min_lag=0):
    """
    Computes sequenceness by cross-correlation

    taken from https://github.com/tobywise/online-aversive-learning
    """
    X_dataf = X_data @ transition_matrix
    X_datar = X_data @ transition_matrix.T

    ff = np.zeros(max_lag - min_lag)
    fb = np.zeros(max_lag - min_lag)

    for lag in range(min_lag, max_lag):

        r = np.corrcoef(X_data[lag:, :].T, numba_roll(X_dataf, lag)[lag:, :].T)
        r = np.diag(r, k=transition_matrix.shape[0])
        forward_mean_corr = np.nanmean(r)

        r = np.corrcoef(X_data[lag:, :].T, numba_roll(X_datar, lag)[lag:, :].T)
        r = np.diag(r, k=transition_matrix.shape[0])
        backward_mean_corr = np.nanmean(r)

        ff[lag - min_lag] = forward_mean_corr
        fb[lag - min_lag] = backward_mean_corr

    return ff, fb


def compute_1step(preds, tf, tb=None, n_shuf=1000, min_lag=0, max_lag=50, 
                  alpha_freq=None,  cross_corr=False):
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
        The default is None.
    n_steps : int, optional
        number of transition steps to look for. Not implemented yet. 
        The default is 1.
    cross_corr : bool, optional
        Additionally to GLM analysis, perform cross correlation.
        The default is False.
    n_jobs : int, optional
        Number of parallel cores to use for some of the sub-analysis.
        The default is 1.
        
    Returns
    -------
    sf : np.ndarray
        forward sequencess for all time lags and shuffles. Row 0 is the 
        non-shuffled version
    sb : np.ndarray
        backward sequencess for all time lags and shuffles. Row 0 is the 
        non-shuffled version
    sf_corr : np.ndarray
        forward cross-correlation for all time lags and shuffles. Row 0 is the 
        non-shuffled version. None if no correlation is done
    sb_corr : np.ndarray
        backward cross-correlation for all time lags and shuffles. Row 0 is the 
        non-shuffled version. None if no correlation is done

    """
    
    nstates = preds.shape[-1]
    # unique permutations
    _, unique_perms, _ = unique_permutations(np.arange(1, nstates + 1), n_shuf) 
 
    
    seq_fwd = nan(n_shuf, max_lag + 1)  # forward sequenceness
    seq_bkw = nan(n_shuf, max_lag + 1)  # backward sequencenes
    seq_fwd_corr = nan(n_shuf, max_lag + 1)  # forward cross-correlation
    seq_bkw_corr = nan(n_shuf, max_lag + 1)  # backward cross-correlation

    if tb is None:
        # backwards is transpose of forwards
        tb = tf.T

    ## GLM: state regression, with other lags

    betas = find_betas(preds, nstates, max_lag, alpha_freq=alpha_freq)
    # betas = find_betas_optimized(X, nstates, max_lag, alpha_freq=alpha_freq)
    # np.testing.assert_array_almost_equal(betas, betas2, decimal= 12)

    # reshape the coeffs for regression to be in the order of ilag x (nstates x nstates)
    betasn_ilag_stage = np.reshape(betas, [max_lag, nstates ** 2], order='F');

    for i in range(n_shuf):
        rp = unique_perms[i, :]  # select next unique permutation of transitions
        tf_perm = tf[rp, :][:, rp]
        tb_perm = tb[rp, :][:, rp]              
        t_auto = np.eye(nstates)  # control for auto correlations
        t_const = np.ones([nstates, nstates])  # keep betas in same range

        # create our design matrix for the second step analysis
        dm = np.vstack([squash(tf_perm), squash(tb_perm), squash(t_auto), squash(t_const)]).T
        
        # now calculate regression coefs for use with transition matrix
        bbb = pinv_(dm) @ (betasn_ilag_stage.T)  #%squash(ones(nstates))

        seq_fwd[i, 1:] = bbb[0, :]  # forward coeffs
        seq_bkw[i, 1:] = bbb[1, :]  # backward coeffs

        # only calculate cross-correlation if requested, it's rather expensive to compute
        if not cross_corr: continue
        seq_fwd_corr[i, :-1], seq_bkw_corr[i, :-1] = cross_correlation(preds, tf, max_lag, min_lag)

    return seq_fwd, seq_bkw, seq_fwd_corr, seq_bkw_corr


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
