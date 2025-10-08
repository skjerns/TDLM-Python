# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 15:10:07 2024

core functions for Temporally Delayed Linear Modelling

@author: simon.kern
"""
import os
import numpy as np
import logging
from tdlm.utils import unique_permutations, seq2tf, tf2seq
from tdlm import utils
from scipy.linalg import toeplitz
from numba import njit
from scipy.stats import zscore as zscore_func  # with alias to prevent clash
from scipy.stats import ttest_1samp
from joblib import Parallel, delayed
from numpy.linalg import pinv

# try:
    # from jax.numpy.linalg import pinv
# except ModuleNotFoundError:
#     logging.warning('jaxlib not installed, can speed up computation')

# some helper functions to make matlab work like python
ones = lambda *args, **kwargs: np.ones(shape=args, **kwargs)
zeros = lambda *args, **kwargs: np.zeros(shape=args, **kwargs)
nan = lambda *args: np.full(shape=args, fill_value=np.nan)
squash = lambda arr: np.ravel(arr, 'F')  # MATLAB uses Fortran style reshaping



def _find_betas(probas: np.ndarray, n_states: int, max_lag: int, alpha_freq=None):
    """for prediction matrix X (states x time), get transitions up to max_lag.
    Similar to cross-correlation, i.e. shift rows of matrix iteratively

    paralellizeable version
    """
    n_bins = max_lag + 1;

    # design matrix is now a matrix of nsamples X (n_states*max_lag)
    # with each column a shifted version of the state vector (shape=nsamples)
    dm = np.hstack([toeplitz(probas[:, kk].ravel(),
                             np.ravel([zeros(n_bins, 1)]))[:, 1:]
                    for kk in range(n_states)])

    betas = nan(n_states * max_lag, n_states);

    ## GLM: state regression, with other lags
    #TODO: Check if this does what is expected
    bins = alpha_freq if alpha_freq else max_lag

    for ilag in list(range(bins)):
        # create individual GLMs for each time lagged version
        ilag_idx = np.arange(0, n_states * max_lag, bins) + ilag;
        # add a vector of ones for controlling the regression
        ilag_X = np.pad(dm[:, ilag_idx], [[0, 0], [0, 1]], constant_values=1)

        # add control for certain time lags to reduce alpha
        # Now find coefficients that solve the linear regression for this timelag
        # this a the second stage regression
        # print(ilag_X.shape)
        ilag_betas = pinv(ilag_X) @ probas;  # if SVD fails, use slow, exact solution
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
def _cross_correlation(probas, tf, tb, max_lag=40, min_lag=0):
    """
    Computes sequenceness by cross-correlation

    taken from https://github.com/tobywise/online-aversive-learning
    """
    probas_f = probas @ tf
    probas_b = probas @ tb

    ff = np.zeros(max_lag - min_lag)
    fb = np.zeros(max_lag - min_lag)

    for lag in range(min_lag, max_lag):

        r = np.corrcoef(probas[lag:, :].T, _numba_roll(probas_f, lag)[lag:, :].T)
        r = np.diag(r, k=tf.shape[0])
        forward_mean_corr = np.nanmean(r)

        r = np.corrcoef(probas[lag:, :].T, _numba_roll(probas_b, lag)[lag:, :].T)
        r = np.diag(r, k=tb.shape[0])
        backward_mean_corr = np.nanmean(r)

        ff[lag - min_lag] = forward_mean_corr
        fb[lag - min_lag] = backward_mean_corr

    return ff, fb


def signflit_test(sx, n_perms=1000, rng=None):
    """run signflip permutation test to check for significant sequenceness"""
    assert sx.ndim==2, 'sx must be 2D'
    t_maxes = []
    t_true = ttest_1samp(sx, axis=0, popmean=0)[1].max()
    for i in range(n_perms):
        bits = np.random.choice([-1, 1], size=(len(sx)))
        sx_perm = (sx.T*bits).T
        t_perm = ttest_1samp(sx_perm, axis=0, popmean=0)[1]
        t_maxes += [np.max(t_perm)]
    p = (t_true>t_maxes).mean()
    return p, t_true, t_maxes


def sequenceness_crosscorr(probas, tf, tb=None, n_shuf=1000, min_lag=0, max_lag=50,
                           alpha_freq=None):

    n_states = probas.shape[-1]
    # unique permutations
    unique_perms = unique_permutations(np.arange(1, n_states + 1), n_shuf)


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
        seq_fwd_corr[i, :-1], seq_bkw_corr[i, :-1] = _cross_correlation(probas,
                                                                        tf_perm,
                                                                        tb_perm,
                                                                        max_lag=max_lag,
                                                                        min_lag=min_lag)
    return seq_fwd_corr, seq_bkw_corr




# @profile
def compute_1step(probas, tf, tb=None, n_shuf=100, min_lag=0, max_lag=50,
                  alpha_freq=None, max_true_trans=None, seed=None):
    """
    Calculate 1-step-sequenceness for probability estimates and transitions.

    Parameters
    ----------
    probas : np.ndarray
        2d matrix with predictions, shape= (timesteps, n_states), where each
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
        steps of the probas time dimension.
    alpha_freq : int, optional
        Alpha oscillation frequency to control for. Time shifted copies of the
        signal are added in this frequency to the GLM, acting as a confounds.
        Warning: Must be supplied in sample points, not in Hertz!
        The default is None.
    max_true_trans : int, optional
        Maximum number of transitions that should be be overlapping between the
        real sequence and shuffles. E.g. if your sequence is A->B->C, the
        permutation B->C->A would contain one overlapping transition B->C.
        Setting max_true_trans=0 would remove this permutation from the test.
        The default is None, i.e. no limit.
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
    # implicit conversion off probability lists to arrays
    probas = np.array(probas)

    # checks and balances
    assert probas.ndim==2, 'probas must be 2d but is {probas.ndim=}'
    assert tf.ndim==2, f'transition matrix must be 2d but is {tf.ndim=}'
    assert tf.shape[0]==tf.shape[1], f'transition matrix must be square {tf.shape=}'
    assert len(tf)==probas.shape[1], f'{len(tf)=} must be same as {probas.shape[1]}'

    if seed is not None:
        np.random.seed(seed)
    n_states = probas.shape[-1]
    # unique permutations
    unique_perms = unique_permutations(np.arange(n_states), n_shuf,
                                       max_true_trans=max_true_trans)

    n_perms = len(unique_perms)  # this might be different to requested n_shuf!

    seq_fwd = nan(n_perms, max_lag + 1)  # forward sequenceness
    seq_bkw = nan(n_perms, max_lag + 1)  # backward sequencenes

    if tb is None:
        # backwards is transpose of forwards
        tb = tf.T

    ## GLM: state regression, with other lags

    betas = _find_betas(probas, n_states, max_lag, alpha_freq=alpha_freq)
    # betas = find_betas_optimized(X, n_states, max_lag, alpha_freq=alpha_freq)
    # np.testing.assert_array_almost_equal(betas, betas2, decimal= 12)

    # reshape the coeffs for regression to be in the order of ilag x (n_states x n_states)
    betasn_ilag_stage = np.reshape(betas, [max_lag, n_states ** 2], order='F');

    for i in range(n_perms):
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


def compute_2step(probas, tf, tb=None, n_steps=2, n_shuf=None, min_lag=0, max_lag=50,
                  alpha_freq=None, seed=None):
    """
    # 2step tdlm version. for now this is a copy of the MATLAB code, did not
    have time yet to implement the generalized version.

    I do think there are conceptual problems with this implementation,
    therefore, I do not recommend using the method without further consideration
    e.g. if our data contains A->B, but never C, we will _still_ find backwards
    sequenceness evidence simply because (C*B) is regressed on A for the back-
    wards case and will induce spurious sequenceness of A->B->C when there is
    no triplet replay
    """
    if seed is not None:
        np.random.seed(seed)
    assert n_steps==2, " >2 steps is not implemented yet"

    # seq = tf2seq(tf)
    n_states = probas.shape[-1]

    unique_perms = unique_permutations(np.arange(n_states), n_shuf)

    n_perms = len(unique_perms)  # this might be different to requested n_shuf!

    # create all two step transitions from our transition matrix
    tf_y = []
    tf_x2 = []
    tf_x1 = []

    seq_starts = np.where(tf)
    for x1, x2 in zip(*seq_starts):
        for y in np.where(tf[x2, :])[0]:
            tf_x1 += [x1]
            tf_x2 += [x2]
            tf_y  += [y]

    tr_y = tf_x1
    tr_x2 = tf_x2
    tr_x1 = tf_y

    tf2 = zeros(len(tf_y), n_states);
    tf_auto = zeros(len(tf_y), n_states);

    for i in range(len(tf_y)):
        tf2[i,tf_y[i]] = 1;
        tf_auto[i, np.unique([tf_x1[i], tf_x2[i]])]=1;

    tr2 = zeros(len(tr_y), n_states);
    tr_auto = zeros(len(tr_y), n_states);
    for i in range(len(tr_y)):
        tr2[i, tr_y[i]] = 1;
        tr_auto[i, np.unique([tr_x1[i], tr_x2[i]])]=1;


    x2_bin = np.full([max_lag, len(probas)] + [n_states] * n_steps, np.nan)

    # Initialize variables
    x = probas
    y = x

    # First loop
    for lag in range(1, max_lag + 1):
        pad = np.zeros((lag, n_states))
        x1 = np.vstack([pad, pad, x[:-2 * lag, :]])
        x2 = np.vstack([pad, x[:-lag, :]])

        for i in range(n_states):
            for j in range(n_states):
                x2_bin[lag - 1, :, i, j] = x1[:, i] * x2[:, j]

    beta_f = np.full((max_lag, len(tf_y), n_states), np.nan)
    beta_b = np.full((max_lag, len(tr_y), n_states), np.nan)

    # Second loop
    for lag in range(max_lag):
        x_matrix = x2_bin[lag, :, :, :]

        for state in range(len(tf_y)):
            x_fwd = x_matrix[:, :, tf_x2[state]]
            x_bkw = x_matrix[:, :, tr_x2[state]]

            temp_f = pinv(np.hstack([x_fwd, np.ones((len(x_fwd), 1))])) @ y
            beta_f[lag, state, :] = temp_f[tf_x1[state], :]

            temp_b = pinv(np.hstack([x_bkw, np.ones((len(x_bkw), 1))])) @ y
            beta_b[lag, state, :] = temp_b[tr_x1[state], :]

    beta_f = beta_f.reshape((max_lag, len(tf_y) * n_states), order='F')
    beta_b = beta_b.reshape((max_lag, len(tr_y) * n_states), order='F')

    seq_fwd = nan(n_perms, max_lag+1);
    seq_bkw = nan(n_perms, max_lag+1);

    # Third loop
    for shuffle_idx in range(n_perms):
        random_permutation = unique_perms[shuffle_idx, :]

        # 2nd level
        const_f = np.ones((len(tf_y), n_states))
        const_r = np.ones((len(tr_y), n_states))

        tf_shuffled = tf2[:, random_permutation]
        tr_shuffled = tr2[:, random_permutation]

        cc = pinv(np.hstack([
            squash(tf_shuffled)[:, None],
            squash(tf_auto)[:, None],
            squash(const_f)[:, None]
        ])) @ beta_f.T
        seq_fwd[shuffle_idx, 1:] = cc[0, :]

        cc = pinv(np.hstack([
            squash(tr_shuffled)[:, None],
            squash(tr_auto)[:, None],
            squash(const_r)[:, None]
        ])) @ beta_b.T
        seq_bkw[shuffle_idx, 1:] = cc[0, :]
    return seq_fwd, seq_bkw


#%% __main__ -  quick debugging
if __name__=='__main__':
    import stimer
    import mat73

    data = mat73.loadmat('./tests/matlab_code/simulate_replay_longerlength_results.mat')
    probas = data['preds']
    tf = data['TF']
    sf_matlab = data['sf1'].squeeze()
    sb_matlab = data['sb1'].squeeze()
    max_lag = int(data['maxLag'])
    n_shuf = int(data['nShuf'])
    unique_perms = data['uniquePerms']-1

    # monkey patch uperms, to give equivalent results to MATLAB
        # print(tdlm.utils.unique_permutations([1,2,3]))
    with stimer:
        sf, sb = compute_2step(probas, tf, max_lag=max_lag,
                                    n_shuf=n_shuf)
