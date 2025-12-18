# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 15:10:07 2024

util functions for Temporally Delayed Linear Modelling

@author: simon.kern
"""
import hashlib
import math
import numpy as np
import pandas as pd
import warnings
from numpy.linalg import pinv, eigh
from scipy.signal import lfilter
from itertools import permutations
from numba import njit

def hash_array(arr, dtype=np.int64, truncate=8):
    """
    create a persistent hash for a numpy array based on the byte representation
    only the last `truncate` (default=8) characters are returned for simplicity

    Parameters
    ----------
    arr : np.ndarray
        DESCRIPTION.
    dtype : type, optional
        which data type to use. smaller type will be faster.
        The default is np.int64.

    Returns
    -------
    str
        unique hash for that array.

    """
    arr = arr.astype(dtype)
    sha1_hash = hashlib.sha1(arr.flatten("C")).hexdigest()
    return sha1_hash[:truncate]


def _trans_overlap(seq1=None, seq2=None, trans1=None, trans2=None):
    """
    calculate how many overlapping 1 step transitions exist between
    seq1 and seq2. For optimization reasons instead of the sequence, already
    computed transitions can also be supplied

    """
    if trans1 is None:
        trans1 = set(zip(seq1[:-1], seq1[1:]))
    if trans2 is None:
        trans2 = set(zip(seq2[:-1], seq2[1:]))
    return len(trans1.intersection(trans2))


def unique_permutations(X, k=None, max_true_trans=None):
    """"""
    X = np.array(X).squeeze()
    assert X.ndim==1
    assert len(X) > 1

    uniques, uind, c = np.unique(X, return_index=True, return_counts=True)

    max_perms = math.factorial(len(uniques))

    if k is None:
        k = max_perms;  # default to computing all unique permutations

    if  k > max_perms:
        raise ValueError(f'requested {k=} larger than all possible permutations {max_perms=}')


    # enumerate all transitions in case max_overlap is set
    trans = set(zip(X[:-1], X[1:]))

    if k is None:
        # if all permutations are requested, fastest way is to simply enumerate
        generator = permutations(X)
        # make sure the non-permuted version in position 0
        uperms = [next(generator)]
        uperms += set(permutations(X))
        if max_true_trans:
            uperms_filtered = []
            for perm in uperms[1:]:
                if _trans_overlap(seq1=perm, trans2=trans)<=max_true_trans:
                    uperms_filtered += [perm]

    # if upper bound is given, draw valid samples until we have enough
    else:
        seq = tuple(X)
        uperms = set([seq])

        # need to store the discarded items to prevent running into a loop
        if max_true_trans is not None:
            discarded = set()

        # add permutations to the set until we reach k
        while len(uperms) < k:
            perm = tuple(np.random.permutation(seq))

            # only add if it contains non-true transitions
            if max_true_trans is not None:
                if _trans_overlap(seq1=perm, trans2=trans)>max_true_trans:
                    discarded.add(perm)
                    continue
                if (len(discarded.union(uperms))) == max_perms:
                    warnings.warn(f'Fewer valid permutations {len(uperms)=} possible than {k=} requested')
                    break
            uperms.add(perm)

        # make sure the non-permuted version in position 0
        uperms.remove(seq)
        all_perms = list([seq])
        all_perms += uperms

    return np.array(all_perms)



def _unique_permutations_MATLAB(X, k=None):
    """
    DEPRECATED

    original implementation of the unique permutation function from MATLAB

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

    max_perms = math.factorial(len(uniques))

    if k is None:
        k = max_perms;  # default to computing all unique permutations

    if  k > max_perms:
        raise ValueError('requested {k=} larger than all possible permutations')

    uniques = uniques.tolist()
    x = np.array([uniques.index(i) for i in X.tolist()])

    c = sorted(c)
    nPerms = np.prod(np.arange(c[-1] + 1, np.sum(c) + 1)) / np.prod([math.factorial(x) for x in c[:-1]])
    nPerms = int(nPerms)
    # % computation of permutation
    # Basics
    n = len(X);


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


def tf2seq(transition_matrix):
    """
    Convert a transition matrix into a sequence string.
    If there are disjoint sequences, separate them with "_".

    :param transition_matrix: A square numpy array representing the transition matrix.
                              Each row should have at most one outgoing transition (1).
    :return: A string representing the sequence(s), e.g., "ABC_DEF".
    """
    n_states = transition_matrix.shape[0]
    visited = set()
    sequences = []

    def find_sequence(start_state):
        """Helper function to find a sequence starting from a given state."""
        sequence = []
        current_state = start_state
        while current_state not in visited:
            sequence.append(current_state)
            visited.add(current_state)
            next_state = np.argmax(transition_matrix[current_state])  # Find the next state
            if transition_matrix[current_state, next_state] == 0:  # No valid transition
                break
            current_state = next_state
        return sequence

    # Iterate through all states to find disjoint sequences
    for state in range(n_states):
        if state not in visited and np.sum(transition_matrix[state]) > 0:  # Unvisited and has outgoing transitions
            sequence = find_sequence(state)
            sequences.append(sequence)

    # Convert numeric sequences to character sequences and join with "_"
    sequence_strings = []
    for sequence in sequences:
        sequence_str = ''.join(chr(65 + state) for state in sequence)
        sequence_strings.append(sequence_str)

    return '_'.join(sequence_strings)


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
    e.g. ABCDEFGE.  AB->C BC->D ..."""
    import pandas as pd
    triplets = []
    if n_states is None:
        n_states = max(char2num(seq))+1
    TF2 = np.zeros([n_states**2, n_states], dtype=int)
    for i, p1 in enumerate(seq):
        if i+2>=len(seq): continue
        if p1=='_': continue
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


def simulate_meeg(length, sfreq, n_channels=64, cov=None, autocorr=0.95, rng=None):
    """
    Simulate M/EEG resting-state data.

    Parameters
    ----------
    length : float
        Total duration of the signal in seconds.
    sfreq : float
        Sampling frequency in Hz (samples per second).
    n_channels : int, optional
        Number of EEG channels (default is 64).
    cov : numpy.ndarray, optional
        Covariance matrix of shape (n_channels, n_channels).
        If None, a random covariance matrix is generated.
    autocorr : float, optional
        Temporal correlation of each sample with its neighbour samples.
    rng : numpy.random.Generator, optional
        Random number generator.

    Returns
    -------
    eeg_data : numpy.ndarray
        Simulated EEG data of shape (n_samples, n_channels).
    """
    assert 0 <= autocorr < 1
    n_samples = int(length * sfreq)
    rng = np.random.default_rng(rng)

    # 1. Setup Covariance (Cholesky Decomposition)
    if cov is None:
        A = rng.normal(size=(n_channels, n_channels))
        # Create symmetric positive-definite matrix
        cov = (A + A.T) / 2
        _, U = np.linalg.eig(cov)
        # Reconstruct with positive eigenvalues
        cov = U @ np.diag(np.abs(rng.normal(size=n_channels))) @ U.T
    else:
        n_channels = len(cov)

    # Compute Mixing Matrix (L) from Covariance
    # We use Cholesky: Cov = L @ L.T
    # If Cov is not strictly positive definite, use SVD as fallback (omitted for speed)
    L = np.linalg.cholesky(cov)

    # 2. Generate White Noise (Standard Normal)
    Z = rng.standard_normal((n_samples, n_channels))

    # 3. Apply Temporal Filter to White Noise
    # Original logic: noise was scaled by autocorr before addition
    # Filter: y[n] = autocorr * y[n-1] + (autocorr * x[n])
    # To match original magnitude logic: Scale input noise by autocorr
    Z *= autocorr

    # Correction for initial state to ensure X[0] ~ Standard Normal before mixing
    # We want X[0] to be pure noise, not filtered.
    # Current Z[0] is random. The filter will effectively do Z[0] = Z[0].
    # So X[0] will be correct.

    # Apply Filter along time axis (axis 0)
    # b=[1], a=[1, -autocorr]
    # We use zi to handle initial conditions smoothly if needed,
    # but strictly Z starts random, so standard filter is fine.
    Z = lfilter([1], [1, -autocorr], Z, axis=0)

    # 4. Apply Spatial Mixing (Matrix Multiplication)
    # X = Z_filtered @ L.T
    # This moves the heavy O(N*M^2) operation to a single highly optimized BLAS call
    X = Z @ L.T

    return X

def simulate_classifier_patterns(n_patterns=10, n_channels=306, noise=4,
                                 scale=1,  n_train_per_stim=18, rng=None):
    """
    Generates synthetic training data and labels matching MATLAB TDLM logic.

    Parameters
    ----------
    n_patterns : int
        Number of unique stimulus patterns.
    n_channels : int
        Number of sensor channels.
    noise : float
        Standard deviation of background noise.
    n_train_per_stim : int
        Repetitions per stimulus.
    rng : int or np.random.Generator, optional
        Random int seed or generator.

    Returns
    -------
    training_data : np.ndarray
        Simulated sensor data.
    training_labels : np.ndarray
        Labels (0 for null, 1-N for stimuli).
    patterns : np.ndarray
        Ground truth patterns.
    """
    rng = np.random.default_rng(rng)

    # Setup dimensions
    n_null = n_train_per_stim * n_patterns
    n_stim_total = n_patterns * n_train_per_stim
    n_total = n_null + n_stim_total

    # Generate Patterns
    common_pattern = rng.normal(size=(1, n_channels))
    patterns = np.tile(common_pattern, (n_patterns, 1)) + \
               rng.standard_normal((n_patterns, n_channels))

    # Construct Base Data
    base_noise = noise * rng.standard_normal((n_total, n_channels))
    stim_signal = np.tile(patterns, (n_train_per_stim, 1))

    # Stack Null (zeros) on top of Stim (signal)
    signal_component = np.vstack([
        np.zeros((n_null, n_channels)),
        stim_signal
    ])

    training_data = base_noise + signal_component

    # Generate Labels
    stim_labels = np.tile(np.arange(1, n_patterns + 1), n_train_per_stim)
    training_labels = np.concatenate([
        np.zeros(n_null, dtype=int),
        stim_labels
    ])

    # Inject Extra Noise to half the patterns
    n_noise_groups = n_patterns // 2

    if n_noise_groups > 0:
        more_noise_inds = rng.choice(np.arange(1, n_patterns + 1),
                                     size=n_noise_groups,
                                     replace=False)

        for idx in more_noise_inds:
            # MATLAB 1-based logic conversion: (Ind-1)*N + 1 : Ind*N
            start_rel = (idx - 1) * n_train_per_stim
            end_rel = idx * n_train_per_stim

            s_idx = n_null + start_rel
            e_idx = n_null + end_rel

            segment_len = e_idx - s_idx
            training_data[s_idx:e_idx, :] += rng.standard_normal((segment_len, n_channels))

    return training_data*scale, training_labels, patterns*scale



def insert_events(data, insert_data, insert_labels, n_events, lag=8, jitter=0,
                  n_steps=1, refractory=16, distribution='constant',
                  transitions=None, sequence=None, return_onsets=False, rng=None):
    """
    inject decodable events into M/EEG data according to a certain pattern.


    Parameters
    ----------
    data : TYPE
        DESCRIPTION.
    insert_data : np.ndarray
        data that should be inserted. Length must be the same as insert_labels.
        Must be 2D, with the second dimension being the sensor dimension.
        If insert_data is 3D, last dimension is taken as a time dimension
    insert_labels : np.ndarray
        list of class labels/ids for the insert_data.
    mean_class: bool
        insert the mean of the class if True, else insert a random single event
        from insert_data.
    lag : TYPE, optional
        Sample space distance individual reactivation events events.
        The default is 7 (e.g. 70 ms replay speed time lag).
    jitter : int, optional
        By how many sample points to jitter the events (randomly).
        The default is 0.
    refractory: int | list of two int
        how many samples of blocking there should be before and after each
        sequence start and sequence end. If integer, apply same to both sides.
        If list of two ints, interpret as left and right blocking period.
        If set to None, disregard and allow overlapping sequences.
    transitions: list of list
        the sequence transitions that should be sampled from.
        if it is a 1d list, transitions will be extracted automatically
    n_steps : int, optional
        Number of events to insert. The default is 2
    distribution : str | np.ndarray, optional
        How replay events should be distributed throughout the time series.
        Can either be 'constant', 'increasing' or 'decreasing' or a p vector
        with probabilities for each sample point in data.
        The default is 'constant'.
    rng : np.random.Generator | int
        random generator or integer seed

    Returns
    -------
    data : np.ndarray (shape=data.shape)
        data with inserted events.
    (optional) return_onsets: pd.DataFrame
        table with onsets of events
    """
    if isinstance(insert_labels, list):
        insert_labels = np.array(insert_labels)
    if isinstance(insert_data, list):
        insert_data = np.array(insert_data)
    import logging
    assert len(insert_data) == len(insert_labels), 'each data point must have a label'
    assert insert_data.ndim in [2, 3]
    assert insert_labels.ndim == 1
    assert data.ndim==2
    assert data.shape[1] == insert_data.shape[1]
    assert min(insert_labels)==0, 'insert_labels must start at 0 and be consecutive'

    if isinstance(distribution, np.ndarray):
        assert len(distribution) == len(data)
        assert distribution.ndim == 1
        assert np.isclose(distribution.sum(), 1), 'Distribution must sum to 1, but {distribution.sum()=}'

    assert sequence is None or transitions is None, 'either sequence or transitions must be supplied'

    # no events requested? simply return
    if not n_events:
        return (data, pd.DataFrame()) if return_onsets else data

    # assume refractory period is valid for both sides
    if isinstance(refractory, int):
        refractory = [refractory, refractory]

    if sequence is not None:
        transitions = [sequence[i:i+n_steps+1]for i, _ in enumerate(sequence[:-n_steps])]
        assert len(transitions) == len(sequence)-n_steps, 'sanity check failed'

    transitions = np.squeeze(transitions)

    assert transitions.shape[1]==n_steps+1, \
        f'each transition must have exactly {n_steps}+1 steps'

    del sequence # for safety, can be removed later

    # convert data to 3d
    if insert_data.ndim==2:
        insert_data = insert_data.reshape([*insert_data.shape, 1])

    # work on copy of array to prevent mutable changes
    data_sim = data.copy()

    # get reproducible seed
    rng = np.random.default_rng(rng)

    # Calculate probability distribution based on the specified distribution type
    if isinstance(distribution, str):
        if distribution=='constant':
            p = np.ones(len(data))
            p = p/p.sum()
        elif distribution=='decreasing':
            p = np.linspace(1, 0, len(data))**2
            p = p/p.sum()
        elif distribution=='increasing':
            p = np.linspace(0, 1, len(data))**2
            p = p/p.sum()
        else:
            raise ValueError(f'unknown {distribution=}')
    elif isinstance(distribution, (list, np.ndarray)):
        distribution = np.array(distribution)
        assert len(distribution)==len(data), f'{distribution.shape=} != {len(data)=}'
        assert np.isclose(distribution.sum(), 1), f'{distribution.sum()=} must be sum=1'
        p = distribution
    else:
        raise ValueError(f'distribution must be string or p-vector, {distribution=}')

    # block impossible starting points (i.e. out of bounds)
    tspan = insert_data.shape[-1] # timespan of one patter
    event_length = n_steps*lag + tspan -1  # time span of one replay events
    p[-event_length:] = 0  # dont start events at end of resting state
    p[:tspan] = 0  # block beginning of resting state
    p = p/p.sum()  # normalize probability vector again after removing indices

    replay_start_idxs = []
    all_idx = np.arange(len(data))

    # iteratively select starting index for replay event
    # such that replay events are not overlapping
    for i in range(n_events):

        # Choose a random idx from the available indices to start replay event
        start_idx = rng.choice(all_idx, p=p)
        replay_start_idxs.append(start_idx)



        # Update the p array to zero out the region around the chosen index to prevent overlap
        if refractory is not None:
            # next block the refractory period to prevent overlap
            block_start = start_idx - refractory[0]
            block_end   = start_idx + refractory[1] + 1

            block_start = max(block_start, 0)
            block_end = min(block_end, len(p))

            p[block_start: block_end] = 0

            # normalize to create valid probability distribution
            p = p/p.sum()

        # check that we actually still have enough positions to insert
        # another event of length lag*n_steps. Probably the function fails
        # beforehand though.
        if (p>0).sum() < n_steps*lag:
            raise ValueError(f'no more positions to insert events! {n_events=} too high?')



    # save data about inserted events here and return if requested
    events = {'event_idx': [],
              'pos': [],
              'step': [],
              'class_idx': [],
              'span': [],
              'jitter': []}

    for idx,  start_idx in enumerate(replay_start_idxs):
        smp_jitter = 0  # starting with no jitter
        pos = start_idx  # pos indicates where in data we insert the next event

        # randomly sample a transition that we would take
        trans = rng.choice(transitions)

        for step, class_idx  in enumerate(trans):
            # choose which item should be inserted based on sequence order
            # or take a single event (more noisy)
            data_class = insert_data[insert_labels==class_idx]
            idx_cls_i = rng.choice(np.arange(len(data_class)))
            insert_data_i = data_class[idx_cls_i]
            assert insert_data_i.ndim==2

            # time spans of the segments we want to insert
            t_start = pos - tspan // 2
            t_end = t_start + tspan
            data_sim[t_start:t_end, :] += insert_data_i.T
            logging.debug(f'{start_idx=} {pos=} {class_idx=}')

            events['event_idx'] += [idx]
            events['pos'] += [pos]
            events['step'] += [step]
            events['class_idx'] += [class_idx]
            events['span'] += [insert_data_i.shape[-1]]
            events['jitter'] += [smp_jitter]

            # increment pos to select position of next reactivation event
            smp_jitter = rng.integers(-jitter, jitter+1) if jitter else 0
            pos += lag + smp_jitter  # add next sequence step

    if return_onsets:
        df_onsets = pd.DataFrame(events)
        df_onsets['n_events'] = n_events
        return (data_sim, df_onsets)

    return data_sim


def create_travelling_wave(hz, seconds, sfreq, chs_pos, source_idx=0, speed=50):
    """
    Create a sinus wave of shape (size, len(sensor_pos)), where each
    entry in the second dimension is phase shifted according to propagation
    speed and the euclidean distance between sensor positions.

    Parameters
    ----------
    hz : float
        The frequency of the sinus curve in Hz.
    sfreq : int
        The sampling rate of the signal in Hz.
    chs_pos : np.array or list
        A list of 2d sensor/channel positions [(x, y), ...], with coordinates
        given in cm. Phase shift of the wave will be calculated according to
        the euclidean distance between sensors/channels.
    source_idx : int, optional
        Index of the sensor/channel at which the oscillation should start
        with phase 0 and travel from there to all other positions.
    speed : float, optional
        Speed of wave in m/second. The default is 0.5m/second which is
        a good average for alpha waves.

    Returns
    -------
    wave : np.ndarray
        Array of shape (size, len(sensor_pos)) representing the travelling wave.
    """
    if speed == 0 :
        speed = np.inf

    speed = speed * 100  # convert to cm/s, as positions are in cm
    # Convert sensor_pos to a numpy array if it's not already
    chs_pos = np.array(chs_pos)

    # Number of sensors
    n_sensors = len(chs_pos)
    size = int(seconds * sfreq)

    # Time array
    t = np.arange(size) / sfreq

    # Initialize wave array
    wave = np.zeros((size, n_sensors))

    # Calculate distances from the source sensor to all other sensors
    distances = np.linalg.norm(chs_pos - chs_pos[source_idx], axis=1)

    # Calculate the time delays for each sensor
    time_delays = distances / (speed)

    # Generate the sinusoidal wave for each sensor with the corresponding phase shift
    for i in range(n_sensors):
        wave[:, i] = np.sin(2 * np.pi * hz * (t - time_delays[i]))

    return wave


if __name__=='__main__':
    import seaborn as sns
    import stimer
    with stimer:
        x1 = simulate_meeg(600, 100, 306, rng=0)
    with stimer:
        x2 = simulate_meeg_fast(600, 100, 306, rng=0)
