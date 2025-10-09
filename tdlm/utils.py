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


def simulate_meeg(length, sfreq, n_channels=64, cov=None, autocorr=0.95):
    """
    Simulate M/EEG resting-state data.

    Parameters:
    - length: float
        Total duration of the signal in seconds.
    - sfreq: float
        Sampling frequency in Hz (samples per second).
    - n_channels: int, optional
        Number of EEG channels (default is 64).
    - cov: numpy.ndarray, optional
        Covariance matrix of shape (n_channels, n_channels).
        If None, a random covariance matrix is generated
    - autocorr: float, optional
        temporal correlation of each sample with its neighbour samples in time.

    this code is loosely based but optimized version of
         https://github.com/YunzheLiu/TDLM/blob/master/Simulate_Replay.m

    Returns:
    - eeg_data: numpy.ndarray
        Simulated EEG data of shape (n_samples, n_channels).
    """
    assert 0<=autocorr<1

    n_samples = int(length * sfreq)  # Total number of samples

    if str(type(cov))=="<class 'mne.cov.Covariance'>":
        # extract cov from mne Covariance object
        cov = cov.data

    if cov is not None and n_channels is not None:
        assert len(cov)==n_channels, \
            'n_channels must be the same as covariance size'

    # If covariance matrix is not provided, generate a random one
    if cov is None:
        # Generate a random symmetric covariance matrix
        A = np.random.randn(n_channels, n_channels)
        symA = (A + A.T) / 2  # Symmetrize to make it symmetric
        # Eigen decomposition to ensure positive semi-definite
        eigenvalues, eigenvectors = np.linalg.eigh(symA)
        # Adjust eigenvalues to be positive
        eigenvalues = np.abs(eigenvalues) + 0.1
        cov = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    else:
        n_channels = len(cov)

    # Initialize the EEG data array
    eeg_data = np.zeros((n_samples, n_channels))

    # Generate initial sample
    eeg_data[0, :] = np.random.multivariate_normal(np.zeros(n_channels), cov)

    # precompute noise by drawing a noise vector for each channel across time
    noise = np.random.multivariate_normal(np.zeros(n_channels), cov,
                                          size=n_samples)

    b = [1.0]           # numerator
    a = [1.0, -autocorr]  # denominator

    # Use scipy.signal.lfilter for each channel (C-accelerated time loop).
    # Provide the initial sample as part of the filter's state so that x(1) = autocorr*x(0) + noise(1).
    for ch in range(n_channels):
        # "zi" is the filter state; we set it so that the very first sample equals eeg_data[0, ch].
        # The shape of zi must be (max(len(a), len(b)) - 1) = 1 for AR(1).
        channel_noise = noise[1:, ch]  # we skip noise(0) because x(0) is already set

        # The "zi" (initial filter state) must produce x(1) = autocorr*x(0) + e(1).
        # With an AR(1), the length of zi = max(len(a), len(b)) - 1 = 1.
        # When using lfilter([1], [1, -r], x, zi=[z]),
        # the first output sample is y(0) = b[0]*x(0) + z = x(0) + z.
        #
        # We need y(0) = x(1) = autocorr*x(0) + noise(1).
        # So z = autocorr*x(0).
        zi = [autocorr * eeg_data[0, ch]]

        # Apply filtering to get x(1..n_samples-1).
        channel_out, _ = lfilter(b, a, channel_noise, zi=zi)

        # Store the result into [1..n_samples-1] for the channel
        eeg_data[1:, ch] = channel_out
    return eeg_data


def simulate_eeg_localizer(n_samples, n_classes, noise=1.0, n_channels=64):
    raise NotImplementedError()


def insert_events(data, insert_data, insert_labels, sequence, n_events,
                  lag=7, jitter=0, n_steps=2,  distribution='constant',
                  return_onsets=False, rng=None):
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
    n_steps : int, optional
        Number of events to insert. The default is 2
    distribution : str | np.ndarray, optional
        How replay events should be distributed throughout the time series.
        Can either be 'constant', 'increasing' or 'decreasing' or a p vector
        with probabilities for each sample point in data.
        The default is 'constant'.
    rng : RandomState | int
        random state or integer seed
    Raises
    ------
    ValueError
        DESCRIPTION.
    Exception
        DESCRIPTION.

    Returns
    -------
    data : np.ndarray (shape=data.shape)
        data with inserted events.
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

    if isinstance(distribution, np.ndarray):
        assert len(distribution) == len(data)
        assert distribution.ndim == 1
        assert np.isclose(distribution.sum(), 1), 'Distribution must sum to 1, but {distribution.sum()=}'

    # convert data to 3d
    if insert_data.ndim==2:
        insert_data = insert_data.reshape([*insert_data.shape, 1])

    # work on copy of array to prevent mutable changes
    data_sim = data.copy()

    # get reproducible seed
    rng = np.random.default_rng(rng)

    # Define default parameters for replay generation
    # defaults = {'dist':7,
    #             'n_events':250,
    #             'tp':31,
    #             'seqlen':3,
    #             'direction':'fwd',
    #             'distribution':'constant',
    #             'trange': 0}

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

    # Calculate length of the replay sequence
    padding = n_steps*lag + data.shape[-1]
    p[-padding:] = 0  # can't insert events starting here, would be too long
    p = p/p.sum()

    replay_start_idxs = []
    all_idx = np.arange(len(data))

    # iteratively select starting index for replay event
    # such that replay events are not overlapping
    for i in range(n_events):
        # next set all indices of p to zero where events will be inserted
        # this way we can prevent overlap of replay event trains
        # Find available indices where events can be inserted
        available_indices = np.where(p > 0)[0]

        # Ensure that there are enough available indices to choose from
        if len(available_indices) < n_events - i:
            raise ValueError(f"Not enough available indices to insert all events without overlap, {n_events=} too high")

        # Choose a random index from the available indices
        start_idx = rng.choice(all_idx, p=p)

        # this is the calculated end index
        end_idx = start_idx + lag * n_steps + insert_data.shape[-1]
        assert end_idx<len(p)
        # Update the p array to zero out the region around the chosen index to prevent overlap
        p[start_idx:end_idx] = 0

        # normalize to create valid probability distribution
        p = p/p.sum()

        # Append the chosen index to the list of starting indices
        replay_start_idxs.append(start_idx)

    # save data about inserted events here and return if requested
    events = {'idx': [],
              'pos': [],
              'step': [],
              'class_idx': [],
              'span': [],
              'jitter': []}

    for idx,  start_idx in enumerate(replay_start_idxs):
        smp_jitter = 0  # starting with no jitter
        pos = start_idx  # pos indicates where in data we insert the next event

        # choose the starting class such that the n_steps can actually be taken
        # at that position to finish the sequence without looping to beginning
        seq_i = rng.choice(np.arange(len(sequence)-n_steps))
        for step in range(n_steps+1):
            # choose which item should be inserted based on sequence order
            class_idx = sequence[seq_i]
            # or take a single event (more noisy)
            data_class = insert_data[insert_labels==class_idx]
            idx_cls_i = rng.choice(np.arange(len(data_class)))
            insert_data_i = data_class[idx_cls_i]
            assert insert_data_i.ndim==2

            # time spans of the segments we want to insert
            t = insert_data_i.shape[-1]

            data_sim[pos-t//2:pos+1+t//2, :] += insert_data_i.T
            logging.debug(f'{start_idx=} {pos=} {class_idx=}')

            events['idx'] += [idx]
            events['pos'] += [pos]
            events['step'] += [step]
            events['class_idx'] += [class_idx]
            events['span'] += [insert_data_i.shape[-1]]
            events['jitter'] += [smp_jitter]

            # increment pos to select position of next reactivation event
            smp_jitter = rng.integers(-jitter, jitter+1) if jitter else 0
            pos += lag + smp_jitter  # add next sequence step
            seq_i += 1  # increment sequence id for next step

    if return_onsets:
        df_onsets = pd.DataFrame(events)
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


# if __name__=='__main__':
#     np.random.seed(0)
#     x1 = simulate_meeg(60,100)
#     np.random.seed(0)
#     x2 = simulate_meeg_opt(60,100)
