#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 22 21:35:04 2025

@author: simon
"""

import numpy as np
import pandas as pd
from scipy.signal import lfilter

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
    # If Cov is not strictly positive definite
    L = np.linalg.cholesky(cov)

    # 2. Generate White Noise (Standard Normal)
    Z = rng.standard_normal((n_samples, n_channels))

    # 3. Apply Temporal Filter to White Noise
    # Original logic: noise was scaled by autocorr before addition
    # Filter: y[n] = autocorr * y[n-1] + (autocorr * x[n])
    # To match original magnitude logic: Scale input noise by autocorr
    Z *= autocorr

    # Apply Filter along time axis (axis 0)
    # b=[1], a=[1, -autocorr]
    # We use zi to handle initial conditions smoothly if needed,
    # but strictly Z starts random, so standard filter is fine.
    Z = lfilter([1], [1, -autocorr], Z, axis=0)

    # 4. Apply Spatial Mixing (Matrix Multiplication)
    # X = Z_filtered @ L.T
    # This moves the heavy O(N*M^2) operation to a
    # single highly optimized BLAS call
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

    # common "ERP"-style pattern that is common to all patterns
    common_pattern = rng.normal(size=(1, n_channels))
    # then repeat pattern for each class and add some gaussian noise
    patterns = np.tile(common_pattern, (n_patterns, 1)) + \
               rng.standard_normal((n_patterns, n_channels))

    # base noise that is added to the trials
    base_noise = noise * rng.standard_normal((n_total, n_channels))

    # construct individual trials, later add noise
    stim_signal = np.tile(patterns, (n_train_per_stim, 1))

    # create matrix with n_null empty spaces and the individual trials
    signal_component = np.vstack([
        np.zeros((n_null, n_channels)),
        stim_signal
    ])

    # add noise to the individual trials
    training_data = base_noise + signal_component

    # Generate Labels, the zero class will act as negative samples later on
    stim_labels = np.tile(np.arange(1, n_patterns + 1), n_train_per_stim)
    training_labels = np.concatenate([
        np.zeros(n_null, dtype=int),
        stim_labels
    ])

    # Inject Extra Noise to half the patterns
    n_noise_groups = n_patterns // 2

    if n_noise_groups > 0:
        # choose which classes to make more noisy, but don't add noise
        # to the negative zero class
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

    # training data includes the zero null class, which is basically just noise
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
        Example: [5, 10] would block 5 steps before the an event start,
        and 10 steps after the last event point. If an event starts at sample
        100 with 2 steps and a lag of 8, the last event point would be at
        100+8+8, so the period of (100-5) to (100+8+8+10+1) would be blocked
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
            block_end   = start_idx + lag*n_steps + refractory[1] + 1

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
