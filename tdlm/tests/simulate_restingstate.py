# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 15:50:44 2024

@author: Simon.Kern
"""
import numpy as np
import mne
from mne import create_info
from mne.io import RawArray


# Load your existing resting state MEG recording
raw = mne.io.read_raw_fif('/zi/flstorage/group_klips/data/data/Simon/DeSMRRest/seq12/DSMR117/MEG/RS1_trans[localizer1]_tsss_mc.fif', preload=True)

# Inspect the data
raw.plot()

events = mne.find_events(raw, min_duration=3/1000)
t_start = events[np.where(events[:, 2]==10)][0][0] / raw.info['sfreq']
t_end = events[np.where(events[:, 2]==11)][0][0] / raw.info['sfreq']

assert abs((t_end-t_start)-480)<1  # 1 second tolerance

# Compute the noise covariance from the resting state data
noise_cov = mne.compute_raw_covariance(raw, tmin=t_start, tmax=t_end)

# Plot the noise covariance
mne.viz.plot_cov(noise_cov, raw.info)



# Define the forward model (you might need to adapt this to your specific setup)
fwd = mne.read_forward_solution('/zi/flstorage/group_klips/data/data/Simon/DeSMRRest/seq12/DSMR117/DSMR117-fwd.fif')

# Create a source space simulation
# Number of sources, duration, and sampling frequency
n_sources = len(fwd['src'][0]['vertno']) + len(fwd['src'][1]['vertno'])
duration = raw.times[-1]
sfreq = raw.info['sfreq']
n_times = int(duration * sfreq)

# Simulate random source activity
random_state = np.random.RandomState(42)
source_activity = random_state.normal(size=(n_sources, n_times))

# Create a SourceEstimate object
stc = mne.SourceEstimate(source_activity, vertices=[fwd['src'][0]['vertno'], fwd['src'][1]['vertno']],
                         tmin=0, tstep=1/sfreq, subject='sample')
stc.crop(60, 120)
raw.resample(100, n_jobs=-1)
stc.resample(100, n_jobs=-1)
simulated_raw = mne.simulation.simulate_raw(raw.info, [stc], forward=fwd)


# Create random noise based on the noise covariance
noise = mne.simulation.add_noise(simulated_raw.copy(), cov=noise_cov, iir_filter=[0.2, -0.2, 0.02], random_state=42)

# Add the noise to the simulated data
simulated_raw._data += noise._data

# Plot the simulated data
simulated_raw.plot()
