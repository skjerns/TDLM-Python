# TDLM-Python

[![Codespell](https://github.com/skjerns/TDLM-Python/actions/workflows/codespell.yml/badge.svg)](https://github.com/skjerns/TDLM-Python/actions/workflows/codespell.yml) 
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.12623445.svg)](https://doi.org/10.5281/zenodo.12623444)

This repository provides a Python implementation of [TDLM](https://elifesciences.org/articles/66917). Implementation matches the [MATLAB reference](https://github.com/YunzheLiu/TDLM) implementation up to [decimal precision](https://github.com/skjerns/TDLM-Python/blob/master/tdlm/tests/test_matlab_compatibility.py).

   \- Work-in-Progress -

Temporally Delayed Linear Modeling (TDLM) is a method used to quantify the "sequenceness" or sequential structure in time-series data. 

TDLM works by creating a linear model that incorporates time delays between different elements of the sequence. By introducing these delays, the model can capture the temporal dependencies within the sequence. For instance, in the context of analyzing human behavior, TDLM can help identify patterns such as the likelihood of certain reactivation events following others with a specific time lag.

## Installation

`pip install git+https://github.com/skjerns/TDLM-Python/`

## Usage

```python
import tdlm

# first you need a prediction matrix that contains decoded probabilities
# of individual events over time. The matrix must be shape
# (n_states, n_timesteps), e.g. (3, 1000), where each row contains the
# probabilities of that state's reactivation over time.

proba = ... # get your probability matrix somewhere

# next we need to define what transitions are expected.
# in our example we have a simple transition of states A->B->C->A
#    A  B  C
# A [0, 1, 0]
# B [0, 0, 1]
# C [1, 0, 0]
# you can also create it via `tdlm.utils.seq2tf('ABCA')`
tf = [[0, 1, 0], [ 0, 0, 1], [1, 0 , 0]]  # transition matrix

# next, input these two variables into the algorithm to calculate
# the sequenceness, i.e. if the probability time series
# show a dependence across time at a specific time lag
sequenceness_fwd, sequenceness_bkw, * = tdlm.compute_1step(proba, tf)

# results are a matrix of size (n_shuffles, max_lag)
# where entry [0, :] contains the baseline sequenceness
# and [1:, :] contains the shuffled versions, which you can use to 
# compute significance thresholds. 
# each column corresponds to the specific time lag, with column 0 being 
# NaN as it represents the 0-time-lag, which is not currently computed.

# plot results
tdlm.plotting.plot_sequenceness(sequenceness_fwd, sequenceness_bkw)
```

## Functionality

Currently, three different flavours are implemented: Cross-correlation, 1- and 2-step TDLM. More steps could be implemented easily, however, suffer from unclear statistical controls.

```python
# 'classical' cross correlation approach as in Kurth-Nelson 2016
tdlm.cross_correlation(preds, tf)

# 1-step TDLM, e.g. looking for transitions A->B, B->C, C->D independently, ...
tdlm.compute_1step(proba, tf)

# 2-step TDLM, e.g. looking for transitions (A->B)->C, (B->C)->D, ...
tdlm.compute_2step(proba, tf)
```

## Signflip permutation test

To get the classical permutation test as in Liu et al (2021), simply use the `compute_1step.(... n_shuf=X)`.
However, the more robust and less conservative signflip test is also available, use

```python
sf, sb = compute_1step(...)
p_fwd, t_fwd, t_perms = signflip_test(sf, n_perms=1000)
```

to plot your results, use this function

```
plot_tval_distribution(t_fwd, t_perms)
```

  

## ToDos / Contribute

Currently the repo is still quite bare, only providing basic functionality. I have created some [Issues](https://github.com/skjerns/TDLM-Python/issues) to get started. Designing a package is always challenging, as design choices will be difficult to revert, so feel free to contribute and discuss this.

**design decisions:**

* all time dimensions should be in sample steps. E.g. `max_lag=50` means 50 steps in sample space, disregarding the actual sample frequency. It is up to the user to calculate the fitting times.

* Transition matrices should be format of a binary matrix. Helper functions are provided to convert a sequence string in format `ABC_DEF` denoting transitions from A->B->C and D->E->F.

## Citation

If you use this package, please cite as

```
Simon Kern, Yunzhe Liu, & Lennart Wittkuhn. (2025). skjerns/TDLM-Python: v0.4 implement signflip permutation test (v0.4). Zenodo. https://doi.org/10.5281/zenodo.14501157
```
