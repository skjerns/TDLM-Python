# TDLM-Python

[![Codespell](https://github.com/skjerns/TDLM-Python/actions/workflows/codespell.yml/badge.svg)](https://github.com/skjerns/TDLM-Python/actions/workflows/codespell.yml)

This repository provides a Python implementation of [TDLM](https://github.com/YunzheLiu/TDLM).

   \- Work-in-Progress -

Temporally Delayed Linear Modeling (TDLM) is a method used to quantify the "sequenceness" or sequential structure in time-series data. It's particularly useful for analyzing sequences of events or actions where the order of occurrence is significant. TDLM works by creating a linear model that incorporates time delays between different elements of the sequence. By introducing these delays, the model can capture the temporal dependencies within the sequence. For instance, in the context of analyzing human behavior, TDLM can help identify patterns such as the likelihood of certain reactivation events following others with a specific time lag.

## Installation

`pip install git+https://github.com/skjerns/TDLM-Python/`

## Usage

```
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
tf = [[0, 1, 0], [ 0, 0, 1], [1, 0 , 0]]  # transition matrix

# next, input these two variables into the algorithm to 
sequenceness_fwd, sequenceness_bkw, * = tdlm.compute_1step(proba, tf)

# results are a matrix of size (n_shuffles, max_lag)
# where entry [0, :] contains the 

# plot results
tdlm.plotting.plot_sequenceness(sequenceness_fwd, sequenceness_bkw)
```

## ToDos / Contribute

Currently the repo is still quite bare, only providing basic functionality. I have created some [Issues](https://github.com/skjerns/TDLM-Python/issues) to get started. Designing a package is always challenging, as design choices will be difficult to revert, so feel free to contribute and discuss this.

**design decisions:**

* all time dimensions should be in sample steps. E.g. `max_lag=50` means 50 steps in sample space, disregarding the actual sample frequency. It is up to the user to calculate the fitting times.

* The base function for all computations should be `compute_glm` or `compute_crosscorr`, all other functions should use this function modularly to their needs

* Cross correlation and GLM are two different base function that can be used. Other functions take a parameter which is either `glm` or `crosscorr` to denote which function is used. This way it is theoretically possible to extend the repository using other functions as well (e.g. granger causality, which is basically a flavor of crosscorr I guess)

* Transition matrices should be format of a binary matrix. Helper functions are provided to convert a sequence string in format `ABC_DEF` denoting transitions from A->B->C and D->E->F.
