# TDLM-Python
(WIP) Python implementation of Temporally Delayed Linear Modelling

This repository provides a Python implementation of [TDLM](https://github.com/YunzheLiu/TDLM).

   \- Work-in-Progress -

## Installation

`pip install git+https://github.com/skjerns/TDLM-Python/`

## Usage

```
import tdlm

preds = ... # get your prediction matrix somehwere
tf = [[0, 1, 0], [ 0, 0, 1], [1, 0 , 0]]  # transition matrix
sequenceness_fwd, sequenceness_bkw, * = tdlm.compute_1step(preds, tf)

# plot results
tdlm.plotting.plot_sequenceness(sequenceness_fwd, sequenceness_bkw)

```

