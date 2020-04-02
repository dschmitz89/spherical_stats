# spherical_stats
Spherical statistics in Python

## Scope
spherical_stats seeks to implement utilities for analyzing spherical data in Python. It is still under heavy development. For performance, the numba JIT compiler is used as backend.

### So far Implemented are two probability distributions:
Angular central gaussian distribution (ACG)

Elliptically symmetrical gausian distribution (ESAG)

Both can be accessed via a scipy.stats like API

```python
from spherical_stats import ESAG
import numpy as np

esag_params = np.array([1,3,5,2,6])
esag_known = ESAG(esag_params)
samples = esag_known.rvs(500)
esag_unknown = ESAG()
esag_unknown.fit(samples, verbose = True)
```

### Coming up

Documentation

Plotting functionality based on matplotlib

## Installation
Clone the repository

In cloned repository: 
```python
pip install .
```
