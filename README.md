# spherical_stats
![alt text](https://github.com/dschmitz89/spherical_stats/blob/master/Logo_crop.png "")Spherical statistics in Python

## Scope
spherical_stats implements utilities for analyzing spherical data in Python. It is still under heavy development. For performance, the numba JIT compiler is used as backend.

## Features:


* Visualization helper functions to quickly generate data to be plotted with plotly/matplotlib/ipyvolume: 
    * Sphere creation and evaluation of a function over its surface
    * Spherical histogram
* Descriptive statisics: 
    * Spherical mean and spherical variance
    * Orientation tensor
* Parametric distributions:
    * Modeling axial data: Angular central gaussian distribution (ACG)
    * Modeling vector data: Elliptically symmetrical angular gausian distribution (ESAG)

Example usage of the distributions:

```python
from spherical_stats import ESAG
import numpy as np

esag_params = np.array([1,3,5,2,6])
esag_known = ESAG(esag_params)
samples = esag_known.rvs(500)
esag_unknown = ESAG()
esag_unknown.fit(samples, verbose = True)
```

## Coming up

Documentation

Mixture distributions of ESAG and ACG

## Installation
Clone the repository:
```bash
git clone https://github.com/dschmitz89/spherical_stats/
cd spherical_stats
pip install .
```