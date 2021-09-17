# spherical_stats
Spherical statistics in Python

## Installation
```bash
pip install spherical_stats
```
## Documentation

Refer to the [online documentation](https://spherical-stats.readthedocs.io/en/latest/index.html) for examples and API reference.

## Features:

* Visualization helper functions to quickly generate data to be plotted with plotly/matplotlib/ipyvolume: 
    * Sphere creation and evaluation of a function over its surface
    * Spherical histogram
* Descriptive statistics: 
    * Spherical mean and spherical variance
    * Orientation tensor
* Parametric distributions with scipy.stats like API:
    * Modeling axial data: Angular central gaussian distribution (ACG)
    * Modeling vector data: Elliptically symmetrical angular gausian distribution (ESAG)

Example usage of the distributions:

```python
from spherical_stats import ESAG
import numpy as np

esag_params = np.array([1,3,5,2,6])

#Instantiate ESAG class with known parameters
esag_known = ESAG(esag_params)

#generate 500 ESAG samples and calculate their PDF vals
samples = esag_known.rvs(500)
pdf_vals = esag.pdf(samples)

#Instantiate ESAG class and fit distribution parameters given samples
esag_unknown = ESAG()
esag_unknown.fit(samples, verbose = True)
```