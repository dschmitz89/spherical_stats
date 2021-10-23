import numpy as np
from numba import njit, jit
#from _utils import sphericalrand
from scipy import optimize 
from ._descriptive_stats import spherical_mean
from ._utils import rotation_matrix
#from ._coordinate_systems import polar_to_vectors

@njit(cache = True)
def _pdf(mu, kappa, x):

    n_samples, _ = x.shape
    pdf = np.zeros((n_samples, ))
    #print(pdf.shape)
    if kappa == 0:

        return pdf.fill(0.25/np.pi)

    else:

        constant = kappa/(2*np.pi*(1-np.exp(-2*kappa)))
        for i in range(n_samples):

            pdf[i] = constant*np.exp(kappa * ((x[i, :] * mu).sum() -1))

        return pdf

@njit(cache = True)
def _sample(kappa, rot_matrix, size):

    ones = np.ones((size, ))
    zinit = np.random.random(size)

    z = ones + np.log(zinit + (ones - zinit) * np.exp(-2 * kappa))/kappa

    temp = np.sqrt(ones - np.square(z))
    uniformcirle = 2 * np.pi * np.random.random(size)
    x = np.cos(uniformcirle)
    y = np.sin(uniformcirle)

    samples = np.empty((size, 3))

    samples[:, 0] = temp * x
    samples[:, 1] = temp * y
    samples[:, 2] = z

    for i in range(size):

        vec=samples[i, :]
        
        samples[i, :] = rot_matrix.dot(vec)

    return samples

def _fit(data):

    mu = spherical_mean(data)
    S = np.sum(data, axis=0)
    R = S.dot(S)**0.5/data.shape[0]

    @njit(cache = True)
    def obj(s):
        return 1/np.tanh(s)-1./s-R

    kappa = optimize.brentq(obj, 1e-8, 1e8)

    return mu, kappa

class VMF:

    def __init__(self, mu = None, kappa = None):
        
        self.mu = mu
        self.kappa = kappa

    def rvs(self, size = 1):

        if self.mu is not None and self.kappa is not None:
            
            z = np.array([0., 0., 1.])
            rot_matrix = rotation_matrix(z, self.mu)

            samples = _sample(self.kappa, rot_matrix, size)

            return samples

        else:

            raise ValueError("VMF distribution not parameterized. Fit it to data or set parameters manually.")

    def pdf(self, x):

        if self.mu is not None and self.kappa is not None:

            pdf = _pdf(self.mu, self.kappa, x)

            return pdf

        else:

            raise ValueError("VMF distribution not parameterized. Fit it to data or set parameters manually.")

    def fit(self, data):

        self.mu, self.kappa = _fit(data)    

'''
theta = np.deg2rad(30)
phi = 0

MU = polar_to_vectors(theta, phi)
KAPPA = 15

vmf_1 = VMF(MU, 15)

vmfsamples = vmf_1.rvs(500)

vmf_2 = VMF()
vmf_2.fit(vmfsamples)

print(mf_2.mu)


fit_mu, fit_kappa = _fit(vmfsamples)
print(fit_mu)
print(fit_kappa)

import spherical_stats
import matplotlib.pyplot as plt

x, y, z = spherical_stats.sphere(n_grid = 30)

plt.rcParams['figure.figsize'] = [8, 8]
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(
    x, y, z,  rstride=1, cstride=1, color='c', alpha=0.9, linewidth=1)

ax.scatter(vmfsamples[:, 0], vmfsamples[:, 1], vmfsamples[:, 2], color="r",s=20)
#ax.view_init(57, 145)
plt.show()

data=sphericalrand(500)

MU = np.array([1., 0., 0.])
KAPPA = 100

pdfvals = _pdf(MU, KAPPA, data)

print(pdfvals)
#pdfvals = _pdf(MU, KAPPA, data)
'''