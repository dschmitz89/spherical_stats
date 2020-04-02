from numba import njit
from math import erf, sqrt, exp
import numpy as np
import acg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

@njit(cache = True)
def _normal_cdf(v):
    '''Numba implementation of CDF of standard normal distribution
    
    Arg:
        v (float): point where to evaluate CDF

    Returns:
        CDF value (float)
    '''
    return 0.5 * (1+ erf(v/sqrt(2)))

@njit(cache = True)
def _normal_pdf(v):
    '''Numba implementation of PDF of standard normal distribution
    Args:
        v (float): point where to evalulate PDF
        
    Returns:
        PDF value (float)
    '''
    return 1/sqrt(2*np.pi) * exp(-0.5*v*v)

def _random_spd_matrix(normalize = True):

    A = np.random.rand(3,3)

    B = np.dot(A, A.T)
    trace = np.trace(B)

    if normalize:

        B = 3 * B/trace
    
    return B

def resultant_length(vectors):
    
    x_sum = vectors[:,0].sum()
    y_sum = vectors[:,1].sum()
    z_sum = vectors[:,2].sum()
    
    R = sqrt(x_sum * x_sum + y_sum * y_sum + z_sum * z_sum)
    
    return R

def _spherical_mean(vectors):
    
    x_sum = vectors[:,0].sum()
    y_sum = vectors[:,1].sum()
    z_sum = vectors[:,2].sum()
    
    R = sqrt(x_sum * x_sum + y_sum * y_sum + z_sum * z_sum)
        
    mean = 1/R*np.array([x_sum, y_sum, z_sum])
    
    return mean

def _spherical_variance(vectors):
    
    R = resultant_length(vectors)
    n_samples = vectors.shape[0]
    variance = 1 - R/n_samples
    
    return variance

def plot(n_grid, samples):
    
    u = np.linspace(0, np.pi, n_grid)
    
    v = np.linspace(0, 2 * np.pi, n_grid)

    x = np.outer(np.sin(u), np.sin(v))
    y = np.outer(np.sin(u), np.cos(v))
    z = np.outer(np.cos(u), np.ones_like(v))

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    
    ax.plot_surface(x, y, z, zorder=0, rstride=1, cstride=1, \
                        antialiased=False, linewidth=0)
    
    ax.scatter(samples[:,0],samples[:,1],samples[:,2], c='r', s = 8, \
        zorder=1)
    
    plt.show()
    
cov = _random_spd_matrix()
samples = acg.sample(cov, 500)
m = _spherical_mean(samples)
print(_spherical_variance(samples))
print(np.linalg.norm(m))
plot(50, samples)