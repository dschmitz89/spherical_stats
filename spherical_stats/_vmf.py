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
    r"""
    Von Mises-Fisher distribution

    Args:
        mu (optional, ndarray (3, ) ): Mean orientation 
        kappa (optional, float): positive concentration parameter

    The VMF distribution is an isotropic distribution for 
    directional data. Its PDF is typically defined as 

    .. math::

        p_{vMF}(\mathbf{x}| \boldsymbol{\mu}, \kappa) = \frac{\kappa}{4\pi\cdot\text{sinh}(\kappa)}\exp(\kappa \boldsymbol{\mu}^T\mathbf{x})

    Here, the numerically stable variant from (Wenzel, 2012) is used:

    .. math::

        p_{vMF}(\mathbf{x}| \boldsymbol{\mu}, \kappa) = \frac{\kappa}{2\pi(1-\exp(-2\kappa))}\exp(\kappa( \boldsymbol{\mu}^T \mathbf{x}-1))
    
    References:

    Mardia, Jupp. Directional Statistics, 1999. 

    Wenzel. Numerically stable sampling of the von Mises Fisher distribution on  S2. 2012
    """
    def __init__(self, mu = None, kappa = None):
        
        self.mu = mu
        self.kappa = kappa

    def rvs(self, size = 1):
        '''
        Generate samples from the VMF distribution

        Arguments
        ----------
        size : int, optional, default 1
            Number of samples

        Returns
        ----------
        samples : ndarray (size, 3)
            samples as ndarray of shape (size, 3)
        ''' 
        if self.mu is not None and self.kappa is not None:
            
            z = np.array([0., 0., 1.])
            rot_matrix = rotation_matrix(z, self.mu)

            samples = _sample(self.kappa, rot_matrix, size)

            return samples

        else:

            raise ValueError("VMF distribution not parameterized. Fit it to data or set parameters manually.")

    def pdf(self, x):
        '''
        Calculate probability density function of a set of vectors ``x`` given a parameterized 
        VMF distribution

        Arguments
        ----------
        x : ndarray (size, 3)
            Vectors to evaluate the PDF at

        Returns
        ----------
        pdfvals : ndarray (size,)
            PDF values as ndarray of shape (size,)
        '''
        if self.mu is not None and self.kappa is not None:

            pdf = _pdf(self.mu, self.kappa, x)

            return pdf

        else:

            raise ValueError("VMF distribution not parameterized. Fit it to data or set parameters manually.")

    def fit(self, data):
        '''
        Fits the VMF distribution to data

        Arguments
        ----------
        data : ndarray (n, 3)
            Vector data the distribution is fitted to
        '''
        self.mu, self.kappa = _fit(data)

    def angle_within_probability_mass(self, alpha, deg = False):
        r"""
        Calculates the angle which contains probability mass alpha of the VMF density around the mean angle

        Reference: Fayat, 2021. Conversion of the von Mises-Fisher concentration parameter to an equivalent angle.

        https://github.com/rfayat/SphereProba/blob/main/ressources/vmf_integration.pdf

        Arguments
        ----------
        alpha : float
            Probability mass. Must be :math:`0<\alpha<1` 
        deg : optional, default False
            If True, converts the result into degrees

        Returns
        ----------
        angle : float
            Resulting angle
        
        """
        if self.kappa is not None:

            nominator = np.log(1-alpha + alpha * np.exp(-2 * self.kappa))  

            angle = np.arccos(1+nominator/self.kappa)

            if deg == True:
                angle=np.rad2deg(angle)

            return angle

        else:

            raise ValueError("Concentration parameter kappa unknown.")