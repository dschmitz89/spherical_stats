import numpy as np
from scipy.special import erfi
from ._utils import rotation_matrix
from ._descriptive_stats import orientation_matrix
from numba import njit
from scipy.optimize import brentq

class Watson:
    r"""
    Watson distribution

    .. note::

        The Watson distribution is only implemented for positive concentration parameter :math:`\kappa`.
    
    Args:
        mu (optional, ndarray (3, ) ): Mean axis 
        kappa (optional, float): positive concentration parameter

    The Watson distribution is an isotropic distribution for 
    axial data. Its PDF is defined as 

    .. math::

        p_{Watson}(\pm\mathbf{x}| \boldsymbol{\mu}, \kappa) & = M\left(\frac{1}{2},\frac{3}{2},\kappa\right)\exp(\kappa (\boldsymbol{\mu}^T\mathbf{x})^2) \\            
                                                            & = \frac{\sqrt{\pi}\mathrm{erfi}(\sqrt{\kappa})}{2\sqrt{\kappa}}\exp(\kappa (\boldsymbol{\mu}^T\mathbf{x})^2)

    where :math:`M` denotes `Kummer's confluent hypergeometric function <https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.hyp1f1.html#scipy.special.hyp1f1>`_ 
    and :math:`\mathrm{erfi}` the `imaginary error function <https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.erfi.html>`_ .

    References:

    Mardia, Jupp. Directional Statistics, 1999. 

    Chen. Generate Random Samples from von Mises-Fisher and Watson Distributions. 2012
    """
    def __init__(self, mu = None, kappa = None):
        
        self.mu = mu
        self.kappa = kappa

    def rvs(self, size = 1):
        '''
        Generate samples from the Watson distribution

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
            
            sqrt_kappa = np.sqrt(self.kappa)
            constant = np.sqrt(np.pi)*erfi(sqrt_kappa)/(2*sqrt_kappa)
            z = np.array([0., 0., 1.])
            rot_matrix = rotation_matrix(z, self.mu)

            samples = _sample(self.kappa, constant, rot_matrix, size)

            return samples

        else:

            raise ValueError("Watson distribution not parameterized. Fit it to data or set parameters manually.")

    
    def pdf(self, x):
        '''
        Calculate probability density function of a set of vectors ``x`` given a parameterized 
        Watson distribution

        Arguments
        ----------
        x : ndarray (size, 3)
            Vectors to evaluate the PDF at

        Returns
        ----------
        pdfvals : ndarray (size,)
            PDF values as ndarray of shape (size, )
        '''
        if self.mu is not None and self.kappa is not None:
            
            sqrt_kappa = np.sqrt(self.kappa)
            constant = np.sqrt(np.pi)*erfi(sqrt_kappa)/(2*sqrt_kappa)
            pdf = _pdf_wo_constant(self.mu, self.kappa, x)

            pdf = pdf * constant
            return pdf

        else:

            raise ValueError("Watson distribution not parameterized. Fit it to data or set parameters manually.")
    

    def fit(self, data):
        '''
        Fits the Watson distribution to data

        Arguments
        ----------
        data : ndarray (n, 3)
            Vector data the distribution is fitted to
        '''

        T = 1/data.shape[0] * orientation_matrix(data)
        evals, evectors = np.linalg.eigh(T)
        mu_fitted = evectors[:, 2]
        
        intermed_res = np.sum(mu_fitted * (T@mu_fitted))

        def obj(kappa):

            sqrt_kappa = np.sqrt(kappa)
            nominator = (2*np.exp(kappa)*sqrt_kappa - np.sqrt(np.pi) * erfi(sqrt_kappa))/(4*kappa**1.5)
            denominator = np.sqrt(np.pi)*erfi(sqrt_kappa)/(2*sqrt_kappa)

            f = nominator/denominator - intermed_res

            return f

        kappa_fit, root_res = brentq(obj, 1e-4, 500., full_output=True)
        
        if root_res.converged == True:
            self.mu = mu_fitted
            self.kappa = kappa_fit

        else:
            raise ValueError("Concentration parameter could not be estimated.")

@njit(cache = True)
def _pdf_wo_constant(mu, kappa, x):

    n_samples, _ = x.shape
    unnormalized_pdf = np.zeros((n_samples, ))

    for i in range(n_samples):

        unnormalized_pdf[i] = np.exp(kappa * ((x[i, :] * mu).sum())**2)

    return unnormalized_pdf

@njit(cache = True)
def rejection_sampling_numba(kappa, constant, size):

    res_array = np.zeros((size, ))

    #maximal density for given kappa
    maxy = constant * np.exp(kappa)

    number_samples = 0

    while number_samples < size:

        #draw uniform samples 
        x_rand = np.random.uniform(-1.0, 1.0)
        y_rand = np.random.uniform(0.0, maxy)

        #calculate density at position x
        f = constant * np.exp(kappa * x_rand * x_rand)

        #accept or reject
        if y_rand < f:

            res_array[number_samples] = x_rand
            number_samples +=1
    
    return res_array

@njit(cache = True)
def _sample(kappa, constant, rot_matrix, size):

    ones = np.ones((size, ))

    z = rejection_sampling_numba(kappa, constant, size)

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