# -*- coding: utf-8 -*-

import numpy as np
from scipy.optimize import minimize
from numba import njit
from math import erf, sqrt, exp
from time import time
from ._utils import _invertmatrix

def _rvs(params, size):
    '''
    Calculates random samples from ESAG distribution
    
    Args:
        params (ndarray [5,]): ESAG parameters
        size (int): number of requested samples
        
    Returns:
        samples (ndarray [size,3]): ESAG samples
        
    '''

    mu = params[:3]
    gamma_1 = params[3]
    gamma_2 = params[-1]

    #first compute inverse covariance matrix

    inv_cov = _calc_inv_cov_matrix(mu, gamma_1, gamma_2)

    cov = _invertmatrix(inv_cov)

    unnormalized_samples = np.random.multivariate_normal(mu, cov, size)
    
    norms = np.linalg.norm(unnormalized_samples, axis=1)[:,np.newaxis]

    samples = unnormalized_samples/norms

    return samples

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

@njit(cache = True)
def _calc_inv_cov_matrix(mu, gamma_1, gamma_2):
    '''
    Calculates the inverse covariance matrix of ESAG
    
    Args:
        
        mu (ndarray [3,]): mu parameters of ESAG
        gamma_1 (float): gamma_1 parameter of ESAG
        gamma_2 (float): gamma_2 parameter of ESAG
        
    Returns:
        
        inv_cov (ndarray [3,3]): inverse covariance matrix 
    '''
    #xi1 and xi2 (eq. 14)

    mu_1 = mu[0]
    mu_2 = mu[1]
    mu_3 = mu[2]

    mu_0 = np.sqrt(mu_2 * mu_2 + mu_3 * mu_3)

    norm_mu = np.linalg.norm(mu)
    
    xi_1 = np.array([-mu_0 * mu_0, mu_1 * mu_2, mu_1*mu_3])/(mu_0 * norm_mu)

    xi_2 = np.array([0, -mu_3, mu_2])/mu_0

    first_bracket = np.outer(xi_1, xi_1.T) - np.outer(xi_2, xi_2.T)
    second_bracket = np.outer(xi_1, xi_2.T) + np.outer(xi_2, xi_1.T)
    factor = np.sqrt(gamma_1 * gamma_1 + gamma_2 * gamma_2 +1) -1
    third_bracket = np.outer(xi_1, xi_1.T) + np.outer(xi_2, xi_2.T)

    inv_cov = np.eye(3) + gamma_1 * first_bracket + gamma_2 * second_bracket + factor * third_bracket

    return inv_cov

@njit(cache = True)
def _likelihood(x, inv_cov, mu, mu_squared):
    '''
    Calculates the likelihood of vector x given the inverse covariance matrix, 
    mu and mu_squared
    
    Args:
        x (ndarray [3,]): direction vector whose likelihood is evaluated
        inv_cov (ndarray [3,3]): inverse covariance matrix
        mu (ndarray [3,]): mu vector of ESAG
        mu_squared (float): squared norm of mu
        
    Returns:
        
        l (float): likelihood of x given the parameters
    '''

    c_1 = np.sum(x*np.dot(inv_cov,x))
    c_2 = np.sum(x*mu)

    alpha = c_2/sqrt(c_1)

    cdf = _normal_cdf(alpha)
    pdf = _normal_pdf(alpha)
    m_2=(1+alpha * alpha) *cdf  + alpha * pdf

    l = 1/(2*np.pi)*c_1**(-3/2)*np.exp(0.5 * (alpha * alpha - mu_squared))*m_2

    return l

@njit(cache = True)
def _pdf(vectors, params):#, probabilities):
    '''
    Calculates the ESAG PDF of a set of vectors
           
    Args:
        vectors (ndarray [n,3]): set of direction vectors 
        params (ndarray [5,]): ESAG parameters 

    Returns:

        probabilities (ndarray [n,]): ESAG pdf values
    '''

    params = params.astype(vectors.dtype)#(np.float64)

    mu = params[:3]
    gamma_1 = params[3]
    gamma_2 = params[-1]

    inv_cov = _calc_inv_cov_matrix(mu, gamma_1, gamma_2)

    mu_squared = np.sum(np.square(mu))
    
    probabilities = np.empty(vectors.shape[0],vectors.dtype)
    for _ in range(vectors.shape[0]):

        probabilities[_] = _likelihood(vectors[_, :], inv_cov, mu, mu_squared)
        
    return probabilities

@njit(cache = True)
def _log_likelihood(params, samples):
    '''
    Computes log likelihood of params given the samples

    Args:

        params (ndarray [5,]): ESAG params
        samples (ndarray [n,3]): vectors 

    Returns:

        log-likelihood (float)
    '''

    probabilities = _pdf(samples, params)

    return - np.log(probabilities).sum()
    
def _fit(vectors, print_summary = False):
    '''
    Fits ESAG distribution to a sample of vectors

    Args:
        vectors (ndarray [n,3]): vectors to fit ESAG at
        print_summary (bool, Optional): print fit info

    Returns:
        optimized_params (ndarray [5]): Fitted ESAG params
    '''

    starting_guesses = (1,1,1,1e-5,1e-5)

    if print_summary:

        t0 = time()

    mle_result = minimize(_log_likelihood,starting_guesses, args=(vectors), \
                          method='L-BFGS-B')

    if print_summary:

        fit_time = time() - t0

    optimized_params = mle_result.x
    
    if print_summary:
        
        optimized_loglikelihood = mle_result.fun
        n_iterations = mle_result.nit
        mu_opt = optimized_params[:3]
        print("ESAG Fit Summary ")
        print("Maximum Likelihood parameters: ")
        print("mu={}, gammas={}".format(mu_opt,optimized_params[-2:]))
        print("Principal vector: {}".format(mu_opt/np.linalg.norm(mu_opt)))
        print("Minimized Log Likelihood: {}".format(optimized_loglikelihood))
        print("Optimization iterations: {}".format(n_iterations))
        print("Elapsed fitting time: {:10.3f}".format(fit_time))
        
    return optimized_params

class ESAG(object):
    r'''
    Elliptically symmetrical angular Central Gaussian distribution

    Args:
        params (optional, ndarray (5, ) ): Parameters of the distribution
        
    ``params`` are the following: :math:`(\mu_0, \mu_1, \mu_2, \gamma_1, \gamma_2)`.

    The principal orientation vectors is given by the normalized vector :math:`\boldsymbol{\mu}=[\mu_0, \mu_1, \mu_2]^T/||[\mu_0, \mu_1, \mu_2]^T||` 
    and the shape of the distribution is controlled by the parameters :math:`\gamma_1` and :math:`\gamma_2`.

    Notes
    -------
    The formula of the ESAG PDF is quite complicated, developers are referred to the reference below. 

    Note that unlike the original Matlab implementation the distribution is fitted using the L-BFGS-B algorithm
    based on a finite difference approximation of the gradient. So far, this has proven to work succesfully.

    Reference: Paine et al. An elliptically symmetric angular Gaussian distribution, 
    Statistics and Computing volume 28, 689–697 (2018)

    '''
    
    def __init__(self, params = None):
        
        self.params = params
        
        if self.params is not None:
            
            if not self.params.dtype == np.float64:

                self.params = self.params.astype(float)

    def fit(self, vectors, verbose=False):
        '''
        Fits the elliptically symmetrical angular Central Gaussian distribution to data

        Arguments
        ----------
        vectors : ndarray (n, 3)
            Vector data the distribution is fitted to
        verbose : bool, optional, default False
            Print additional information about the fit
        '''
                
        self.params = _fit(vectors, print_summary = verbose)
    
    def pdf(self, vectors):
        '''
        Calculate probability density function of a set of vectors ``x`` given a parameterized 
        elliptically symmetric angular Central Gaussian distribution

        Arguments
        ----------
        x : ndarray (size, 3)
            Vectors to evaluate the PDF at

        Returns
        ----------
        pdfvals : ndarray (size,)
            PDF values as ndarray of shape (size,)
        '''

        if self.params is not None:

            if vectors.size == 3:
                vectors = vectors.reshape(1, -1)

            return _pdf(vectors, self.params)

        else:
            raise ValueError("ESAG distibution not parameterized yet. Set parameters or fit ESAG to data.")
    
    def rvs(self, size = 1):
        '''
        Generate samples from the elliptically symmetric angular central gaussian distribution

        Arguments
        ----------
        size : int, optional, default 1
            Number of samples

        Returns
        ----------
        samples : ndarray (size, 3)
            samples as ndarray of shape (size, 3)
        '''   

        if self.params is not None: 
            return _rvs(self.params, size)

        else:
            raise ValueError("ESAG distibution not parameterized yet. Set parameters or fit ESAG to data.")
        