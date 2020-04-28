# -*- coding: utf-8 -*-


import numpy as np
from numba import njit

@njit()
def frobeniusnorm(a):

    return np.trace(a*a)

def sample(cov, size):

    #test cov for positive definiteness

    if np.all(np.linalg.eigvals(cov) > 0):

        print("positive definite")
    
    zero = np.array([0,0,0])

    unnormalized_samples = np.random.multivariate_normal(zero, cov, size)

    norms = np.linalg.norm(unnormalized_samples, axis=1)[:,np.newaxis]

    samples = unnormalized_samples/norms

    return samples

@njit(cache = True)
def pdf(vectors, cov):

    inv_cov = np.linalg.pinv(cov)
    constant = 1/(4* np.pi * np.sqrt(np.linalg.det(cov)))
    n_samples = vectors.shape[0]

    pdf = np.empty((n_samples,))

    for _ in range(n_samples):

        x = vectors[_,:]

        pdf[_] = np.sum(x * np.dot(inv_cov,x))**(-1.5)

    pdf = constant * pdf

    return pdf

@njit(cache = True)
def fit(vectors):
    '''
    Finds MLE of ACG distribution using the fix point algorithm
    Source: Tyler, Statistical  analysis  for  the  angular  central Gaussian  distribution  on  the  sphere
    Biometrika  (1987), 74,  3, pp.  579-89
    '''

    l = np.eye(3)

    n_samples = vectors.shape[0]

    iter = 0

    frob = 1e12

    while frob > 1e-8:

        l_last = l

        l_inv = np.linalg.inv(l)

        num = np.zeros((3,3))
        denum = 0

        for _ in range(n_samples):
            
            vec = vectors[_, :]

            factor = np.sum(vec*np.dot(l_inv,vec))
            num += np.outer(vec, vec.T)/factor
            denum +=1/factor
            
        l = 3 * num/denum

        iter +=1

        diffmatrix = l - l_last

        frob = frobeniusnorm(diffmatrix)

        if iter == 1000:
            print("Exceeded maximum number of iterations")
            break
        
    return l

class ACG(object):
    
    def __init__(self, cov_matrix = None):
        
        self.cov_matrix = cov_matrix
        
        #if covariance matrix provided, check positive definitenes
        
        if self.cov_matrix is not None:
            
            if np.all(np.linalg.eigvals(self.cov_matrix) > 0):
                
                pass
            
            else:
            
                raise TypeError("Covariance matrix must be positive definite!")
            
    def fit(self, vectors):
        
        self.cov_matrix = fit(vectors)
        
    def rvs(self, size):
        
        return sample(self.cov_matrix, size)
    
    def pdf(self, x):
        
        return pdf(x, self.cov_matrix)