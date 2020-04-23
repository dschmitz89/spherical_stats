# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import random
from numba import njit
from time import time

@njit()
def frobeniusnorm(a):

    return np.trace(a*a)

def _random_spd_matrix(normalize = True):

    A = random.rand(3,3)

    B = np.dot(A, A.T)
    trace = np.trace(B)

    if normalize:

        B = 3 * B/trace
    
    return B

def sample(cov, size):

    #test cov for positive definiteness

    if np.all(np.linalg.eigvals(cov) > 0):

        print("positive definite")
    
    zero = np.array([0,0,0])

    unnormalized_samples = np.random.multivariate_normal(zero, cov, size)

    norms = np.linalg.norm(unnormalized_samples, axis=1)[:,np.newaxis]

    samples = unnormalized_samples/norms

    return samples

def plot(samples_1, samples_2, n_grid):

    u = np.linspace(0, np.pi, n_grid)
    
    v = np.linspace(0, 2 * np.pi, n_grid)

    x = np.outer(np.sin(u), np.sin(v))
    y = np.outer(np.sin(u), np.cos(v))
    z = np.outer(np.cos(u), np.ones_like(v))

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    ax.plot_surface(x, y, z, zorder=0, rstride=1, cstride=1, \
        antialiased=False, linewidth=0, alpha=0.5)

    ax.scatter(samples_1[:,0], samples_1[:,1], samples_1[:,2], c='r',zorder = 1)
    ax.scatter(samples_2[:,0], samples_2[:,1], samples_2[:,2], c='k',zorder = 1)

    plt.show()

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
    
'''   
cov_matrix = _random_spd_matrix()

vecs = sample(cov_matrix, 500)

#first_acg = ACG(cov_matrix)
#print(first_acg.cov_matrix)
empty_acg = ACG()
empty_acg.fit(vecs)
print(cov_matrix)
print(empty_acg.cov_matrix)
pdf_vals = empty_acg.pdf(vecs)
print(pdf_vals)


cc = CC('ACG_PDF') 
cc.verbose = True
cc.output_dir='/data/PLI-Group/Daniel/Development/Directional_Distributions/pyesag/'

@cc.export('pdf','f8[:](f8[:,:],f8[:,:])')
def aot_pdf(vectors, cov):

    inv_cov = np.linalg.pinv(cov)
    constant = 1/np.sqrt(np.linalg.det(cov))
    n_samples = vectors.shape[0]

    pdf = np.empty((n_samples,))

    for _ in range(n_samples):

        x = vectors[_,:]

        pdf[_] = constant * np.sum(x * np.dot(inv_cov,x))**(-1.5)

    return pdf

#cc.compile()


cov_matrix = _random_spd_matrix()

#tr_cov = np.trace(cov_matrix)
print(cov_matrix)
#print("Original inverse cov matrix: {}".format(np.linalg.pinv(cov_matrix)))

acg_samples = sample(cov_matrix, 5000)

mle_sol = fit(acg_samples)
print(mle_sol)
t0 = time()
mle_sol = fit(acg_samples)
print(time() - t0)
#mle_samples, _ = _sample_avg(mle_sol, 500)
#print(mle_samples.shape)
#plot(acg_samples, mle_samples, 50)
#print(np.linalg.pinv(mle_sol))

t0 = time()
pdf_vals_2 = pdf(acg_samples, cov_matrix)
print(time() - t0)
t0 = time()
pdf_vals_2 = pdf(acg_samples, cov_matrix)
print(time() - t0)
'''




#plot(acg_samples, samples, 50)
