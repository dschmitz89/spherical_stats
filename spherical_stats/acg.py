import numpy as np
from numba import njit, guvectorize, float64, float32, void
from .utils import _invertmatrix
    
types = ["void(float64[:, :], float64[:, :], float64[:])", \
        "void(float32[:, :], float32[:, :], float32[:])"]
@guvectorize(types, "(n, m), (m, m) -> (n)", nopython=True, target="cpu", cache = True)
def _pdf(vectors, cov, pdfvals):

    inv_cov = _invertmatrix(cov).astype(vectors.dtype)
    constant = 1/(4* np.pi * np.sqrt(np.linalg.det(cov)))
    
    size = vectors.size
    n_samples = int(size/3)
        
    for _ in range(n_samples):

        x = vectors[_,:]

        pdfvals[_] = np.sum(x * (inv_cov@x))**(-1.5)#np.dot(inv_cov,x)

    pdfvals = constant * pdfvals
    
@njit(cache = True)
def fit(vectors, tol):
    '''
    Finds MLE of ACG distribution using the fix point algorithm
    Source: Tyler, Statistical  analysis  for  the  angular  central Gaussian  distribution  on  the  sphere
    Biometrika  (1987), 74,  3, pp.  579-89
    '''

    l = np.eye(3)

    n_samples = vectors.shape[0]

    iter = 0

    frob = 1e12

    while frob > tol:

        l_last = l

        l_inv = _invertmatrix(l).astype(vectors.dtype)

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

        frob = np.trace(diffmatrix*diffmatrix)

        if iter == 1000:
            print("Exceeded maximum number of iterations")
            break
        
    return l

class ACG(object):
    
    def __init__(self, cov_matrix = None):
        
        self.cov_matrix = cov_matrix
        
        #if covariance matrix provided, check positive definitenes
        
        if self.cov_matrix is not None:
            
            if not np.all(np.linalg.eigvals(self.cov_matrix) > 0):
                
                raise TypeError("Covariance matrix must be positive definite!")
            
    def fit(self, vectors, tol = 1e-4):
        
        self.cov_matrix = fit(vectors, tol)
        
    def rvs(self, size):
        
        zero = np.array([0,0,0])

        unnormalized_samples = np.random.multivariate_normal(zero, self.cov_matrix, size)

        norms = np.linalg.norm(unnormalized_samples, axis=1)[:,np.newaxis]

        samples = unnormalized_samples/norms

        return samples
    
    def pdf(self, x):
        
        if x.size == 3:
            x = x.reshape(1, -1)
        
        pdfvals = np.empty(x.shape[0], dtype=x.dtype)
        _pdf(x, self.cov_matrix, pdfvals)
        
        return pdfvals