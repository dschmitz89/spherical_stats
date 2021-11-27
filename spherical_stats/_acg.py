import numpy as np
from numba import njit, guvectorize, float64, float32, void
from ._utils import _invertmatrix
    
@njit(cache = True)
def _pdf(vectors, cov):

    inv_cov = _invertmatrix(cov).astype(vectors.dtype)
    constant = 1/(4* np.pi * np.sqrt(np.linalg.det(cov)))
    
    size = vectors.size
    n_samples = int(size/3)
    
    pdfvals = np.empty((n_samples, ), vectors.dtype)

    for _ in range(n_samples):

        x = vectors[_,:]

        pdfvals[_] = np.sum(x * np.dot(inv_cov,x))**(-1.5)

    pdfvals = constant * pdfvals

    return pdfvals
    
@njit(cache = True)
def fit(vectors, tol):

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
    r'''
    Angular Central Gaussian distribution

    Args:
        cov_matrix (optional, ndarray (3, 3) ): Covariance matrix of the ACG distribution

    The angular central gaussian distribution is an elliptically symmetrical distribution
    for axial data. Its PDF is defined as 

    .. math::

        p_{ACG}(\pm\mathbf{x}|\mathbf{\Lambda}) = \frac{1}{4\pi\sqrt{|\Lambda|}}(\mathbf{x}\Lambda^{-1}\mathbf{x})^{-\frac{3}{2}}

    with covariance matrix :math:`\Lambda` and its determinant :math:`|\Lambda|` . :math:`\Lambda` must be positive definite.

    Notes
    -------
    Reference: Tyler, Statistical  analysis  for  the  angular  central Gaussian  distribution  on  the  sphere
    Biometrika  (1987), 74,  3, pp.  579-89
    '''
    
    def __init__(self, cov_matrix = None):
        
        self.cov_matrix = cov_matrix
        
        #if covariance matrix provided, check positive definitenes
        
        if self.cov_matrix is not None:
            
            if not np.all(np.linalg.eigvals(self.cov_matrix) > 0):
                
                raise TypeError("Covariance matrix must be positive definite!")
            
    def fit(self, vectors, tol = 1e-4):
        '''
        Fits the Angular Central Gaussian Distribution to data

        Arguments
        ----------
        vectors : ndarray (n, 3)
            Vector data the distribution is fitted to
        tol : float, optional, default 1e-4
            Convergence tolerance of the fix point algorithm. Tolerance is measured as the difference between the Frobenius 
            norms of the estimated covariance matrix between two iterations.
        '''
        
        self.cov_matrix = fit(vectors, tol)
        
    def rvs(self, size = 1):
        '''
        Generate samples from the Angular central gaussian distribution

        Arguments
        ----------
        size : int, optional, default 1
            Number of samples

        Returns
        ----------
        samples : ndarray (size, 3)
            samples as ndarray of shape (size, 3)
        '''

        if self.cov_matrix != None:

            zero = np.array([0,0,0])

            unnormalized_samples = np.random.multivariate_normal(zero, self.cov_matrix, size)

            norms = np.linalg.norm(unnormalized_samples, axis=1)[:,np.newaxis]

            samples = unnormalized_samples/norms

            return samples

        else:

            raise ValueError("ACG distribution not parameterized. Fit it to data or set covariance matrix manually.")
    
    def pdf(self, x):
        '''
        Calculate probability density function of a set of vectors ``x`` given a parameterized 
        Angular Central Gaussian distribution

        Arguments
        ----------
        x : ndarray (size, 3)
            Vectors to evaluate the PDF at

        Returns
        ----------
        pdfvals : ndarray (size,)
            PDF values as ndarray of shape (size,)
        '''
        if self.cov_matrix is not None:  
            if x.size == 3:
                x = x.reshape(1, -1)
            
            pdfvals = _pdf(x, self.cov_matrix)
            
            return pdfvals
        else:
            raise ValueError("ACG distribution not parameterized. Fit it to data or set covariance matrix manually")