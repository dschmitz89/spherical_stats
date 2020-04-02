# -*- coding: utf-8 -*-

import numpy as np
from scipy.optimize import minimize
from numba import njit
from math import erf, sqrt, exp
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from time import time
#import utils

def rvs(params, size):
    '''
    Calculates random samples from ESAG distribution
    
    Args:
        params (ndarray [5,]): ESAG parameters
        size (int): number of requested samples
        
    Returns:
        samples (ndarray [size,3]): ESAG samples
        
    '''

    params = params.astype(float)
    mu = params[:3]
    gamma_1 = params[3]
    gamma_2 = params[-1]

    #first compute inverse covariance matrix

    inv_cov = _calc_inv_cov_matrix(mu, gamma_1, gamma_2)

    cov = np.linalg.pinv(inv_cov)

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
def pdf(vectors, params):#, probabilities):
    '''
    Calculates the ESAG PDF of a set of vectors
           
    Args:
        vectors (ndarray [n,3]): set of direction vectors 
        params (ndarray [5,]): ESAG parameters 

    Returns:

        probabilities (ndarray [n,]): ESAG pdf values
    '''

    params = params.astype(np.float64)

    mu = params[:3]
    gamma_1 = params[3]
    gamma_2 = params[-1]

    inv_cov = _calc_inv_cov_matrix(mu, gamma_1, gamma_2)

    mu_squared = np.sum(np.square(mu))
    
    probabilities = np.empty(vectors.shape[0],)
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

    probabilities = pdf(samples, params)

    return - np.log(probabilities).sum()
    
def fit(vectors, optimizer='L-BFGS-B', print_summary = False):
    '''
    Fits ESAG distribution to a sample of vectors

    Args:
        vectors (ndarray [n,3]): vectors to fit ESAG at
        optimizer (str, Optiona, default='L-BFGS-B'): 
            which one of scipy's optimizers to use
        return_logp (bool, Optional): also return logp
        print_summary (bool, Optional): print fit info

    Returns:
        optimized_params: Fitted ESAG params as numpy array (5,)
        optimized_likelihood: logp value at found minimum
    '''

    starting_guesses = (1,1,1,1e-5,1e-5)

    if print_summary:

        t0 = time()

    mle_result = minimize(_log_likelihood,starting_guesses, args=(vectors), method=optimizer)

    if print_summary:

        fit_time = time() - t0

    optimized_params = mle_result.x
    
    if print_summary:
        
        optimized_loglikelihood = mle_result.fun

    if print_summary:

        n_iterations = mle_result.nit
        mu_opt = optimized_params[:3]
        print("ESAG Fit Summary ")
        print("Maximum Likelihood parameters: ")
        print("mu={}, gammas={}".format(mu_opt,optimized_params[-2:]))
        print("Principal vector: {}".format(mu_opt/np.linalg.norm(mu_opt)))
        print("Minimized Log Likelihood: {}".format(optimized_loglikelihood))
        print("Optimization iterations: {}".format(n_iterations))
        print("Elapsed fitting time: {:10.3f}".format(fit_time))

    #if return_logp:
            
    #    return optimized_params, optimized_loglikelihood
    
    else:
        
        return optimized_params

def plot(esag_params, n_grid = 50, plot_esag_density = True, 
        const_face_area = True, samples = None, 
        color_samples = False, show_axis = True):
    '''
    Function to plot ESAG on the sphere using matplotlib

    Params:
        esag_params (ndarray[5,]): ESAG params 
        n_grid (int, optional, default=50): grid lines of the plotted sphere
        plot_esag_density (bool, default = True): colorcode sphere with ESAG density
        const_face_area (bool, Optional, default = True): sample surface so that
        every face has the same area
        samples (ndarray[n,3], Optional): samples to plot on sphere
        color_samples (bool, Optional, default = False): color samples acc. to their ESAG density
        show_axis (bool, optional, default = True): show axes
    '''
    #create sphere
    
    if const_face_area:

        u = np.flip(np.arccos(np.linspace(-1,1,n_grid)))
    
    else:

        u = np.linspace(0, np.pi, n_grid)
    
    v = np.linspace(0, 2 * np.pi, n_grid)

    x = np.outer(np.sin(u), np.sin(v))
    y = np.outer(np.sin(u), np.cos(v))
    z = np.outer(np.cos(u), np.ones_like(v))

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    if show_axis is not True:

        ax.axis('off')

    else:

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
              
    if plot_esag_density:
        
        #calculate esag density for faces of sphere surface
        #create vector with faces coordinates
        
        u_grid, v_grid = np.meshgrid(u, v)
    
        vertex_vecs = np.array([np.sin(v_grid)*np.sin(u_grid), np.cos(v_grid)*np.sin(u_grid),np.cos(u_grid)])

        vertex_vecs = vertex_vecs.reshape(3,n_grid*n_grid)
        
        #calculate esag density for the faces
        
        pdfs_faces = pdf(vertex_vecs.T,esag_params).reshape(n_grid,n_grid)
        
        #create colormap for the faces
        
        normed_facecolors = plt.Normalize(vmin=pdfs_faces.min(), vmax=pdfs_faces.max())
        
        facecolors = cm.viridis(normed_facecolors(pdfs_faces.T))
        
        ax.plot_surface(x, y, z, zorder=0, facecolors=facecolors, \
                    rstride=1, cstride=1, antialiased=False, linewidth=0)
        
    else:
        
        ax.plot_surface(x, y, z, zorder=0, rstride=1, cstride=1, \
                        antialiased=False, linewidth=0)
    
    #if additionally vector samples are provided, plot them

    if samples is not None:
        
        #if required colorcode directions by their ESAG PDF

        if color_samples:       
    
            samples_pdf = pdf(samples, esag_params)
            
            sample_colors_cb = plt.Normalize(vmin=samples_pdf.min(), vmax=samples_pdf.max())
            
            sample_colors = cm.viridis(sample_colors_cb(samples_pdf))
        
            ax.scatter(samples[:,0],samples[:,1],samples[:,2], c=sample_colors, s = 8, \
                       zorder=1)
        else:
            
            ax.scatter(samples[:,0],samples[:,1],samples[:,2], c='r', s = 8, \
                       zorder=1)
    
    #set initial view to center of the distribution at principal axis
    
    mu = esag_params[:3]
    mu = mu/np.linalg.norm(mu)
    azim = np.rad2deg(np.arctan2(mu[1],mu[0]))
    elev = np.rad2deg(np.arcsin(mu[2]))
    ax.view_init(azim=azim, elev=elev)
    
    plt.show()
    
class ESAG(object):
    
    def __init__(self, params = None):
        
        self.params = params
    
    def fit(self, vectors, optimizer='L-BFGS-B', \
		verbose=False):
        
        self.params = fit(vectors, optimizer=optimizer, \
                          print_summary = verbose)
    
    def pdf(self, vectors):
        
        return pdf(vectors, self.params)
    
    def rvs(self, size):
        
        return rvs(self.params, size)
        
