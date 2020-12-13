#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 15:22:05 2020

@author: tyrion
"""
import numpy as np
from numba import njit

@njit(cache=True)
def _invertmatrix(m):
    '''Calculates the inverse of a 3x3 matrix analytically'''

    #calculate determinant
    det = m[0,0] * m[1,1] * m[2,2] + m[1,0] * m[2,1] * m[0,2] + m[2,0] * m[1,2] - m[0,0] * m[2,1] * m[1,2] - \
        m[2,0] * m[1,1] * m[0,2] - m[1,0] * m[0,1] * m[2,2]

    ##initialize inverted matrix
    m_inv = np.empty((3, 3))

    m_inv[0, 0] = m[1, 1] * m[2, 2] - m[1, 2] * m[2, 1]
    m_inv[0, 1] = m[0, 2] * m[2, 1] - m[0, 1] * m[2, 2]
    m_inv[0, 2] = m[0, 1] * m[1, 2] - m[0, 2] * m[1, 1]
    m_inv[1, 0] = m[1, 2] * m[2, 0] - m[1, 0] * m[2, 2]
    m_inv[1, 1] = m[0, 0] * m[2, 2] - m[0, 2] * m[2, 0]
    m_inv[1, 2] = m[0, 2] * m[1, 0] - m[0, 0] * m[1, 2]
    m_inv[2, 0] = m[1, 0] * m[2, 1] - m[1, 1] * m[2, 0]
    m_inv[2, 1] = m[0, 1] * m[2, 0] - m[0, 0] * m[2, 1]
    m_inv[2, 2] = m[0, 0] * m[1, 1] - m[0, 1] * m[1, 0]

    m_inv /= det

    return m_inv

def random_spd_matrix(normalize = True):

    A = np.random.rand(3,3)

    B = np.dot(A, A.T)
    trace = np.trace(B)

    if normalize:

        B = 3 * B/trace
    
    return B

def orientation_matrix(vectors, eigen = True):
    '''
    Calculates the orientation matrix and its Eigen decomposition
    of a set of vectors

    Parameters
    ----------
    vectors : ndarray [n,3]
        sample of n vectors
    eigen : bool, optional
        indicate that Eigen decomposition shall be performed. 
        The default is True.

    Returns
    -------
    orientation_matrix
    ndarray [3,3,]
    
    or
    
    eigenvals: ndarray[3]
    eigenvecs: ndarray[3,3]

    '''
    x_vals = vectors[:, 0]
    y_vals = vectors[:, 1]
    z_vals = vectors[:, 2]
    
    orientation_matrix = np.zeros((3,3))
    
    orientation_matrix[0, 0] = np.sum(x_vals * x_vals)
    xysum = np.sum(x_vals * y_vals)
    orientation_matrix[0, 1] = xysum
    orientation_matrix[1, 0] = xysum
    xzsum = np.sum(x_vals * z_vals)
    orientation_matrix[0, 2] = xzsum
    orientation_matrix[2, 0] = xzsum
    orientation_matrix[1, 1] = np.sum(y_vals * y_vals)
    yzsum = np.sum(y_vals * z_vals)
    orientation_matrix[1, 2] = yzsum
    orientation_matrix[2, 1] = yzsum
    orientation_matrix[2, 2] = np.sum(z_vals * z_vals)
    
    if eigen == True:
        
        eigenvals, eigenvecs = np.linalg.eigh(orientation_matrix)
        
        n_samples = vectors.shape[0]
        
        eigenvals = eigenvals/n_samples
        
        return eigenvals, eigenvecs
    
    else:
        
        return orientation_matrix    