#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 15:22:05 2020

@author: tyrion
"""
import numpy as np
from numba import njit

def sphericalrand(size):
    
    ones = np.ones((size, ))
    u = 2 * np.random.rand(size) - ones
    phi = 2 * np.pi * np.random.rand(size)

    theta = np.sqrt(ones - np.square(u))
    samples = np.zeros((size, 3))
    samples[:, 0] = theta * np.cos(phi)
    samples[:, 1] = theta = np.sin(phi)
    samples[:, 2] = u

    return samples

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