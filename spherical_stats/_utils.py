#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 15:22:05 2020

@author: tyrion
"""
import numpy as np
from numba import njit
from os.path import dirname

PATH = dirname(__file__)

def load_northpole():
    '''
    Load example data for finding the historical position of the north pole

    Reference: Paine et al. An elliptically symmetric angular Gaussian distribution, 
    Statistics and Computing volume 28, 689–697 (2018)

    Returns
    ----------
    data : ndarray (33, 3)
    '''

    data = np.loadtxt(open(PATH + '/tasmanianData.csv', 'rb'), delimiter=",")

    return data.T

def sphericalrand(size=1):
    '''
    Generate uniform random samples on a sphere

    Arguments
    ----------
    size : int, optional, default 1
        Number of samples

    Returns
    ----------
    samples : ndarray (size, 3)
    '''
    ones = np.ones((size, ))
    u = 2 * np.random.rand(size) - ones
    phi = 2 * np.pi * np.random.rand(size)

    theta = np.sqrt(ones - np.square(u))
    samples = np.zeros((size, 3))
    samples[:, 0] = theta * np.cos(phi)
    samples[:, 1] = theta * np.sin(phi)
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

#@njit(cache = True)
def rotation_matrix(a, b):
    """ The rotation matrix that takes a onto b.

    Parameters
    ----------
    a, b : numpy.array
        Three dimensional vectors defining the rotation matrix

    Returns
    -------
    M : numpy.array
        Three by three rotation matrix

    Notes
    -----
    Source:
    https://spherical-kde.readthedocs.io/en/latest/_modules/spherical_kde/utils.html

    StackExchange post:
        https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
    """
    v = np.cross(a, b)
    s = v.dot(v)**0.5
    if s == 0:
        return np.eye(3)
    c = np.dot(a, b)
    Id = np.eye(3)
    #v1 = v[0]
    #v2 = v[1]
    #v3 = v[2]
    v1, v2, v3 = v
    vx = np.array([[0, -v3, v2],
                      [v3, 0, -v1],
                      [-v2, v1, 0]])
    vx2 = np.matmul(vx, vx)
    R = Id + vx + vx2 * (1-c)/s**2
    return R