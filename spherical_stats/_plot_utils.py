#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 20:58:05 2020

@author: tyrion
"""
import numpy as np

def sphere(n_grid = 30, equalize_areas = True):
    '''
    Create vectors to conveniently plot a sphere

    Arguments
    ----------
    n_grid  : int, optional, default 30
        Number of grid points for the sphere for both
        longitude and lattitude
    equalize_areas : bool, optional, default True
        If True, enforces that surface patches are of same area

    Returns
    ----------
    x : ndarray 
    y : ndarray 
    z : ndarray 
    '''

    if equalize_areas == True:
        u = np.arccos(np.linspace(-1, 1, n_grid))

    else:    
        u = np.linspace(0, np.pi, n_grid)
    

    v = np.linspace(0, 2 * np.pi, n_grid)

    x = np.outer(np.sin(u), np.sin(v))
    y = np.outer(np.sin(u), np.cos(v))
    z = np.outer(np.cos(u), np.ones_like(v))
   
    return x, y, z

def evaluate_on_sphere(func, n_grid = 30, equalize_areas = True):
    '''
    Evaluate a function over a sphere

    Arguments
    ----------
    func : callable
        Must be of the form ndarray (n, 3)-> (n, )
    n_grid  : int, optional, default 30
        Number of grid points for the sphere for both
        longitude and lattitude
    equalize_areas : bool, optional, default True
        If True, enforces that surface patches are of same area

    Returns
    ----------
    f : ndarray (n, )
    '''    
    if equalize_areas == True:
        u = np.arccos(np.linspace(-1, 1, n_grid))

    else:    
        u = np.linspace(0, np.pi, n_grid)
    v = np.linspace(0, 2 * np.pi, n_grid)
    
    u_grid, v_grid = np.meshgrid(u, v)

    vertex_vecs = np.array([np.sin(v_grid)*np.sin(u_grid), \
                        np.cos(v_grid)*np.sin(u_grid),np.cos(u_grid)])

    vertex_vecs = vertex_vecs.reshape(3,n_grid*n_grid).T

    #calculate function values for the sphere faces

    f_faces = func(vertex_vecs).reshape(n_grid,n_grid).T
    
    return f_faces

def spherical_hist(vectors, n_grid = 100):
    '''
    Basic spherical histogram

    Arguments
    ----------
    vectors : ndarray (n, 3)
        Vectors to calculate histogram of
    n_grid  : int, optional, default 100
        number of grid points for the sphere for both
        longitude and lattitude

    Returns
    ----------
    hist : ndarray
    x : ndarray 
    y : ndarray 
    z : ndarray 
    '''
        
    #u = np.linspace(0, np.pi , n_grid, endpoint = True)
    u = np.flip(np.arccos(np.linspace(-1,1,n_grid, endpoint = True)))
    v = np.linspace(-np.pi, np.pi, n_grid, endpoint = True)
    
    phis = np.arctan2(vectors[:,1], vectors[:,0])
    #print(np.rad2deg(phis))
    thetas = np.arccos(vectors[:,2])
    
    hist,thetabins, phibins = np.histogram2d(thetas, phis, bins = [u,v])
    
    #u_surface = np.linspace(0, np.pi , n_grid -1, endpoint = True)
    u_surface = np.flip(np.arccos(np.linspace(-1,1,n_grid -1, endpoint = True)))
    v_surface = np.linspace(-np.pi, np.pi, n_grid -1, endpoint = True)
    #print(u_centers)
    x = np.outer(np.sin(u_surface), np.cos(v_surface))
    y = np.outer(np.sin(u_surface), np.sin(v_surface))
    z = np.outer(np.cos(u_surface), np.ones_like(v_surface))

    return hist, x, y, z