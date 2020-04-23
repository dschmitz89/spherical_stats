#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 20:58:05 2020

@author: tyrion
"""
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

def plot_sphere(axis, n_grid, dist = None):
    
    u = np.linspace(0, np.pi, n_grid)
    
    v = np.linspace(0, 2 * np.pi, n_grid)

    x = np.outer(np.sin(u), np.sin(v))
    y = np.outer(np.sin(u), np.cos(v))
    z = np.outer(np.cos(u), np.ones_like(v))
        
    if dist is not None:
        
        u_grid, v_grid = np.meshgrid(u, v)
    
        vertex_vecs = np.array([np.sin(v_grid)*np.sin(u_grid), \
                            np.cos(v_grid)*np.sin(u_grid),np.cos(u_grid)])
    
        vertex_vecs = vertex_vecs.reshape(3,n_grid*n_grid)
        
        #calculate probability density for the faces
        
        pdfs_faces = dist.pdf(vertex_vecs.T).reshape(n_grid,n_grid)
        
        #create colormap for the faces
        
        normed_facecolors = plt.Normalize(vmin=pdfs_faces.min(), \
                                          vmax=pdfs_faces.max())
        
        facecolors = cm.viridis(normed_facecolors(pdfs_faces.T))

        axis.plot_surface(x, y, z, zorder=0, facecolors=facecolors, \
                        rstride=1, cstride=1, antialiased=False, linewidth=0)    
    else:
        
        axis.plot_surface(x, y, z, zorder=0, color='b', \
                        rstride=1, cstride=1, antialiased=False, linewidth=0)  