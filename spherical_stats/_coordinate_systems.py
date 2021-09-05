#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  2 16:13:25 2020

@author: tyrion
"""

import numpy as np

def vectors_to_polar(vectors, deg = False):
    
    phi = np.arctan2(vectors[:, 1],vectors[:, 0])
    theta = np.arccos(vectors[:, 2])
    
    if deg == True:
        
        phi = np.rad2deg(phi)
        theta = np.rad2deg(theta)
        
    return theta, phi

def polar_to_vectors(theta, phi, deg = False):
    
    if deg == True:
        
        phi = np.deg2rad(phi)
        theta = np.deg2rad(theta)
        
    sintheta = np.sin(theta)
    
    vecs = np.empty((theta.shape[0],3))
    vecs[:, 0] = sintheta*np.cos(phi)
    vecs[:, 1] = sintheta*np.sin(phi)
    vecs[:, 2] = np.cos(theta)
    
    return vecs

def vectors_to_geographical(vectors, deg = False):
    
    longitude = np.arctan2(vectors[:, 1],vectors[:, 0])
    latitude = np.arcsin(vectors[:, 2])
    
    if deg == True:
        
        latitude = np.rad2deg(latitude)
        longitude = np.rad2deg(longitude)
        
    return latitude, longitude

def geographical_to_vectors(latitude, longitude, deg = False):
    
    if deg == True:
        
        latitude = np.deg2rad(latitude)
        longitude = np.deg2rad(longitude)
        
    vectors = np.empty((latitude.shape[0],3))
    
    coslat = np.cos(latitude)
    
    vectors[:, 0] = coslat * np.cos(longitude)
    vectors[:, 1] = coslat * np.sin(longitude)
    vectors[:, 2] = np.sin(latitude)
    
    return vectors