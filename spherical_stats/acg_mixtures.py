#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 21 12:59:29 2020

@author: tyrion
"""
import numpy as np
from spherical_stats import ACG, random_spd_matrix, sphere
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import acg
import esag
from scipy.optimize import minimize
from numba import njit, jit
from time import time
from scipy.stats import random_correlation
from copy import deepcopy

cov_1 = random_correlation.rvs((2.9,0.05,0.05))
cov_2 = random_correlation.rvs((0.05,0.05,2.9))
print(cov_1 - cov_2)
print(cov_1)
print(cov_2)
'''
cov_1 = np.array([[1.15506578, 1.0871748 , 0.74831267],\
       [1.0871748 , 1.23689076, 0.8049742 ],\
       [0.74831267, 0.8049742 , 0.60804345]])

cov_2 = np.array([[0.79920515, 0.52283267, 0.9488305 ],\
       [0.52283267, 0.4445119 , 0.54286523],\
       [0.9488305 , 0.54286523, 1.75628295]])
print(cov_1 - cov_2)
'''
acg_1 = ACG(cov_1)
#print(acg_1.cov_matrix)
acg_2 = ACG(cov_2)

w_1 = float(0.7)
w_2 = float(float(1) - w_1)
#print("w2: {}".format(w_2))

samples = 5000

samples_acg_1 = acg_1.rvs(int(w_1 * samples))
samples_acg_2 = acg_2.rvs(int(w_2 * samples))

mix_samples = np.concatenate((samples_acg_1, samples_acg_2), axis = 0)
#print(mix_samples.shape)

true_params = [w_1, w_2, cov_1/np.trace(cov_1), cov_2/np.trace(cov_2)]

'''
x, y, z = sphere(20)

ax = plt.axes(projection='3d')
ax.plot_surface(x, y, z, zorder=0, alpha=0.2,\
            rstride=1, cstride=1, antialiased=False, linewidth=0)
ax.scatter(samples_acg_1[:,0],samples_acg_1[:,1], samples_acg_1[:,2], c='r', s = 5)
ax.scatter(samples_acg_2[:,0],samples_acg_2[:,1], samples_acg_2[:,2], c='k', s = 5)
plt.show()
'''

#params = (w_1, w_2, cov_1, cov_2)
@njit(cache = True)
def compute_Sjk(k, weights, cov_matrices, vec):
    
    pdf_k = weights[k] * acg.likelihood(vec, cov_matrices[k, :, :])
    num = weights[0] * acg.likelihood(vec, cov_matrices[0, :, :]) + \
        weights[1] * acg.likelihood(vec, cov_matrices[1, :, :]) 
    
    sjk = pdf_k/num
    
    return sjk

@njit(cache = True)
def all_sjk(vecs, weights, cov_matrices):
    #print("comp. sjk")

    n_samples = vecs.shape[0]
    sjk_array = np.empty((n_samples,2))
    
    for _ in range(n_samples):
        
        vec = vecs[_, :]
        
        for k in range(2):
            
            sjk_array[_, k] = compute_Sjk(k, weights, cov_matrices, vec)
            
    return sjk_array

@njit(cache = True)
def compute_Q(vectors, sjk_array, weights, cov_matrices):
    
    #print("comp. Q")
    n_samples = vectors.shape[0]
    
    Q = 0
    
    for _ in range(n_samples):
        
        vec = vectors[_, :]
                
        for w in range(2):
                        
            Q += sjk_array[_, w] * np.log(weights[w]*acg.likelihood(vec, cov_matrices[w, :, :]))
            
    return Q

@njit(cache = True)         
def update_weights(sjk_array):
    #print("updating weights")
    n_samples = sjk_array.shape[0]
    weights = 1/n_samples * sjk_array.sum(axis = 0)
    
    return weights

@njit(cache = True)
def cost_fun(params, sjk_column, vecs):
    
    A = np.array([[params[0],params[1],params[2]],\
                  [0,params[3],params[4]], \
                      [0,0,params[5]]])
    x = A@A.T
    #print(3*x/np.trace(x))
    #acg = ACG(x)
    
    log_pdf_vals = np.log(acg._pdf(vecs, x))
    
    fun = - np.sum(sjk_column * log_pdf_vals)

    return fun

def EM_ACGM(vecs, weights, cov_matrices):
        
    qdiff = 1e12
    
    q = 0
    i = 0
    paramdiff = 100
    matrixnorms = np.array([1e12,1e12])
    converged = False
    
    tol = 1e-6
    rel_tol = 1e-6
    rel_error = float(1)
    
    while (paramdiff > tol and rel_error > rel_tol):
    #for _ in range(500):
        
        print("")
        print("Iteration: {}".format(i))
        #print("Diff Q: {}".format(qdiff))
        print("current weights: {}".format(weights))
        #print("current covariances: {}".format(cov_matrices))
        #print("True params: {}".format(true_params))
        print("Weights Diff Norm: {}".format(paramdiff))
        #print("Diff cov 1: {}".format(params[2] - true_params[2]))
        #print("Diff cov 2: {}".format(params[3] - true_params[3]))
        q_last = q
        last_weights = deepcopy(weights)
        last_cov_matrices = deepcopy(cov_matrices)
        #expectation step
        t0 = time()
        sjk_array = all_sjk(vecs, weights, cov_matrices)
        #print("sjk time: {}".format(time() - t0))        
        #print("sjk: {}".format(sjk_array.shape))
        t0 = time()
        q = compute_Q(vecs, sjk_array, weights, cov_matrices)
        
        #print("q time: {}".format(time() - t0))        
        
        ##maximization step
        
        #update weights
        
        weights = update_weights(sjk_array)
        #print("updated weights: {}".format(updated_weights))
        #params[0] = updated_weights[0]
        
        #params[1] = 1 - updated_weights[0]
        
        #update covariance matrices
        
        for _ in range(2):
            
            if (matrixnorms[_] > 1e-6 or i % 5 == 0):
                
                #cholesky decomposition of current covariance matrix estimates
                #print("cov before optimization: {}".format(params[_ +2]))
                A = np.linalg.cholesky(cov_matrices[_])
                #print("A: {}".format(A))
                #extract upper triangle of A
                
                A = A[np.tril_indices(3)]
                #print("optimizing cov {}".format(_))
                #print("Starting with: {}".format(A))
                #execute optimization
                #t0 = time()
                
                opt_res = minimize(cost_fun, A, args = (sjk_array[:, _], vecs), \
                                   method='L-BFGS-B')
                #print("opt time: {}".format(time() - t0))
                A_updated = np.array([[opt_res.x[0],opt_res.x[1],opt_res.x[2]],\
                      [0,opt_res.x[3],opt_res.x[4]], [0,0,opt_res.x[5]]])
                A_cov = A_updated@A_updated.T
                cov_matrices[_, :, :] = A_cov/np.trace(A_cov)
                #print("cov after optimization: {}".format(params[_ +2]))            
                optimized_fun = opt_res.fun
                n_iters = opt_res.nit
                #print("L-BFGS-B iterations: {}".format(n_iters))
               
        rel_error = abs((q - q_last)/q)
        print("rel change logl: {}".format(rel_error))
        #print("qdiff: {}".format(qdiff))
        #if qdiff < 0:
            
        #    break
        #print(params)
        #print("last params: {}".format(last_params))
        paramdiff = np.linalg.norm(weights - last_weights)
        matrixdiff = cov_matrices - last_cov_matrices           
        matrixnorms = np.array([acg.frobeniusnorm(matrixdiff[0, ...]), \
                                acg.frobeniusnorm(matrixdiff[1, ...])])
        #print("matrixnorms: {}".format(matrixnorms))
        i +=1
        if i > 300:
            
            converged = False
            print("Expectation-Maximization did not converge within 200 iterations")
            break
        
        if (paramdiff < tol or rel_error < rel_tol):
            
            converged = True
            
    return converged, weights, cov_matrices

def fit_ACGM(vecs, n_retry = 10):
    
    for _ in range(n_retry):
        print("Expectation Maximization run #{}".format(_ +1))
        initial_weights = np.random.dirichlet((1, 1), 1).ravel()
        print(initial_weights)
        initial_cov_matrices = np.stack((random_spd_matrix(), random_spd_matrix()), axis=0)
        print(initial_cov_matrices)
        converged, weights, cov_matrices = EM_ACGM(vecs, initial_weights,\
                                                   initial_cov_matrices)
            
        if converged == True:
            
            break
        
    if converged == False:
        
        print("ACG Mixture fitting failed")
        weights, cov_matrices = None
        
    return converged, weights, cov_matrices

start_weights = np.array([0.5, 0.5])
start_covs = np.stack((random_spd_matrix(), random_spd_matrix()), axis=0)

#print(start_covs.shape)
start_params = [0.5, 0.5, random_spd_matrix(), random_spd_matrix()]

conv, weights, covs = fit_ACGM(mix_samples)
print("Fitted weights: {}".format(weights))
print("Fitted covs: {}".format(covs))
print("true weights: {}".format(np.array([w_1, w_2])))
print("true covariances: {}".format([cov_1/np.trace(cov_1), cov_2/np.trace(cov_2)]))