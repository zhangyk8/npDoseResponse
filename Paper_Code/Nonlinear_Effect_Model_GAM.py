#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Yikun Zhang
Last Editing: November 16, 2025

Bootstrap inference on the dose-response curve by linear additive spline model 
with a traditional regression adjustment form under the nonlinear effect model.
"""

import numpy as np
import pickle
from pygam import LinearGAM, s, te
import sys

job_id = int(sys.argv[1])
print(job_id)

#=======================================================================================#

# Simple effect model
for n in [500, 1000, 2000, 5000]:
    np.random.seed(123)
    S = np.concatenate([2*np.random.rand(n).reshape(-1,1) - 1, 
                        2*np.random.rand(n).reshape(-1,1) - 1], axis=1)
    Z = 4*S[:,0] + S[:,1]
    E = 0.2*np.random.rand(n) - 0.1
    T = np.cos(np.pi*Z**3) + Z/4 + E
    Y = T**2 + T + 10*Z + np.random.normal(loc=0, scale=1, size=n)
    X = np.concatenate([T.reshape(-1,1), S], axis=1)

    # Estimate the dose-response curve and its derivative 
    t_eval = np.linspace(min(T)+0.01, max(T)-0.01, 200)

    if job_id == 1:
        # Estimate the regression function via linear additive (spline) model
        gam = LinearGAM(s(0) + te(1,2)).fit(X, Y)
        mu_mat = np.zeros((n, t_eval.shape[0]))
        beta_mat = np.zeros((n, t_eval.shape[0]))
        for i in range(t_eval.shape[0]):
            X_mat = np.concatenate([t_eval[i]*np.ones((n,1)), X[:,1:]], axis=1)
            mu_mat[:,i] = gam.predict(X_mat)
        for j in range(n):
            eval_mat = np.concatenate([t_eval.reshape(-1,1), 
                                       np.repeat(X[j,1:].reshape(1,-1), t_eval.shape[0], axis=0)], axis=1)
            pd_t = gam.partial_dependence(term=0, X=eval_mat)
            beta_mat[j,:] = np.gradient(pd_t, t_eval[1]-t_eval[0])
        
        m_est = np.mean(mu_mat, axis=0)
        theta_est = np.mean(beta_mat, axis=0)
        
        # Fine-tune the spline degree
        gam_tune = LinearGAM(s(0) + te(1,2)).gridsearch(X, Y, keep_best=True)
        mu_mat = np.zeros((n, t_eval.shape[0]))
        beta_mat = np.zeros((n, t_eval.shape[0]))
        for i in range(t_eval.shape[0]):
            X_mat = np.concatenate([t_eval[i]*np.ones((n,1)), X[:,1:]], axis=1)
            mu_mat[:,i] = gam_tune.predict(X_mat)
        for j in range(n):
            eval_mat = np.concatenate([t_eval.reshape(-1,1), 
                                       np.repeat(X[j,1:].reshape(1,-1), t_eval.shape[0], axis=0)], axis=1)
            pd_t = gam_tune.partial_dependence(term=0, X=eval_mat)
            beta_mat[j,:] = np.gradient(pd_t, t_eval[1]-t_eval[0])
            
        m_est_tune = np.mean(mu_mat, axis=0)
        theta_est_tune = np.mean(beta_mat, axis=0)

    # Bootstrap the original data
    np.random.seed(job_id)
    ind = np.random.choice(n, size=n, replace=True)
    X_boot = X[ind,:]
    Y_boot = Y[ind]

    gam_boot = LinearGAM(s(0) + te(1,2)).fit(X_boot, Y_boot)
    mu_boot = np.zeros((n, t_eval.shape[0]))
    beta_boot = np.zeros((n, t_eval.shape[0]))
    for i in range(t_eval.shape[0]):
        X_mat_boot = np.concatenate([t_eval[i]*np.ones((n,1)), X_boot[:,1:]], axis=1)
        mu_boot[:,i] = gam_boot.predict(X_mat_boot)
    for j in range(n):
        eval_mat_boot = np.concatenate([t_eval.reshape(-1,1), 
                                   np.repeat(X_boot[j,1:].reshape(1,-1), t_eval.shape[0], axis=0)], axis=1)
        pd_t = gam_boot.partial_dependence(term=0, X=eval_mat_boot)
        beta_boot[j,:] = np.gradient(pd_t, t_eval[1]-t_eval[0])
        
    m_est_boot = np.mean(mu_boot, axis=0)
    theta_est_boot = np.mean(beta_boot, axis=0)
    
    # Fine-tune the spline degree
    gam_boot_tune = LinearGAM(s(0) + te(1, 2)).gridsearch(X_boot, Y_boot, keep_best=True)
    mu_boot = np.zeros((n, t_eval.shape[0]))
    beta_boot = np.zeros((n, t_eval.shape[0]))
    for i in range(t_eval.shape[0]):
        X_mat_boot = np.concatenate([t_eval[i]*np.ones((n,1)), X_boot[:,1:]], axis=1)
        mu_boot[:,i] = gam_boot_tune.predict(X_mat_boot)
    for j in range(n):
        eval_mat_boot = np.concatenate([t_eval.reshape(-1,1), 
                                        np.repeat(X_boot[j,1:].reshape(1,-1), t_eval.shape[0], axis=0)], axis=1)
        pd_t = gam_boot_tune.partial_dependence(term=0, X=eval_mat_boot)
        beta_boot[j,:] = np.gradient(pd_t, t_eval[1]-t_eval[0])
        
    m_est_boot_tune = np.mean(mu_boot, axis=0)
    theta_est_boot_tune = np.mean(beta_boot, axis=0)
    
    if job_id == 1:
        with open('./Results_New/Nonlinear_Eff_GAM_'+str(job_id)+'_n'+str(n)+'.dat', "wb") as file:
            pickle.dump([m_est, m_est_tune, theta_est, theta_est_tune, 
                         m_est_boot, m_est_boot_tune, theta_est_boot, theta_est_boot_tune], file)
    else:
        with open('./Results_New/Nonlinear_Eff_GAM_'+str(job_id)+'_n'+str(n)+'.dat', "wb") as file:
            pickle.dump([m_est_boot, m_est_boot_tune, theta_est_boot, theta_est_boot_tune], file)
            