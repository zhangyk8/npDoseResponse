#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Yikun Zhang
Last Editing: April 30, 2024

Bootstrap inference on the dose-response curve by the simple integral estimator
and traditional regression adjustment estimator under the simple effect model 1.
"""

import numpy as np
import pickle
from NPDoseResponse import DerivEffect, IntegEst, RoTBWLocalPoly, RegAdjust
import sys

job_id = int(sys.argv[1])
print(job_id)

#=======================================================================================#

# Simple effect model
n = 2000
np.random.seed(123)
S = np.concatenate([2*np.random.rand(n).reshape(-1,1) - 1, 
                    2*np.random.rand(n).reshape(-1,1) - 1], axis=1)
Z = 4*S[:,0] + S[:,1]
E = 0.2*np.random.rand(n) - 0.1
T = np.cos(np.pi*Z**3) + Z/4 + E
Y = T**2 + T + 10*Z + np.random.normal(loc=0, scale=1, size=n)
X = np.concatenate([T.reshape(-1,1), S], axis=1)

# Estimate the dose-response curve and its derivative 
t_qry = np.linspace(min(T)+0.01, max(T)-0.01, 200)

if job_id == 1:
    h_cur, b_cur = RoTBWLocalPoly(Y, X, kernT="epanechnikov", kernS="epanechnikov")

    theta_est = DerivEffect(Y, X, t_eval=t_qry, h_bar=None, kernT_bar="gaussian", 
                h=h_cur, b=b_cur, degree=2, deriv_ord=1, kernT="epanechnikov", kernS="epanechnikov")

    m_est = IntegEst(Y, X, t_eval=t_qry, h_bar=None, kernT_bar="gaussian", 
                    h=h_cur, b=b_cur, degree=2, deriv_ord=1, kernT="epanechnikov", kernS="epanechnikov")

    Y_RA = RegAdjust(Y, X, t_eval=t_qry, degree=2, deriv_ord=0, h=h_cur, b=h_cur, 
                     kernT="epanechnikov", kernS="epanechnikov")

    Y_RA_deriv = RegAdjust(Y, X, t_eval=t_qry, degree=2, deriv_ord=1, h=h_cur, b=b_cur,
                     kernT="epanechnikov", kernS="epanechnikov")

# var_est = AsymVarSurrogate(Y, X, t_eval=t_qry, h=None, kernT='epanechnikov', 
#                            kernS="epanechnikov", h_den=None)

# Bootstrap the original data
np.random.seed(job_id)
ind = np.random.choice(n, size=n, replace=True)
X_boot = X[ind,:]
Y_boot = Y[ind]

h_boot, b_boot = RoTBWLocalPoly(Y_boot, X_boot, kernT="epanechnikov", kernS="epanechnikov")

# Estimate the dose-response curve and its derivative on the bootstrapping data
theta_est_boot = DerivEffect(Y_boot, X_boot, t_eval=t_qry, h_bar=None, kernT_bar="gaussian", 
                             h=h_boot, b=b_boot, degree=2, deriv_ord=1, kernT="epanechnikov", 
                             kernS="epanechnikov")

m_est_boot = IntegEst(Y_boot, X_boot, t_eval=t_qry, h_bar=None, kernT_bar="gaussian", 
                        h=h_boot, b=b_boot, degree=2, deriv_ord=1, kernT="epanechnikov", kernS="epanechnikov")

Y_RA_boot = RegAdjust(Y_boot, X_boot, t_eval=t_qry, degree=2, deriv_ord=0, h=h_boot, b=b_boot, 
                 kernT="epanechnikov", kernS="epanechnikov")

Y_RA_deriv_boot = RegAdjust(Y_boot, X_boot, t_eval=t_qry, degree=2, deriv_ord=1, h=h_boot, b=b_boot,
                 kernT="epanechnikov", kernS="epanechnikov")

if job_id == 1:
    with open('./Results_New/Simple_Eff_Bootstrap_'+str(job_id)+'_new1.dat', "wb") as file:
        pickle.dump([theta_est, m_est, Y_RA, Y_RA_deriv, theta_est_boot, m_est_boot, Y_RA_boot, Y_RA_deriv_boot], file)
else:
    with open('./Results_New/Simple_Eff_Bootstrap_'+str(job_id)+'_new1.dat', "wb") as file:
        pickle.dump([theta_est_boot, m_est_boot, Y_RA_boot, Y_RA_deriv_boot], file)