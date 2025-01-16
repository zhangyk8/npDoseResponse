#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Yikun Zhang
Last Editing: April 16, 2024

Apply the simple integral estimator with bootstrap inference to the PM2.5 CMR data
with only spatial confounders.
"""

import numpy as np
import pickle
from NPDoseResponse import DerivEffect, IntegEst, RegAdjust, RoTBWLocalPoly
import sys
import pandas as pd

job_id = int(sys.argv[1])
print(job_id)

#=======================================================================================#

dat = pd.read_csv('PM25_county_CMR_avg.csv')
n = dat.shape[0]
T = dat['PM2.5'].values
X = dat[['PM2.5', 'lng', 'lat']].values
Y = dat['CMR'].values

# Estimate the dose-response curve and its derivative 
t_qry = np.linspace(min(T)+0.01, max(T)-0.01, 200)

if job_id == 1:
    h_cur, b_cur = RoTBWLocalPoly(Y, X, kernT="epanechnikov", kernS="epanechnikov", 
                                  C_h=16*np.std(T), C_b=23*np.std(X[:,1:], axis=0))

    theta_est = DerivEffect(Y, X, t_eval=t_qry, h_bar=None, kernT_bar="gaussian", 
                h=h_cur, b=b_cur, degree=2, deriv_ord=1, kernT="epanechnikov", kernS="epanechnikov")

    m_est = IntegEst(Y, X, t_eval=t_qry, h_bar=None, kernT_bar="gaussian", 
                    h=h_cur, b=b_cur, degree=2, deriv_ord=1, kernT="epanechnikov", kernS="epanechnikov")

    Y_RA = RegAdjust(Y, X, t_eval=t_qry, degree=2, deriv_ord=0, h=h_cur, b=h_cur, 
                     kernT="epanechnikov", kernS="epanechnikov")

    Y_RA_deriv = RegAdjust(Y, X, t_eval=t_qry, degree=2, deriv_ord=1, h=h_cur, b=b_cur,
                     kernT="epanechnikov", kernS="epanechnikov")

# Bootstrap the original data
np.random.seed(job_id)
ind = np.random.choice(n, size=n, replace=True)
X_boot = X[ind,:]
Y_boot = Y[ind]

h_boot, b_boot = RoTBWLocalPoly(Y_boot, X_boot, kernT="epanechnikov", kernS="epanechnikov", 
                              C_h=16*np.std(X_boot[:,0]), C_b=23*np.std(X_boot[:,1:], axis=0))

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
    with open('./Results_New/PM25_App_TS_Bootstrap_'+str(job_id)+'_new_bw.dat', "wb") as file:
        pickle.dump([theta_est, m_est, Y_RA, Y_RA_deriv, theta_est_boot, m_est_boot, Y_RA_boot, Y_RA_deriv_boot], file)
else:
    with open('./Results_New/PM25_App_TS_Bootstrap_'+str(job_id)+'_new_bw.dat', "wb") as file:
        pickle.dump([theta_est_boot, m_est_boot, Y_RA_boot, Y_RA_deriv_boot], file)

