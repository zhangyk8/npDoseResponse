#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Yikun Zhang
Last Editing: January 6, 2025

Bootstrap inference on the dose-response curve by our integral estimator and 
traditional regression adjustment estimator under the linear confounding model.
"""

import numpy as np
import pickle
from NPDoseResponse import DerivEffect, IntegEst, RoTBWLocalPoly, RegAdjust
import sys

job_id = int(sys.argv[1])
print(job_id)

#=======================================================================================#

# Linear confounding model
for n in [500, 1000, 2000, 5000]:
    np.random.seed(123)
    S3 = np.concatenate([2*np.random.rand(n).reshape(-1,1) - 1, 
                         2*np.random.rand(n).reshape(-1,1) - 1], axis=1)
    E3 = np.random.rand(n) - 0.5
    T3 = 2*S3[:,0] + S3[:,1] + E3
    Y3 = T3 + 6*S3[:,0] + 6*S3[:,1] + np.random.normal(loc=0, scale=1, size=n)
    X3 = np.concatenate([T3.reshape(-1,1), S3], axis=1)

    # Estimate the dose-response curve and its derivative 
    t_qry3 = np.linspace(min(T3)+0.01, max(T3)-0.01, 200)

    if job_id == 1:
        h_cur3, b_cur3 = RoTBWLocalPoly(Y3, X3, kernT="epanechnikov", kernS="epanechnikov")

        theta_est3 = DerivEffect(Y3, X3, t_eval=t_qry3, h_bar=None, kernT_bar="gaussian", 
                    h=h_cur3, b=b_cur3, degree=2, deriv_ord=1, kernT="epanechnikov", kernS="epanechnikov")

        m_est3 = IntegEst(Y3, X3, t_eval=t_qry3, h_bar=None, kernT_bar="gaussian", 
                        h=h_cur3, b=b_cur3, degree=2, deriv_ord=1, kernT="epanechnikov", kernS="epanechnikov")

        Y_RA3 = RegAdjust(Y3, X3, t_eval=t_qry3, degree=2, deriv_ord=0, h=h_cur3, b=b_cur3, 
                         kernT="epanechnikov", kernS="epanechnikov")

        Y_RA_deriv3 = RegAdjust(Y3, X3, t_eval=t_qry3, degree=2, deriv_ord=1, h=h_cur3, b=b_cur3,
                         kernT="epanechnikov", kernS="epanechnikov")


    # Bootstrap the original data
    np.random.seed(job_id)
    ind = np.random.choice(n, size=n, replace=True)
    X3_boot = X3[ind,:]
    Y3_boot = Y3[ind]

    h_boot3, b_boot3 = RoTBWLocalPoly(Y3_boot, X3_boot, kernT="epanechnikov", kernS="epanechnikov")

    # Estimate the dose-response curve and its derivative on the bootstrapping data
    theta_est3_boot = DerivEffect(Y3_boot, X3_boot, t_eval=t_qry3, h_bar=None, kernT_bar="gaussian", 
                                 h=h_boot3, b=b_boot3, degree=2, deriv_ord=1, kernT="epanechnikov", 
                                 kernS="epanechnikov")

    m_est3_boot = IntegEst(Y3_boot, X3_boot, t_eval=t_qry3, h_bar=None, kernT_bar="gaussian", 
                            h=h_boot3, b=b_boot3, degree=2, deriv_ord=1, kernT="epanechnikov", kernS="epanechnikov")

    Y_RA3_boot = RegAdjust(Y3_boot, X3_boot, t_eval=t_qry3, degree=2, deriv_ord=0, h=h_boot3, b=b_boot3, 
                     kernT="epanechnikov", kernS="epanechnikov")

    Y_RA_deriv3_boot = RegAdjust(Y3_boot, X3_boot, t_eval=t_qry3, degree=2, deriv_ord=1, h=h_boot3, b=b_boot3,
                     kernT="epanechnikov", kernS="epanechnikov")

    if job_id == 1:
        with open('./Results_New/Linear_Conf_Bootstrap_'+str(job_id)+'_new_bw_n'+str(n)+'.dat', "wb") as file:
            pickle.dump([theta_est3, m_est3, Y_RA3, Y_RA_deriv3, theta_est3_boot, m_est3_boot, Y_RA3_boot, Y_RA_deriv3_boot], file)
    else:
        with open('./Results_New/Linear_Conf_Bootstrap_'+str(job_id)+'_new_bw_n'+str(n)+'.dat', "wb") as file:
            pickle.dump([theta_est3_boot, m_est3_boot, Y_RA3_boot, Y_RA_deriv3_boot], file)