#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Yikun Zhang
Last Editing: January 6, 2025

Bootstrap inference on the dose-response curve by our integral estimator and 
traditional regression adjustment estimator under the single confounder model.
"""

import numpy as np
import pickle
from NPDoseResponse import DerivEffect, IntegEst, RoTBWLocalPoly, RegAdjust
import sys

job_id = int(sys.argv[1])
print(job_id)

#=======================================================================================#

# Single confounder model
for n in [500, 1000, 2000, 5000]:
    np.random.seed(123)
    S1 = 2*np.random.rand(n) - 1
    E = np.random.rand(n)*0.6 - 0.3
    T1 = np.sin(np.pi*S1) + E
    Y1 = T1**2 + T1 + 1 + 10*S1 + np.random.normal(loc=0, scale=1, size=n)
    X1 = np.concatenate([T1.reshape(-1,1), S1.reshape(-1,1)], axis=1)

    # Estimate the dose-response curve and its derivative 
    t_qry1 = np.linspace(min(T1)+0.01, max(T1)-0.01, 200)
    
    para = False

    if job_id == 1:
        h_cur1, b_cur1 = RoTBWLocalPoly(Y1, X1, kernT="epanechnikov", kernS="epanechnikov")

        theta_est1 = DerivEffect(Y1, X1, t_eval=t_qry1, h_bar=None, kernT_bar="gaussian", 
                                 h=h_cur1, b=b_cur1, degree=2, deriv_ord=1, kernT="epanechnikov", 
                                 kernS="epanechnikov", parallel=para, processes=60)

        m_est1 = IntegEst(Y1, X1, t_eval=t_qry1, h_bar=None, kernT_bar="gaussian", 
                            h=h_cur1, b=b_cur1, degree=2, deriv_ord=1, kernT="epanechnikov", 
                            kernS="epanechnikov", parallel=para, processes=60)

        Y_RA1 = RegAdjust(Y1, X1, t_eval=t_qry1, degree=2, deriv_ord=0, h=h_cur1, b=b_cur1, 
                         kernT="epanechnikov", kernS="epanechnikov", parallel=para, processes=60)

        Y_RA_deriv1 = RegAdjust(Y1, X1, t_eval=t_qry1, degree=2, deriv_ord=1, h=h_cur1, b=b_cur1,
                         kernT="epanechnikov", kernS="epanechnikov", parallel=para, processes=60)


    # Bootstrap the original data
    np.random.seed(job_id)
    ind = np.random.choice(n, size=n, replace=True)
    X1_boot = X1[ind,:]
    Y1_boot = Y1[ind]

    h_boot1, b_boot1 = RoTBWLocalPoly(Y1_boot, X1_boot, kernT="epanechnikov", kernS="epanechnikov")

    # Estimate the dose-response curve and its derivative on the bootstrapping data
    theta_est1_boot = DerivEffect(Y1_boot, X1_boot, t_eval=t_qry1, h_bar=None, kernT_bar="gaussian", 
                                 h=h_boot1, b=b_boot1, degree=2, deriv_ord=1, kernT="epanechnikov", 
                                 kernS="epanechnikov", parallel=para, processes=60)

    m_est1_boot = IntegEst(Y1_boot, X1_boot, t_eval=t_qry1, h_bar=None, kernT_bar="gaussian", 
                             h=h_boot1, b=b_boot1, degree=2, deriv_ord=1, kernT="epanechnikov", 
                             kernS="epanechnikov", parallel=para, processes=60)

    Y_RA1_boot = RegAdjust(Y1_boot, X1_boot, t_eval=t_qry1, degree=2, deriv_ord=0, h=h_boot1, b=b_boot1, 
                     kernT="epanechnikov", kernS="epanechnikov", parallel=para, processes=60)

    Y_RA_deriv1_boot = RegAdjust(Y1_boot, X1_boot, t_eval=t_qry1, degree=2, deriv_ord=1, h=h_boot1, b=b_boot1,
                     kernT="epanechnikov", kernS="epanechnikov", parallel=para, processes=60)

    if job_id == 1:
        with open('./Results_New/Single_Conf_Bootstrap_'+str(job_id)+'_new_bw_n'+str(n)+'.dat', "wb") as file:
            pickle.dump([theta_est1, m_est1, Y_RA1, Y_RA_deriv1, theta_est1_boot, m_est1_boot, Y_RA1_boot, Y_RA_deriv1_boot], file)
    else:
        with open('./Results_New/Single_Conf_Bootstrap_'+str(job_id)+'_new_bw_n'+str(n)+'.dat', "wb") as file:
            pickle.dump([theta_est1_boot, m_est1_boot, Y_RA1_boot, Y_RA_deriv1_boot], file)

