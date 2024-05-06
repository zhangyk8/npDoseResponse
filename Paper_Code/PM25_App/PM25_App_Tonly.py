#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Yikun Zhang
Last Editing: April 18, 2024

Apply the local polynomial regression with bootstrap inference to the PM2.5 CMR data
with treatment only.
"""

import numpy as np
import pickle
from NPDoseResponse import LocalPolyReg1D
import sys
import pandas as pd

job_id = int(sys.argv[1])
print(job_id)

#=======================================================================================#

dat = pd.read_csv('PM25_county_CMR_avg.csv')
n = dat.shape[0]
T = dat['PM2.5'].values
# X = dat[['PM2.5', 'lng', 'lat', 'population_2000',
#        'civil_unemploy', 'median_HH_inc', 'femaleHH_ns_pct', 'vacant_HHunit',
#        'owner_occ_pct', 'eduattain_HS', 'pctfam_pover']].values
Y = dat['CMR'].values

# Estimate the dose-response curve and its derivative 
t_qry = np.linspace(min(T)+0.01, max(T)-0.01, 200)

m_est = LocalPolyReg1D(Y, T, h=7*np.std(T), x_eval=t_qry, degree=1, deriv_ord=0, kernel='epanechnikov')

theta_est = LocalPolyReg1D(Y, T, h=7*np.std(T), x_eval=t_qry, degree=2, deriv_ord=1, kernel='epanechnikov')

# Bootstrap the original data
np.random.seed(job_id)
ind = np.random.choice(n, size=n, replace=True)
T_boot = T[ind]
Y_boot = Y[ind]

# Estimate the dose-response curve and its derivative on the bootstrapping data
m_est_boot = LocalPolyReg1D(Y_boot, T_boot, h=7*np.std(T_boot), x_eval=t_qry, degree=1, 
                                deriv_ord=0, kernel='epanechnikov')

theta_est_boot = LocalPolyReg1D(Y_boot, T_boot, h=7*np.std(T_boot), x_eval=t_qry, degree=2, 
                            deriv_ord=1, kernel='epanechnikov')


with open('./Results_New/PM25_App_Tonly_Bootstrap_'+str(job_id)+'_new2.dat', "wb") as file:
    pickle.dump([theta_est, m_est, theta_est_boot, m_est_boot], file)
