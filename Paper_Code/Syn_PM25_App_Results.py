#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Yikun Zhang
Last Editing: Jan 9, 2024

Description: Synthesize the outputs from our simulation studies
"""

import numpy as np
import pandas as pd
import pickle

#=======================================================================================#

# Regress Y on T and spatial locations
B = 1000
theta_est_boot_arr1 = []
m_est_boot_arr1 = []
Y_RA_boot_arr1 = []
Y_RA_deriv_boot_arr1 = []
for b in range(1, B+1):
    if b == 1:
        with open('./Results_New/PM25_App_TS_Bootstrap_'+str(b)+'_new_bw.dat', "rb") as file:
            theta_est1, m_est1, Y_RA1, Y_RA_deriv1, theta_est_boot1, m_est_boot1, Y_RA_boot1, Y_RA_deriv_boot1 = pickle.load(file)
    else:
        with open('./Results_New/PM25_App_TS_Bootstrap_'+str(b)+'_new_bw.dat', "rb") as file:
            theta_est_boot1, m_est_boot1, Y_RA_boot1, Y_RA_deriv_boot1 = pickle.load(file)
    theta_est_boot_arr1.append(theta_est_boot1)
    m_est_boot_arr1.append(m_est_boot1)
    Y_RA_deriv_boot_arr1.append(Y_RA_deriv_boot1)
    Y_RA_boot_arr1.append(Y_RA_boot1)
    
theta_est_boot_arr1 = np.array(theta_est_boot_arr1)
m_est_boot_arr1 = np.array(m_est_boot_arr1)
Y_RA_deriv_boot_arr1 = np.array(Y_RA_deriv_boot_arr1)
Y_RA_boot_arr1 = np.array(Y_RA_boot_arr1)
    
with open('./Syn_Results/PM25_App_TS_new_bw.dat', "wb") as file:
    pickle.dump([theta_est1, m_est1, Y_RA1, Y_RA_deriv1, 
                 theta_est_boot_arr1, m_est_boot_arr1, Y_RA_deriv_boot_arr1, Y_RA_boot_arr1], file)
        
        

# Regress Y on T only
B = 1000
theta_est_boot_arr2 = []
m_est_boot_arr2 = []
for b in range(1, B+1):
    with open('./Results_New/PM25_App_Tonly_Bootstrap_'+str(b)+'_new_bw.dat', "rb") as file:
        theta_est2, m_est2, theta_est_boot2, m_est_boot2 = pickle.load(file)
    theta_est_boot_arr2.append(theta_est_boot2)
    m_est_boot_arr2.append(m_est_boot2)


theta_est_boot_arr2 = np.array(theta_est_boot_arr2)
m_est_boot_arr2 = np.array(m_est_boot_arr2)
    
with open('./Syn_Results/PM25_App_Tonly_new_bw.dat', "wb") as file:
    pickle.dump([theta_est2, m_est2, theta_est_boot_arr2, m_est_boot_arr2], file)
    
    

# Regress Y on T and all other covariates
B = 1000
theta_est_boot_arr3 = []
m_est_boot_arr3 = []
Y_RA_boot_arr3 = []
Y_RA_deriv_boot_arr3 = []
for b in range(1, B+1):
    if b == 1:
        with open('./Results_New/PM25_App_Full_Bootstrap_'+str(b)+'_new_bw.dat', "rb") as file:
            theta_est3, m_est3, Y_RA3, Y_RA_deriv3, theta_est_boot3, m_est_boot3, Y_RA_boot3, Y_RA_deriv_boot3 = pickle.load(file)
    else:
        with open('./Results_New/PM25_App_Full_Bootstrap_'+str(b)+'_new_bw.dat', "rb") as file:
            theta_est_boot3, m_est_boot3, Y_RA_boot3, Y_RA_deriv_boot3 = pickle.load(file)
    theta_est_boot_arr3.append(theta_est_boot3)
    m_est_boot_arr3.append(m_est_boot3)
    Y_RA_deriv_boot_arr3.append(Y_RA_deriv_boot3)
    Y_RA_boot_arr3.append(Y_RA_boot3)
    
theta_est_boot_arr3 = np.array(theta_est_boot_arr3)
m_est_boot_arr3 = np.array(m_est_boot_arr3)
Y_RA_deriv_boot_arr3 = np.array(Y_RA_deriv_boot_arr3)
Y_RA_boot_arr3 = np.array(Y_RA_boot_arr3)
    
with open('./Syn_Results/PM25_App_Full_new_bw.dat', "wb") as file:
    pickle.dump([theta_est3, m_est3, Y_RA3, Y_RA_deriv3, 
                 theta_est_boot_arr3, m_est_boot_arr3, Y_RA_deriv_boot_arr3, Y_RA_boot_arr3], file)
    