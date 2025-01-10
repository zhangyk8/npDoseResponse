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

# Single Confounder Model
for n in [500, 1000, 2000, 5000]:
    B = 1000
    theta_est_boot_arr1 = []
    m_est_boot_arr1 = []
    Y_RA_boot_arr1 = []
    Y_RA_deriv_boot_arr1 = []
    for b in range(1, B+1):
        if b == 1:
            with open('./Results_New/Single_Conf_Bootstrap_'+str(b)+'_new_bw_n'+str(n)+'.dat', "rb") as file:
                theta_est1, m_est1, Y_RA1, Y_RA_deriv1, theta_est_boot1, m_est_boot1, Y_RA1_boot, Y_RA_deriv1_boot = pickle.load(file)
        else:
            with open('./Results_New/Single_Conf_Bootstrap_'+str(b)+'_new_bw_n'+str(n)+'.dat', "rb") as file:
                theta_est_boot1, m_est_boot1, Y_RA1_boot, Y_RA_deriv1_boot = pickle.load(file)
        theta_est_boot_arr1.append(theta_est_boot1)
        m_est_boot_arr1.append(m_est_boot1)
        Y_RA_deriv_boot_arr1.append(Y_RA_deriv1_boot)
        Y_RA_boot_arr1.append(Y_RA1_boot)
    theta_est_boot_arr1 = np.array(theta_est_boot_arr1)
    m_est_boot_arr1 = np.array(m_est_boot_arr1)

    Y_RA_deriv_boot_arr1 = np.array(Y_RA_deriv_boot_arr1)
    Y_RA_boot_arr1 = np.array(Y_RA_boot_arr1)
    
    with open('./Syn_Results/Single_Conf_new_bw_n'+str(n)+'.dat', "wb") as file:
        pickle.dump([theta_est1, m_est1, Y_RA1, Y_RA_deriv1, 
                     theta_est_boot_arr1, m_est_boot_arr1, Y_RA_deriv_boot_arr1, Y_RA_boot_arr1], file)
        
        

# Linear Confounding Model
for n in [500, 1000, 2000, 5000]:
    B = 1000
    theta_est3_boot_arr = []
    m_est3_boot_arr = []
    Y_RA_boot_arr3 = []
    Y_RA_deriv_boot_arr3 = []
    for b in range(1, B+1):
        if b == 1:
            with open('./Results_New/Linear_Conf_Bootstrap_'+str(b)+'_new_bw_n'+str(n)+'.dat', "rb") as file:
                theta_est3, m_est3, Y_RA3, Y_RA_deriv3, theta_est3_boot, m_est3_boot, Y_RA3_boot, Y_RA_deriv3_boot = pickle.load(file)
        else:
            with open('./Results_New/Linear_Conf_Bootstrap_'+str(b)+'_new_bw_n'+str(n)+'.dat', "rb") as file:
                theta_est3_boot, m_est3_boot, Y_RA3_boot, Y_RA_deriv3_boot = pickle.load(file)
        theta_est3_boot_arr.append(theta_est3_boot)
        m_est3_boot_arr.append(m_est3_boot)
        Y_RA_deriv_boot_arr3.append(Y_RA_deriv3_boot)
        Y_RA_boot_arr3.append(Y_RA3_boot)
    theta_est3_boot_arr = np.array(theta_est3_boot_arr)
    m_est3_boot_arr = np.array(m_est3_boot_arr)
    
    Y_RA_deriv_boot_arr3 = np.array(Y_RA_deriv_boot_arr3)
    Y_RA_boot_arr3 = np.array(Y_RA_boot_arr3)
    
    with open('./Syn_Results/Linear_Conf_new_bw_n'+str(n)+'.dat', "wb") as file:
        pickle.dump([theta_est3, m_est3, Y_RA3, Y_RA_deriv3, 
                     theta_est3_boot_arr, m_est3_boot_arr, Y_RA_deriv_boot_arr3, Y_RA_boot_arr3], file)
        

# Nonlinear Effect Model
for n in [500, 1000, 2000, 5000]:
    B = 1000
    theta_est_boot_arr = []
    m_est_boot_arr = []
    Y_RA_boot_arr = []
    Y_RA_deriv_boot_arr = []
    for b in range(1, B+1):
        if b == 1:
            with open('./Results_New/Nonlinear_Eff_Bootstrap_'+str(b)+'_new_bw_n'+str(n)+'.dat', "rb") as file:
                theta_est, m_est, Y_RA, Y_RA_deriv, theta_est_boot, m_est_boot, Y_RA_boot, Y_RA_deriv_boot = pickle.load(file)
        else:
            with open('./Results_New/Nonlinear_Eff_Bootstrap_'+str(b)+'_new_bw_n'+str(n)+'.dat', "rb") as file:
                theta_est_boot, m_est_boot, Y_RA_boot, Y_RA_deriv_boot = pickle.load(file)
        theta_est_boot_arr.append(theta_est_boot)
        m_est_boot_arr.append(m_est_boot)
        Y_RA_deriv_boot_arr.append(Y_RA_deriv_boot)
        Y_RA_boot_arr.append(Y_RA_boot)
    theta_est_boot_arr = np.array(theta_est_boot_arr)
    m_est_boot_arr = np.array(m_est_boot_arr)
    
    Y_RA_deriv_boot_arr = np.array(Y_RA_deriv_boot_arr)
    Y_RA_boot_arr = np.array(Y_RA_boot_arr)
    
    with open('./Syn_Results/Nonlinear_Eff_new_bw_n'+str(n)+'.dat', "wb") as file:
        pickle.dump([theta_est, m_est, Y_RA, Y_RA_deriv, 
                     theta_est_boot_arr, m_est_boot_arr, Y_RA_deriv_boot_arr, Y_RA_boot_arr], file)