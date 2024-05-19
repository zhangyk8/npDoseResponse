# -*- coding: utf-8 -*-

# Author: Yikun Zhang
# Last Editing: May 19, 2024

# Description: This script contains the utility functions for the main functions 
# in our package.

import numpy as np
from .rbf import KernelRetrieval

#=======================================================================================#


def RoTBWLocalPoly(Y, X, kernT="epanechnikov", kernS="epanechnikov", C_h=7, C_b=3):
    '''
    Compute the rule-of-thumb bandwidth selector in Eq.(A1) of 
    Yang and Tschernig (1999).
    
    Parameters
    ----------
        Y: (n,)-array
            The outcomes of n observations.
        
        X: (n,d+1)-array
            The first column of X is the treatment/exposure variable, while 
            the other d columns are confounding variables of n observations.
            
        kernT, kernS: str
            The names of kernel functions for the treatment/exposure variable 
            and confounding variables. (Default: "epanechnikov".)
            
        C_h,C_b: float
            The scaling factors for the rule-of-thumb bandwidth parameters. 
            (Default: C_h=7, C_b=3.)
    
    Return
    ----------
        h: float
            The rule-of-thumb bandwidth parameter for the treatment/exposure variable.
            
        b: (d,)-array
            The rule-of-thumb bandwidth vector for the confounding variables.
    '''
    n = X.shape[0]  ## Number of data points
    d = X.shape[1] - 1
    
    kernT, sigmaK_sq, K_sq = KernelRetrieval(kernT)
    # Apply the rule-of-thumb bandwidth selector in Eq.(A1) of Yang and Tschernig (1999)
    p_coeff = np.polyfit(X[:,0], Y, 4)
    sec_deriv = 12*p_coeff[0]*X[:,0] + 6*p_coeff[1]*X[:,0] + 2*X[:,0]
    C_fun = np.mean(sec_deriv**2)
    T = X[:,0].reshape(-1,1)
    lhs = np.concatenate([np.ones((n,1)), X, T**2, T**3, T**4], axis=1)
    rcond = np.finfo(lhs.dtype).eps * max(*lhs.shape)
    beta = np.linalg.lstsq(lhs, Y, rcond=rcond)[0]
    # Compute the residual sum of squares
    resid = np.sum((Y - np.dot(lhs, beta))**2) * (np.max(X[:,0]) - np.min(X[:,0]))/ (n-5)
    sigmaK_sq = sigmaK_sq**2
    # ROT
    h = ((K_sq*resid)/(4*n*sigmaK_sq*C_fun))**(1/5) * (n**(d/(5*(d+5)))) * C_h
    
    kernS, sigmaK_sq, K_sq = KernelRetrieval(kernS)
    sec_deriv = np.zeros((n, d))
    for i in range(1, d+1):
        # Fit a fourth-order polynomial to each confounding variable
        p_coeff = np.polyfit(X[:,i], Y, 4)
        sec_deriv[:,i-1] = 12*p_coeff[0]*X[:,i] + 6*p_coeff[1]*X[:,i] + 2*X[:,i]
    C_fun = np.sum(np.diag(np.dot(sec_deriv.T, sec_deriv))/n)
    lhs = np.concatenate([np.ones((n,1)), X, T**2, T**3, T**4], axis=1)
    # lhs = np.concatenate([np.ones((n,1)), X[:,1:]], axis=1)
    rcond = np.finfo(lhs.dtype).eps * max(*lhs.shape)
    beta = np.linalg.lstsq(lhs, Y, rcond=rcond)[0]
    resid = np.sum((Y - np.dot(lhs, beta))**2) * (np.max(X[:,1:], axis=0) - np.min(X[:,1:], axis=0)) / (n-5)
    sigmaK_sq = sigmaK_sq**2
    K_sq = K_sq**d
    b = ((K_sq*d*resid)/(4*n*sigmaK_sq*C_fun))**(1/(d+5)) * C_b
    
    return h, b


def HatMatrix(X, degree=2, deriv_ord=1, h=None, b=None, print_bw=True, 
              kernT="epanechnikov", kernS="epanechnikov"):
    '''
    Compute the hat matrix of the local polynomial regression when it is viewed 
    as a linear smoother.
    
    Parameters
    ----------
        X: (n,d+1)-array
            The first column of X is the treatment/exposure variable, while 
            the other d columns are confounding variables of n observations.
            
        degree: int
            Degree of local polynomials. (Default: degree=2.)
            
        deriv_ord: int
            The order of the estimated derivative the conditional mean outcome function. 
            (Default: deriv_ord=1. Then, it estimates the partial derivative of the 
            conditional mean outcome function with respect to the treatment variable.)
            
        h,b: float
            The bandwidth parameters for the treatment/exposure variable and 
            confounding variables. 
            
        print_bw: boolean
            The indicator of whether the current bandwidth parameters should be
            printed to the console. (Default: print_bw=True.)
            
        kernT, kernS: str
            The names of kernel functions for the treatment/exposure variable 
            and confounding variables. (Default: "epanechnikov".)
            
    Return
    ----------
        hat_mat: (n,n)-array
            The hat matrix.
    '''
    n = X.shape[0]  ## Number of data points
    d = X.shape[1] - 1
    x_eval = X.copy()
    
    if print_bw:
        print("The current bandwidth for treament variable in the local polynomial regression is "+ str(h) + ".\n")
        print("The current bandwidth for confounding variables in the local polynomial regression is "+ str(b) + ".\n")
    
    kernT, sigmaK_sq, K_sq = KernelRetrieval(kernT)
    kernS, sigmaK_sq, K_sq = KernelRetrieval(kernS)
    hat_mat = np.zeros((n, n))
    for i in range(x_eval.shape[0]):
        weights = kernT((X[:,0] - x_eval[i,0])/h) * np.prod(kernS((X[:,1:] - x_eval[i,1:])/b), axis=1)
        # Filter out the data points with zero weights to speed up regressions with kernels of compact support
        inds = np.where(np.abs(weights) > 1e-26)[0]
        X_dat = np.zeros((n, degree+1+d))
        # X_dat[:,:(degree+1)] = (((X[:,0] - x_eval[i,0])/h).reshape(-1,1))**(np.arange(degree+1))
        for p in range(degree + 1):
            X_dat[:,p] = ((X[:,0] - x_eval[i,0])/h)**p
        X_dat[:,(degree+1):] = (X[:,1:] - x_eval[i,1:])/b
        X_dat = X_dat[inds,:]
        W = np.diag(weights[inds])
        design_mat = np.dot(np.dot(X_dat.T, W), X_dat)
        try:
            hat_mat_samp = np.dot(np.linalg.inv(design_mat), np.dot(X_dat.T, W))
        except:
            # Add some small quantities to the diagonal matrix to prevent the singularity of the matrix
            design_mat = design_mat + 1e-16*np.eye(design_mat.shape[0])
            hat_mat_samp = np.dot(np.linalg.inv(design_mat), np.dot(X_dat.T, W))
        hat_mat[i,inds] = hat_mat_samp[deriv_ord,:]
    return hat_mat
