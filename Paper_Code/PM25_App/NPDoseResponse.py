#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Yikun Zhang
Last Editing: May 5, 2024

This file contains the implementation of local polynomial regression as well as 
our proposed integral estimator and its derivative estimator.
"""

import numpy as np
import rbf as rbf
from sklearn.model_selection import KFold
import ray

#=======================================================================================#

def LocalPolyReg(Y, X, x_eval=None, degree=2, deriv_ord=1, h=None, b=None, 
                 kernT="epanechnikov", kernS="epanechnikov", 
                 h_lst=np.linspace(0.5, 15, 30), b_lst=np.linspace(0.2, 6, 30)):
    '''
    (Partial) Local polynomial regression for estimating the conditional mean outcome 
    function and its partial derivatives. We use higher order local monomials for 
    the treatment variable and first-order local monomials for the confounding variables.
    
    
    Parameters
    ----------
        Y: (n,)-array
            The outcomes of n observations.
        
        X: (n,d+1)-array
            The first column of X is the treatment/exposure variable, while 
            the other d columns are confounding variables of n observations.
            
        x_eval: (m,d+1)-array
            The coordinates of the m evaluation points. (Default: x_eval=None. 
            Then, x_eval=X.)
            
        degree: int
            Degree of local polynomials. (Default: degree=2.)
            
        deriv_ord: int
            The order of the estimated derivative the conditional mean outcome 
            function. (Default: deriv_ord=1. Then, it estimates the partial 
            derivative of the conditional mean outcome function with respect to 
            the treatment variable.)
            
        h,b: float
            The bandwidth parameters for the treatment/exposure variable and 
            confounding variables. (Default: h=None, b=None. Then, the rule-of-thumb 
            bandwidth selector in Eq.(A1) of Yang and Tschernig (1999) is used.)
            
        kernT, kernS: str
            The names of kernel functions for the treatment/exposure variable 
            and confounding variables. (Default: "epanechnikov".)
    '''
    if x_eval is None: 
        x_eval = X.copy()
        
    if (h is None) and (b is None):
        # Apply the rule-of-thumb bandwidth selector in Eq.(A1) of Yang and Tschernig (1999)
        h, b = RoTBWLocalPoly(Y, X, kernT=kernT, kernS=kernS)
    elif h is None:
        h, b_can = RoTBWLocalPoly(Y, X, kernT=kernT, kernS=kernS)
    elif b is None:
        h_can, b = RoTBWLocalPoly(Y, X, kernT=kernT, kernS=kernS)
        
    if (h == "cv") and (b == "cv"):
        cv_mse = np.zeros((len(h_lst), len(b_lst)))
        for i in range(len(h_lst)):
            for j in range(len(b_lst)):
                Y_reg_est = LocalPolyRegMain(Y, X, x_eval=X, degree=1, deriv_ord=0, 
                                             h=h_lst[i], b=b_lst[i], 
                                             kernT=kernT, kernS=kernS)
                # Leave-one-out CV (with hat matrix trick)
                hat_mat = HatMatrix(X, degree=2, deriv_ord=1, h=h_lst[i], b=b_lst[i], kernT=kernT, kernS=kernS)
                cv_mse[i,j] = np.mean(((Y - Y_reg_est) / (1 - np.diag(hat_mat)))**2)
        argmin_ind = np.unravel_index(cv_mse.argmin(), cv_mse.shape)
        h_opt = h_lst[argmin_ind[0]]
        b_opt = b_lst[argmin_ind[1]]
        h = h_opt
        b = b_opt
        
    print("The current bandwidth for treament variable in the local polynomial regression is "+ str(h) + ".\n")
    print("The current bandwidth for confounding variables in the local polynomial regression is "+ str(b) + ".\n")
    
    Y_est = LocalPolyRegMain(Y, X, x_eval=x_eval, degree=degree, deriv_ord=deriv_ord, 
                             h=h, b=b, kernT=kernT, kernS=kernS)
    return Y_est


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
    '''
    n = X.shape[0]  ## Number of data points
    d = X.shape[1] - 1
    
    kernT, sigmaK_sq, K_sq = rbf.KernelRetrieval(kernT)
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
    
    kernS, sigmaK_sq, K_sq = rbf.KernelRetrieval(kernS)
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


def HatMatrix(X, degree=2, deriv_ord=1, h=None, b=None, kernT="epanechnikov", kernS="epanechnikov"):
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
            The bandwidth parameters for the treatment/exposure variable and confounding variables. 
            
        kernT, kernS: str
            The names of kernel functions for the treatment/exposure variable and confounding variables.
            (Default: "epanechnikov".)
    '''
    n = X.shape[0]  ## Number of data points
    d = X.shape[1] - 1
    x_eval = X.copy()
    
    print("The current bandwidth for treament variable in the local polynomial regression is "+ str(h) + ".\n")
    print("The current bandwidth for confounding variables in the local polynomial regression is "+ str(b) + ".\n")
    
    kernT, sigmaK_sq, K_sq = rbf.KernelRetrieval(kernT)
    kernS, sigmaK_sq, K_sq = rbf.KernelRetrieval(kernS)
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


def LocalPolyRegMain(Y, X, x_eval=None, degree=2, deriv_ord=0, h=None, b=None, 
                     kernT="epanechnikov", kernS="epanechnikov"):
    '''
    Main function for computing the local polynomial regression.
    
    
    Parameters
    ----------
        Y: (n,)-array
            The outcomes of n observations.
        
        X: (n,d+1)-array
            The first column of X is the treatment/exposure variable, while 
            the other d columns are confounding variables of n observations.
            
        x_eval: (m,d+1)-array
            The coordinates of the m evaluation points. (Default: x_eval=None. 
            Then, x_eval=X.)
            
        degree: int
            Degree of local polynomials. (Default: degree=2.)
            
        deriv_ord: int
            The order of the estimated derivative the conditional mean outcome 
            function. (Default: deriv_ord=1. Then, it estimates the partial 
            derivative of the conditional mean outcome function with respect to 
            the treatment variable.)
            
        h,b: float
            The bandwidth parameters for the treatment/exposure variable and 
            confounding variables. (Default: h=None, b=None. Then, the rule-of-thumb 
            bandwidth selector in Eq.(A1) of Yang and Tschernig (1999) is used.)
            
        kernT, kernS: str
            The names of kernel functions for the treatment/exposure variable 
            and confounding variables. (Default: "epanechnikov".)
    '''
    kernT, sigmaK_sq, K_sq = rbf.KernelRetrieval(kernT)
    kernS, sigmaK_sq, K_sq = rbf.KernelRetrieval(kernS)
    
    n = X.shape[0]  ## Number of data points
    d = X.shape[1] - 1
    Y_est = np.zeros((x_eval.shape[0],))
    for i in range(x_eval.shape[0]):
        weights = kernT((X[:,0] - x_eval[i,0])/h) * np.prod(kernS((X[:,1:] - x_eval[i,1:])/b), axis=1)
        # Filter out the data points with zero weights to speed up regressions with kernels of compact support
        inds = np.where(np.abs(weights) > 1e-26)[0]
        X_dat = np.zeros((n, degree+1+d))
        # X_dat[:,:(degree+1)] = ((X[:,0] - x_eval[i,0]).reshape(-1,1))**(np.arange(degree+1))
        for p in range(degree + 1):
            X_dat[:,p] = (X[:,0] - x_eval[i,0])**p
        X_dat[:,(degree+1):] = X[:,1:] - x_eval[i,1:]
        weight_sqrt = np.sqrt(weights)[inds]
        lhs = np.dot(np.diag(weight_sqrt), X_dat[inds,:])
        rhs = weight_sqrt*Y[inds]
        rcond = np.finfo(lhs.dtype).eps * max(*lhs.shape)
        beta = np.linalg.lstsq(lhs, rhs, rcond=rcond)[0]
        Y_est[i] = np.math.factorial(deriv_ord)*beta[deriv_ord]
    return Y_est



def DerivEffect(Y, X, t_eval=None, h_bar=None, kernT_bar="gaussian", 
                h=None, b=None, degree=2, deriv_ord=1, kernT="epanechnikov", kernS="epanechnikov"):
    '''
    Estimating the derivative of the dose-response curve via Nadaraya-Watson 
    conditional CDF estimator.
    
    
    Parameters
    ----------
        Y: (n,)-array
            The outcomes of n observations.
            
        X: (n,d+1)-array
            The first column of X is the treatment/exposure variable, while 
            the other d columns are confounding variables of n observations.
            
        t_eval: (m,)-array
            The coordinates of the m evaluation points. (Default: t_eval=None. 
            Then, t_eval=X[:,0].)
            
        h_bar: float
            The bandwidth parameters for the Nadaraya-Watson conditional CDF estimator. 
            (Default: h_bar=None. Then, the Silverman's rule of thumb is applied. 
            See Chen et al.(2016) for details.)
            
        kernT_bar: str
            The name of the kernel function for Nadaraya-Watson conditional CDF 
            estimator. (Default: "gaussian".)
            
        h,b: float
            The bandwidth parameters for the treatment/exposure variable and 
            confounding variables. (Default: h=None, b=None. Then, the rule-of-thumb 
            bandwidth selector in Eq.(A1) of Yang and Tschernig (1999) is used.)
            
        degree: int
            Degree of local polynomials. (Default: degree=2.)
            
        deriv_ord: int
            The order of the estimated derivative the conditional mean outcome function. 
            (Default: deriv_ord=1. Then, it estimates the partial derivative of the 
            conditional mean outcome function with respect to the treatment variable.)
            
        kernT, kernS: str
            The names of kernel functions for the treatment/exposure variable 
            and confounding variables. (Default: "epanechnikov".)
    '''
    
    if t_eval is None: 
        t_eval = X[:,0].copy()
        
    n = X.shape[0]  ## Number of data points
    d = 1
    if h_bar is None:
        # Apply the Silverman's rule of thumb bandwidth in Chen et al.(2016).
        h_bar = (4/(d+2))**(1/(d+4))*(n**(-1/(d+4)))*np.std(X[:,0])
        print("The current bandwidth for the conditional CDF estimator is "+ str(h_bar) + ".\n")
    
    kernT_bar, sigmaK_sq, K_sq = rbf.KernelRetrieval(kernT_bar)
        
    weight_mat = kernT_bar((t_eval - X[:,0].reshape(-1,1)) / h_bar)
    weight_mat = weight_mat / np.sum(weight_mat, axis=0)
    weight_mat[np.isnan(weight_mat)] = 0
    beta_mat = np.zeros((n, t_eval.shape[0]))
    for i in range(t_eval.shape[0]):
        X_mat = np.concatenate([t_eval[i]*np.ones((n,1)), X[:,1:]], axis=1)
        beta_mat[:,i] = LocalPolyReg(Y, X, x_eval=X_mat, degree=degree, deriv_ord=deriv_ord, 
                                     h=h, b=b, kernT=kernT, kernS=kernS)
    theta_C = np.sum(weight_mat * beta_mat, axis=0)
    return theta_C


def IntegEst(Y, X, t_eval=None, h_bar=None, kernT_bar="gaussian", 
                h=None, b=None, degree=2, deriv_ord=1, kernT="epanechnikov", kernS="epanechnikov"):
    '''
    Estimating the dose-response curve via our integral estimator with linear 
    interpolation approximation.
    
    
    Parameters
    ----------
        Y: (n,)-array
            The outcomes of n observations.
            
        X: (n,d+1)-array
            The first column of X is the treatment/exposure variable, while 
            the other d columns are confounding variables of n observations.
            
        t_eval: (m,)-array
            The coordinates of the m evaluation points. (Default: t_eval=None. Then, t_eval=X[:,0].)
            
        h_bar: float
            The bandwidth parameters for the Nadaraya-Watson conditional CDF estimator. 
            (Default: h_bar=None. Then, the Silverman's rule of thumb is applied. 
            See Chen et al.(2016) for details.)
            
        kernT_bar: str
            The name of the kernel function for Nadaraya-Watson conditional CDF estimator.
            (Default: "gaussian".)
            
        h,b: float
            The bandwidth parameters for the treatment/exposure variable and confounding variables. 
            (Default: h=None, b=None. Then, the rule-of-thumb bandwidth selector in Eq.(A1) of 
             Yang and Tschernig (1999) is used.)
            
        degree: int
            Degree of local polynomials. (Default: degree=2.)
            
        deriv_ord: int
            The order of the estimated derivative the conditional mean outcome function. 
            (Default: deriv_ord=1. Then, it estimates the partial derivative of the 
            conditional mean outcome function with respect to the treatment variable.)
            
        kernT, kernS: str
            The names of kernel functions for the treatment/exposure variable 
            and confounding variables. (Default: "epanechnikov".)
    '''
    
    if t_eval is None: 
        t_eval = X[:,0].copy()
    
    T_sort = np.sort(X[:,0])
    n = X.shape[0]  ## Number of data points
    
    # Compute theta_C at the order statistics of T
    theta_est = DerivEffect(Y, X, t_eval=T_sort, h_bar=h_bar, kernT_bar=kernT_bar, 
                            h=h, b=b, degree=degree, deriv_ord=deriv_ord, kernT=kernT, kernS=kernS)
    T_delta = T_sort[1:] - T_sort[:(n-1)]
    
    int_mat_up = np.ones((n,)) * (T_delta*(np.arange(1, n)*theta_est[:(n-1)])).reshape(-1,1)
    int_mat_up = int_mat_up * (np.arange(n-1).reshape(-1,1) < np.arange(n))
    
    int_mat_down = np.ones((n,)) * (T_delta*((n-np.arange(1, n))*theta_est[1:])).reshape(-1,1)
    int_mat_down = int_mat_down * (np.arange(n-1).reshape(-1,1) >= np.arange(n))
    m_samp = np.mean(Y) + np.sum(int_mat_up - int_mat_down, axis=0)/n
    
    m_est = np.interp(t_eval, T_sort, m_samp)
    return m_est


@ray.remote
def DerivEffect_Ray(Y, X, t_eval=None, h_bar=None, kernT_bar="gaussian", 
                h=None, b=None, degree=2, deriv_ord=1, kernT="epanechnikov", kernS="epanechnikov"):
    '''
    Estimating the derivative of the dose-response curve via Nadaraya-Watson conditional CDF estimator.
    
    
    Parameters
    ----------
        Y: (n,)-array
            The outcomes of n observations.
            
        X: (n,d+1)-array
            The first column of X is the treatment/exposure variable, while 
            the other d columns are confounding variables of n observations.
            
        t_eval: (m,)-array
            The coordinates of the m evaluation points. (Default: t_eval=None. Then, t_eval=X[:,0].)
            
        h_bar: float
            The bandwidth parameters for the Nadaraya-Watson conditional CDF estimator. 
            (Default: h_bar=None. Then, the Silverman's rule of thumb is applied. 
            See Chen et al.(2016) for details.)
            
        kernT_bar: str
            The name of the kernel function for Nadaraya-Watson conditional CDF estimator.
            (Default: "gaussian".)
            
        h,b: float
            The bandwidth parameters for the treatment/exposure variable and confounding variables. 
            (Default: h=None, b=None. Then, the rule-of-thumb bandwidth selector in Eq.(A1) of 
             Yang and Tschernig (1999) is used.)
            
        degree: int
            Degree of local polynomials. (Default: degree=2.)
            
        deriv_ord: int
            The order of the estimated derivative the conditional mean outcome function. 
            (Default: deriv_ord=1. Then, it estimates the partial derivative of the 
            conditional mean outcome function with respect to the treatment variable.)
            
        kernT, kernS: str
            The names of kernel functions for the treatment/exposure variable and confounding variables.
            (Default: "epanechnikov".)
    '''
    
    if t_eval is None: 
        t_eval = X[:,0].copy()
        
    n = X.shape[0]  ## Number of data points
    d = 1
    if h_bar is None:
        # Apply the Silverman's rule of thumb bandwidth in Chen et al.(2016).
        h_bar = (4/(d+2))**(1/(d+4))*(n**(-1/(d+4)))*np.std(X[:,0])
        print("The current bandwidth for the conditional CDF estimator is "+ str(h_bar) + ".\n")
    
    kernT_bar, sigmaK_sq, K_sq = rbf.KernelRetrieval(kernT_bar)
        
    weight_mat = kernT_bar((t_eval - X[:,0].reshape(-1,1)) / h_bar)
    weight_mat = weight_mat / np.sum(weight_mat, axis=0)
    weight_mat[np.isnan(weight_mat)] = 0
    beta_mat = np.zeros((n, t_eval.shape[0]))
    for i in range(t_eval.shape[0]):
        X_mat = np.concatenate([t_eval[i]*np.ones((n,1)), X[:,1:]], axis=1)
        beta_mat[:,i] = LocalPolyReg(Y, X, x_eval=X_mat, degree=degree, deriv_ord=deriv_ord, 
                                     h=h, b=b, kernT=kernT, kernS=kernS)
    theta_C = np.sum(weight_mat * beta_mat, axis=0)
    return theta_C


def IntegEst_Parallel(Y, X, t_eval=None, h_bar=None, kernT_bar="gaussian", 
                h=None, b=None, degree=2, deriv_ord=1, kernT="epanechnikov", kernS="epanechnikov"):
    '''
    Estimating the dose-response curve via simple integral estimator with linear interpolation approximation.
    
    
    Parameters
    ----------
        Y: (n,)-array
            The outcomes of n observations.
            
        X: (n,d+1)-array
            The first column of X is the treatment/exposure variable, while 
            the other d columns are confounding variables of n observations.
            
        t_eval: (m,)-array
            The coordinates of the m evaluation points. (Default: t_eval=None. Then, t_eval=X[:,0].)
            
        h_bar: float
            The bandwidth parameters for the Nadaraya-Watson conditional CDF estimator. 
            (Default: h_bar=None. Then, the Silverman's rule of thumb is applied. 
            See Chen et al.(2016) for details.)
            
        kernT_bar: str
            The name of the kernel function for Nadaraya-Watson conditional CDF estimator.
            (Default: "gaussian".)
            
        h,b: float
            The bandwidth parameters for the treatment/exposure variable and confounding variables. 
            (Default: h=None, b=None. Then, the rule-of-thumb bandwidth selector in Eq.(A1) of 
             Yang and Tschernig (1999) is used.)
            
        degree: int
            Degree of local polynomials. (Default: degree=2.)
            
        deriv_ord: int
            The order of the estimated derivative the conditional mean outcome function. 
            (Default: deriv_ord=1. Then, it estimates the partial derivative of the 
            conditional mean outcome function with respect to the treatment variable.)
            
        kernT, kernS: str
            The names of kernel functions for the treatment/exposure variable and confounding variables.
            (Default: "epanechnikov".)
    '''
    
    if t_eval is None: 
        t_eval = X[:,0].copy()
    
    T_sort = np.sort(X[:,0])
    n = X.shape[0]  ## Number of data points
    
    # Compute theta_C at the order statistics of T
    ray.init()
    chunksize = 10
    num_p = T_sort.shape[0]
    result_ids = []
    for i in range(0, num_p, chunksize):
        result_ids.append(DerivEffect_Ray.remote(Y, X, t_eval=T_sort[i:(i+chunksize)], 
                                             h_bar=h_bar, kernT_bar=kernT_bar, 
                            h=h, b=b, degree=degree, deriv_ord=deriv_ord, kernT=kernT, kernS=kernS))
    theta_est = ray.get(result_ids)
    theta_est = np.concatenate(theta_est, axis=0)
    ray.shutdown()
                          
    T_delta = T_sort[1:] - T_sort[:(n-1)]
    
    int_mat_up = np.ones((n,)) * (T_delta*(np.arange(1, n)*theta_est[:(n-1)])).reshape(-1,1)
    int_mat_up = int_mat_up * (np.arange(n-1).reshape(-1,1) < np.arange(n))
    
    int_mat_down = np.ones((n,)) * (T_delta*((n-np.arange(1, n))*theta_est[1:])).reshape(-1,1)
    int_mat_down = int_mat_down * (np.arange(n-1).reshape(-1,1) >= np.arange(n))
    m_samp = np.mean(Y) + np.sum(int_mat_up - int_mat_down, axis=0)/n
    
    m_est = np.interp(t_eval, T_sort, m_samp)
    return m_est


def LocalPolyReg1D(Y, X, h=None, x_eval=None, degree=3, deriv_ord=0, kernel="epanechnikov"):
    '''
    Local polynomial regression in one dimension.
    
    Parameters
    ----------
        Y: (m,)-array
            The y coordinates of m data points.
        
        X: (m,)-array
            The x coordinates of m data points.
            
        h: float
            The bandwidth parameter. (Default: h=None. Then, the rule-of-thumb 
            bandwidth selector in Eq.(A1) of Yang and Tschernig (1999) is used.)
            
        x_eval: (k,)-array
            Vector of evaluation points. (Default: x_eval=None. Then, x_eval=X.)
            
        degree: int
            Degree of local polynomials. (Default: degree=2.)
            
        deriv_ord: int
            The order of derivatives of the regression function that are estimated. 
            (Default: deriv_ord=0. Then, it is the usual local polynomial regression.)
    '''
    if x_eval is None: 
        x_eval = X.copy()
        
    n = X.shape[0]  ## Number of data points
        
    kernel, sigmaK_sq, K_sq = rbf.KernelRetrieval(kernel)
    if h is None:
        # Apply the rule-of-thumb bandwidth selector in Eq.(A1) of Yang and Tschernig (1999)
        p_coeff = np.polyfit(X, Y, 4)
        sec_deriv = 12*p_coeff[0]*X + 6*p_coeff[1]*X + 2*X
        C_fun = np.mean(sec_deriv**2)
        lhs = np.concatenate([np.ones((n,1)), X.reshape(-1,1)], axis=1)
        rcond = np.finfo(lhs.dtype).eps * max(*lhs.shape)
        beta = np.linalg.lstsq(lhs, Y, rcond=rcond)[0]
        # Compute the integrated residual sum of squares
        resid = np.sum((Y - np.dot(lhs, beta))**2) * (np.max(X) - np.min(X))/ (n-5)
        sigmaK_sq = sigmaK_sq**2
        # ROT
        h = ((K_sq*resid)/(4*n*sigmaK_sq*C_fun))**(1/5) * 7
        # h = ((K_sq*resid)/(4*n*sigmaK_sq*C_fun))**(1/5) * (n**(d/(5*(d+5)))) * 5
    print("The current bandwidth for the local polynomial regression is "+ str(h) + ".\n")
    
    Y_est = np.zeros_like(x_eval)
    for i in range(x_eval.shape[0]):
        weights = kernel((X-x_eval[i])/h)
        # Filter out the data points with zero weights to speed up regressions with kernels of local support.
        inds = np.where(np.abs(weights)>1e-26)[0]
        X_dat = np.zeros((n, degree+1))
        for p in range(degree + 1):
            X_dat[:,p] = (X - x_eval[i])**p
        S = np.sqrt(weights)[inds]
        lhs = np.dot(np.diag(S), X_dat[inds,:])
        rhs = S*Y[inds]
        rcond = np.finfo(lhs.dtype).eps * max(*lhs.shape)
        beta = np.linalg.lstsq(lhs, rhs, rcond=rcond)[0]
        Y_est[i] = np.math.factorial(deriv_ord)*beta[deriv_ord]
    return Y_est



def RegAdjust(Y, X, t_eval=None, h=None, b=None, degree=2, deriv_ord=0, 
              kernT="epanechnikov", kernS="epanechnikov"):
    '''
    Estimating the dose-response curve via simple integral estimator with linear interpolation approximation.
    
    
    Parameters
    ----------
        Y: (n,)-array
            The outcomes of n observations.
            
        X: (n,d+1)-array
            The first column of X is the treatment/exposure variable, while 
            the other d columns are confounding variables of n observations.
            
        t_eval: (m,)-array
            The coordinates of the m evaluation points. (Default: t_eval=None. Then, t_eval=X[:,0].)
            
        h,b: float
            The bandwidth parameters for the treatment/exposure variable and confounding variables. 
            (Default: h=None, b=None. Then, the rule-of-thumb bandwidth selector in Eq.(A1) of 
             Yang and Tschernig (1999) is used.)
            
        degree: int
            Degree of local polynomials. (Default: degree=2.)
            
        deriv_ord: int
            The order of the estimated derivative of the conditional mean outcome function. 
            (Default: deriv_ord=0. Then, it estimates the conditional mean outcome function itself.)
            
        kernT, kernS: str
            The names of kernel functions for the treatment/exposure variable and confounding variables.
            (Default: "epanechnikov".)
    '''
    
    if t_eval is None: 
        t_eval = X[:,0].copy()
    
    n = X.shape[0]  ## Number of data points
    beta_mat = np.zeros((n, t_eval.shape[0]))
    for i in range(t_eval.shape[0]):
        X_mat = np.concatenate([t_eval[i]*np.ones((n,1)), X[:,1:]], axis=1)
        beta_mat[:,i] = LocalPolyReg(Y, X, x_eval=X_mat, degree=degree, deriv_ord=deriv_ord, 
                                     h=h, b=b, kernT=kernT, kernS=kernS)
    m_est = np.mean(beta_mat, axis=0)
    return m_est


#=======================================================================================#


def KDE(x, data, h=None):
    '''
    d-dim Euclidean KDE with the Gaussian kernel
    
    Parameters:
        x: (m,d)-array
            The coordinates of m query points in the d-dim Euclidean space.
    
        data: (n,d)-array
            The coordinates of n random sample points in the d-dimensional 
            Euclidean space.
       
        h: float
            The bandwidth parameter. (Default: h=None. Then the Silverman's 
            rule of thumb is applied. See Chen et al.(2016) for details.)
    
    Return:
        f_hat: (m,)-array
            The corresponding kernel density estimates at m query points.
    '''
    n = data.shape[0]  ## Number of data points
    d = data.shape[1]  ## Dimension of the data
    
    if h is None:
        # Apply Silverman's rule of thumb to select the bandwidth parameter 
        # (Only works for Gaussian kernel)
        h = (4/(d+2))**(1/(d+4))*(n**(-1/(d+4)))*np.mean(np.std(data, axis=0))
    print("The current bandwidth for KDE is "+ str(h) + ".\n")
    
    f_hat = np.zeros((x.shape[0], ))
    for i in range(x.shape[0]):
        f_hat[i] = np.mean(np.exp(np.sum(-((x[i,:] - data)/h)**2, axis=1)/2))/ \
                   ((2*np.pi)**(d/2)*np.prod(h))
    return f_hat


def AsymVarSurrogate(Y, X, t_eval, h=None, kernT='epanechnikov', kernS="epanechnikov", h_den=None):
    '''
    Estimating the asymptotic variance of the integral estimator up to a multiplicative constant.
    
    
    Parameters
    ----------
        Y: (n,)-array
            The outcomes of n observations.
            
        X: (n,d+1)-array
            The first column of X is the treatment/exposure variable, while 
            the other d columns are confounding variables of n observations.
            
        t_eval: (m,)-array
            The coordinates of the m evaluation points. (Default: t_eval=None. Then, t_eval=X[:,0].)
            
        h: float
            The bandwidth parameter for the treatment variable in the local polynomial regression. 
            (Default: h=None. Then, the rule-of-thumb bandwidth selector in Eq.(A1) of 
             Yang and Tschernig (1999) is used.)
             
        kernT, kernS: str
            The names of kernel functions for the treatment/exposure variable 
            and confounding variables in the local polynomial regression. (Default: "epanechnikov".)
             
        h_den: float
            The bandwidth parameters for the KDE of the marginal density of T. 
            (Default: h_bar=None. Then, the Silverman's rule of thumb is applied. 
            See Chen et al.(2016) for details.)
    '''
    if t_eval is None: 
        t_eval = X[:,0].copy()
        
    if h is None:
        h, b_can = RoTBWLocalPoly(Y, X, kernT=kernT, kernS=kernS)
    print("The current bandwidth for treament variable in the local polynomial regression is "+ str(h) + ".\n")
    
    n = X.shape[0]  ## Number of data points
    T_sort = np.sort(X[:,0])
    T_delta = T_sort[1:] - T_sort[:(n-1)]
    
    kernT, sigmaK_sq, K_sq = rbf.KernelRetrieval(kernT)
    var_mat_samp = np.zeros((n,n))
    den_est = KDE(x=T_sort.reshape(-1,1), data=T_sort.reshape(-1,1), h=h_den)
    for i in range(n):
        Y_cur = Y[i]
        integ = Y_cur * ((Y_cur - T_sort)/h) * kernT((Y_cur - T_sort)/h) / den_est
        
        int_mat_up = np.ones((n,)) * (T_delta*(np.arange(1, n)*integ[:(n-1)])).reshape(-1,1)
        int_mat_up = int_mat_up * (np.arange(n-1).reshape(-1,1) < np.arange(n))
        
        int_mat_down = np.ones((n,)) * (T_delta*((n-np.arange(1, n))*integ[1:])).reshape(-1,1)
        int_mat_down = int_mat_down * (np.arange(n-1).reshape(-1,1) >= np.arange(n))
        
        var_mat_samp[i,:] = (np.sum(int_mat_up - int_mat_down, axis=0)/n)**2
    var_samp = np.mean(var_mat_samp, axis=0)
    # Use the linear interpolation
    var_est = np.interp(t_eval, T_sort, var_samp)
    return var_est