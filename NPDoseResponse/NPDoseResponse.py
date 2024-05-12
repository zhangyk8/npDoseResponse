# -*- coding: utf-8 -*-

# Author: Yikun Zhang
# Last Editing: May 12, 2024

# Description: This script contains the implementation of local polynomial regression 
# as well as our proposed integral estimator and its localized derivative estimator.

import numpy as np
from .rbf import KernelRetrieval
from multiprocessing import Pool
from functools import partial
from .utils import *

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
    kernT, sigmaK_sq, K_sq = KernelRetrieval(kernT)
    kernS, sigmaK_sq, K_sq = KernelRetrieval(kernS)
    
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


def LocalPolyReg_Fs(x_eval, Y, X, degree=2, deriv_ord=1, h=None, b=None, 
                 kernT="epanechnikov", kernS="epanechnikov", 
                 h_lst=np.linspace(0.5, 15, 30), b_lst=np.linspace(0.2, 6, 30)):
    '''
    (Partial) Local polynomial regression for estimating the conditional mean outcome 
    function and its partial derivatives. We use higher order local monomials for 
    the treatment variable and first-order local monomials for the confounding variables.
    (This function is for multi-process execution only.)
    
    
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


def DerivEffect(Y, X, t_eval=None, h_bar=None, kernT_bar="gaussian", h=None, b=None, 
                degree=2, deriv_ord=1, kernT="epanechnikov", kernS="epanechnikov", 
                parallel=False, processes=20):
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
            
        parallel: boolean
            The indicator of whether the function should be parallel executed by
            multi-processing. (Default: parallel=False.)
            
        processes: int
            The number of processes for parallel execution. (Default: processes=20.)
    '''
    
    if t_eval is None: 
        t_eval = X[:,0].copy()
        
    n = X.shape[0]  ## Number of data points
    d = 1
    if h_bar is None:
        # Apply the Silverman's rule of thumb bandwidth in Chen et al.(2016).
        h_bar = (4/(d+2))**(1/(d+4))*(n**(-1/(d+4)))*np.std(X[:,0])
        print("The current bandwidth for the conditional CDF estimator is "+ str(h_bar) + ".\n")
    
    kernT_bar, sigmaK_sq, K_sq = KernelRetrieval(kernT_bar)
        
    weight_mat = kernT_bar((t_eval - X[:,0].reshape(-1,1)) / h_bar)
    weight_mat = weight_mat / np.sum(weight_mat, axis=0)
    weight_mat[np.isnan(weight_mat)] = 0
    if parallel:
        with Pool(processes=processes) as pool:
            part_fun = partial(LocalPolyReg_Fs, Y=Y, X=X, degree=degree, deriv_ord=deriv_ord, 
                               h=h, b=b, kernT=kernT, kernS=kernS)
            beta_mat = pool.map(part_fun, [np.concatenate([t_eval[i]*np.ones((n,1)), X[:,1:]], axis=1) for i in range(t_eval.shape[0])])
            beta_mat = np.concatenate(beta_mat, axis=0).reshape(t_eval.shape[0], n).T
    else:
        beta_mat = np.zeros((n, t_eval.shape[0]))
        for i in range(t_eval.shape[0]):
            X_mat = np.concatenate([t_eval[i]*np.ones((n,1)), X[:,1:]], axis=1)
            beta_mat[:,i] = LocalPolyReg(Y, X, x_eval=X_mat, degree=degree, deriv_ord=deriv_ord, 
                                         h=h, b=b, kernT=kernT, kernS=kernS)
    
    theta_C = np.sum(weight_mat * beta_mat, axis=0)
    return theta_C


def IntegEst(Y, X, t_eval=None, h_bar=None, kernT_bar="gaussian", h=None, b=None, 
             degree=2, deriv_ord=1, kernT="epanechnikov", kernS="epanechnikov",
             parallel=False, processes=20):
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
        
        parallel: boolean
            The indicator of whether the function should be parallel executed by
            multi-processing. (Default: parallel=False.)
            
        processes: int
            The number of processes for parallel execution. (Default: processes=20.)
    '''
    
    if t_eval is None: 
        t_eval = X[:,0].copy()
    
    T_sort = np.sort(X[:,0])
    n = X.shape[0]  ## Number of data points
    
    # Compute theta_C at the order statistics of T
    theta_est = DerivEffect(Y, X, t_eval=T_sort, h_bar=h_bar, kernT_bar=kernT_bar, 
                            h=h, b=b, degree=degree, deriv_ord=deriv_ord, kernT=kernT, kernS=kernS,
                            parallel=parallel, processes=processes)
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
        
    kernel, sigmaK_sq, K_sq = KernelRetrieval(kernel)
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

