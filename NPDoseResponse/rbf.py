#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Yikun Zhang
Last Editing: March 10, 2024

This file contains the implementation of all the common kernel functions.
"""

import numpy as np

def rectangular(t):
    ind = (np.abs(t)<=1)
    res = np.abs(0.5*ind)
    return res

def triangular(t):
    ind = (np.abs(t)<=1)
    res = np.abs((1-np.abs(t))*ind)
    return res

def epanechnikov(t):
    ind = (np.abs(t)<=1)
    res = np.abs(0.75*(1-t**2)*ind)
    return res

def biweight(t):
    ind = (np.abs(t)<=1)
    res = np.abs(((15/16)*(1-t**2)**2)*ind)
    return res

def triweight(t):
    ind = (np.abs(t)<=1)
    res = np.abs((35/32)*(1-t**2)**3*ind)
    return res

def tricube(t):
    ind = (np.abs(t)<=1)
    res = np.abs((70/81)*(1-np.abs(t)**3)**3*ind)
    return res

def gaussian(t):
    res = (1/np.sqrt(2*np.pi))*np.exp(-0.5*t**2)
    return res

def bigaussian(t):
    res = (2/np.sqrt(np.pi))*(t**2)*np.exp(-t**2)
    return res

def cosine(t):
    ind = (np.abs(t)<=1)
    res = np.abs((np.pi/4)*np.cos(np.pi*t/2)*ind)
    return res

def logistic(t):
    res = 1/(np.exp(t)+2+np.exp(-t))
    return res

def sigmoid(t):
    res = (2/np.pi)/(np.exp(t)+np.exp(-t))
    return res

def silverman(t):
    res = 0.5*np.exp(-np.abs(t)/np.sqrt(2))*np.sin(np.abs(t)/np.sqrt(2)+np.pi/4)
    return res


def KernelRetrieval(name):
    '''
    Retrieving the kernel function, its second moment, and its variance based on 
    the name.
    
    Parameters
    ----------
        name: str
            The name of the kernel function.

    Return
    --------
        kern_func: python function
            The final estimated weights by our debiasing program.
            
        sigmaK_sq: float
            The second moment of the kernel function.
            
        K_sq: float
            The variance of the kernel function.
    '''
    if name == "rectangular":
        return rectangular, 1/3, 1/2
    
    if name == "triangular":
        return triangular, 1/6, 2/3
    
    if name == "epanechnikov":
        return epanechnikov, 1/5, 3/5
    
    if name == "biweight":
        return biweight, 1/7, 5/7
    
    if name == "triweight":
        return triweight, 1/9, 350/429
    
    if name == "tricube":
        return triangular, 35/243, 175/247
    
    if name == "gaussian":
        return gaussian, 1, 1/(2*np.sqrt(np.pi))
    
    if name == "bigaussian":
        return bigaussian, 3/2, 3*np.sqrt(2/np.pi)/8
    
    if name == "cosine":
        return cosine, 1-8/(np.pi**2), np.pi**2/16
    
    if name == "logistic":
        return logistic, np.pi**2/3, 1/6
    
    if name == "sigmoid":
        return sigmoid, np.pi**2/4, 2/(np.pi**2)
    
    if name == "silverman":
        return silverman, 0, 3*np.sqrt(2)/16