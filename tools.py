# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 15:17:33 2021

@author: aoust
"""

import numpy as np
from scipy.sparse import diags

def argmin_cumsum(tuples,target):
    somme = 0
    for aux, tup in enumerate(tuples):
        value, idx, string = tup
        somme+=value
        if somme>=target:
            return aux
    return aux

def gershgorin_bounds(M):
    D = M.diagonal()
    M2 = M - diags(D)
    assert(np.linalg.norm(M2.diagonal())<=1E-6)
    M2 = np.abs(M2)
    radius = M2.sum(axis=0)
    LB,UB = (D-radius).min(), (D+radius).max()
    return np.real(LB),np.real(UB)

def proj_simplex(a, y):
    assert(a>0)
    N = len(y)
    u = (y).copy()
    u = -np.sort(-u)
    aux = (np.cumsum(u) - a)/np.array([i for i in range(1,N+1)])
    for i in range(1,N+1):
        if aux[i-1]<u[i-1]:
            tau = aux[i-1]
    return np.maximum(y - tau,0)
    
    