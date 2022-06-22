# -*- coding: utf-8 -*-
"""

@author: HY
"""

import numpy as np
from numpy.linalg import inv as inv

import matplotlib.pyplot as plt
import time

def ten2mat(tensor, mode):
    return np.reshape(np.moveaxis(tensor, mode, 0), (tensor.shape[mode], -1), order = 'F')

def mat2ten(mat, tensor_size, mode):
    index = list()
    index.append(mode)
    for i in range(tensor_size.shape[0]):
        if i != mode:
            index.append(i)
    return np.moveaxis(np.reshape(mat, list(tensor_size[index]), order = 'F'), 0, mode)

def mySVD(mat):
    ## faster SVD
    [m, n] = mat.shape
    if 2 * m < n:
        u, s, v = np.linalg.svd(mat @ mat.T, full_matrices = 0)
        s = np.sqrt(s)
        tol = n * np.finfo(float).eps * np.max(s)
        idx = np.sum(s > tol)
        return u[:,:idx] , s[:idx],  np.diag(1/s[:idx]) @ u[:,:idx].T @ mat
    elif m > 2 * n:
        v,s,u = my_SVD(mat.T, τ)
        return u, s ,v
    u, s, v = np.linalg.svd(mat, full_matrices = 0)
    return u,s,v
    
def svt_NC(mat, L_l, k, α, ε, ρ):
    τ = α[k]/ρ
    _,σ,_ = mySVD(ten2mat(L_l,k))    
    ωk =  1/σ +ε[k]
    u,s,v =  mySVD(mat)                
    ss = s- τ * ωk
    idx = np.sum(ss>0 )
    vec = ss[:idx].copy()

    return u[:, :idx] @ np.diag(vec) @ v[:idx, :]
    
def compute_rmse(var, var_hat):
    return np.sqrt(np.sum((var - var_hat) ** 2) / var.shape[0])
def compute_mape(var, var_hat):
    return np.sum(np.abs(var - var_hat) / var) / var.shape[0]

def plot(rmse,mape,dataset,missing_rate,mode,duration):

    fig = plt.figure(figsize=(6, 5))
    plt.rc('font',family='Times New Roman')
    ax1= fig.add_subplot(111)
    ax1.plot(mape,label='MAPE')
    
    ax2=ax1.twinx()
    ax2.plot(rmse,color='orange',label='RMSE')
    plt.title('%s - %s - %.1f ' %(mode,dataset,missing_rate),size=15)
    ax1.text( 0.1,0.4, ' MAPE(min): %.3f  loc: %d \n RMSE(min): %.3f  loc: %d \n \n MAPE(last): %.3f \n RMSE(last): %.3f  \n Duration: %d seconds' 
             %(mape.min()*100, mape.argmin(), rmse.min(), rmse.argmin(), mape[-1]*100, rmse[-1],duration), transform=ax1.transAxes, size=15 )
    ax1.legend(loc='upper left',fontsize=14)
    ax2.legend(loc='upper right',fontsize=14)


def TC_PFNC(dense_tensor, sparse_tensor, ρ, ε, α, tol, K, pos_missing, pos_test ):
    ''' dense_tensor:  ground truth tensor
        sparse_tensor: observed tensor
        ε : a small constant
        α : weight parameters of each mode
        tol : tolerance
        K : max iteration
        '''
    ## tensor shape
    dim = np.array(sparse_tensor.shape)
    ## Initialization：
    ## observed tensor 
    M = sparse_tensor.copy()                         # shape: n1*n2*n3
    ## low-rank tensor
    L = sparse_tensor.copy()                         # shape: n1*n2*n3  
    L3 = np.zeros(np.insert(dim, 0, len(dim)))       # shape: 3*n1*n2*n3
    ## multiplier tensor
    T3 = np.zeros(np.insert(dim, 0, len(dim)))       # shape: 3*n1*n2*n3
    ## RMSE and MAPE
    RMSE = np.zeros(K)
    MAPE = np.zeros(K)
    
    it = 0
    while True:

        ## update Lk
        L_l = L.copy()
        for k in range(3):
            L3[k] = mat2ten(svt_NC(ten2mat((M - T3[k]/ρ), k), L_l, k, α, ε, ρ ), dim, k)
        
        ## update M
        M[pos_missing] = np.mean(L3 + T3/ρ, axis=0)[pos_missing]
        
        ## update T
        T3 = T3 + ρ*( L3 - np.broadcast_to(M, np.insert(dim, 0, len(dim))))
        
        ## Update L
        L = np.einsum('k, kmnt -> mnt', α, L3)
        
        ## compute the tolerance
        tole = np.sqrt(np.sum((L - L_l) ** 2)) / np.sqrt(np.sum((L_l) ** 2))
        
        ## compute the MAPE, RMSE 
        mape = compute_mape(dense_tensor[pos_test], L[pos_test])
        rmse = compute_rmse(dense_tensor[pos_test], L[pos_test])
        MAPE[it] = mape
        RMSE[it] = rmse
    
        if (tole < tol) or (it+1 >= K):
            break
    
        if (it + 1) % 1000 == 0:
            print('Iter: {}'.format(it + 1))
            print('MAPE: {:.6}'.format(mape))
            print('RMSE: {:.6}'.format(rmse))
            print()
            
        ## update iteration
        it +=1

    print('Total iteration: {}'.format(it+1))
    print('Tolerance: {:.6}'.format(tole))
    print('Imputation MAPE: {:.6}'.format(mape))
    print('Imputation RMSE: {:.6}'.format(rmse))
    print()

    return L,L3,M,MAPE,RMSE
