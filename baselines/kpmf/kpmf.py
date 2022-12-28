# This is the re-implementation of Tinghui Zhou's Kernelized Probabilistic Matrix Factorization using Python.
# Details can be found in his paper [Kernelized Probabilistic Matrix Factorization: Exploiting Graphs and Side Information]
# Author: Dingyi Zhuang [zdysdsd@gmail.com]

#%%
import sys
import numpy as np 
import pandas as pd
import time
import scipy.io
from numpy.matlib import repmat
#%%
print(sys.version)

#%%
def graphKernel(G,gamma):
    # Compute the covariance matrix based on the regularized graph laplacian kernel
    ## Value:
        # N: number of nodes in the graph
    ## input:
        # G: N*N, undirected graph adjacency matrix, in np.mat format
        # beta: scaler, parameter used in graph kernel
    ## Output:
        # K: N*N, covariance matrix
    N=G.shape[0]
    K=np.zeros((N,N))
    # The degree matrix for laplacian kernel
    Deg=np.diag(np.sum(G,0))
    s=np.sum(G,0)
    Deg=np.diag(s)
    del s
    # The graph laplacian
    L=Deg-G
    K=np.linalg.inv(np.eye(N,N)+gamma*L)
    return K

#%%
def kpmf_sgd(R, mask, D, K_u_inv, K_v_inv, sigma_r, eta, R2, mask2):
    # We only implement the stochastic gradient descent process for learning
    ## Value:
        # N: number of rows in R
        # M: number of columns in R
        # D: latent dimensions
    ## Input:
        # R: trainning matrix
        # mask: N*M indicator matrix where 1 determines a valid data entry
        # D: latent dimensions
        # K_u_inv: precision matrix on rows
        # K_v_inv: precision matrix on columns
        # sigma_r: scaler, variance for the univariate Gaussian process to generate each entry for R
        # eta: learning rate
        # R2: validation matrix
        # mask2: indicator matrix for the validation matrix
    ## Output:
        # U: latent matrix on rows
        # V: latent matrix on columns
        # RMSE: scaler, RMSE on validation set
        # time: runtime
    start_time=time.clock()
    N=R.shape[0]
    M=R.shape[1]

    sumEachRow=np.sum(mask,1)
    sumEachCol=np.sum(mask,0)

    # Initialize the latent matrix U and V
    Rsvd=np.multiply(R,mask)+np.sum(np.multiply(R,mask))/np.sum(mask)*(1-mask)
    U,S,V = np.linalg.svd(Rsvd, full_matrices=False)
    S=np.diag(S)
    U=U*np.sqrt(S)
    V=np.dot(V.transpose(),np.sqrt(S))

    rs=np.sqrt(np.sum(np.multiply((R2-np.dot(V,U).transpose())**2,mask2))/np.sum(mask2))

    # Learning
    maxepoch=100
    minepoch=5
    epsilon=0.00001

    val=np.multiply(R,mask)
    rowind=val.nonzero()[0]
    colind=val.nonzero()[1]
    value=val[rowind,colind]
    rowind=rowind.reshape(-1,1)
    colind=colind.reshape(-1,1)
    value=value.reshape(-1,1)
    del val
    train_vec=np.concatenate((rowind,colind,value),axis=1)
    length=len(value)

    batch_num=12
    batch_size=int(np.round(length/batch_num))
    monmentum=0.2

    U_inc=np.zeros(U.shape)
    V_inc=np.zeros(V.shape)

    for epoch in range(maxepoch):
        train_vec=train_vec[np.random.permutation(length),:]
        for batch in range(batch_num):
            if batch<batch_num-1:
                brow=np.double(train_vec[(batch)*batch_size+1:(batch+1)*batch_size,0])
                bcol=np.double(train_vec[(batch)*batch_size+1:(batch+1)*batch_size,1])
                bval=np.double(train_vec[(batch)*batch_size+1:(batch+1)*batch_size,2])
            elif batch==batch_num-1:
                brow=np.double(train_vec[batch*batch_size+1:length,0])
                bcol=np.double(train_vec[batch*batch_size+1:length,1])
                bval=np.double(train_vec[batch*batch_size+1:length,2])
            else:
                pass

            brow=brow.astype(np.int)
            bcol=bcol.astype(np.int)
            bval=bval.astype(np.int)
            pred=np.sum(np.multiply(U[brow,:],V[bcol,:]),1)
            K_u_invU=(U.transpose()*K_u_inv)
            diag_K_u_inv_U=np.multiply(np.diag(K_u_inv),U)
            K_v_invV=np.dot(V.transpose(),K_v_inv)
            diag_K_v_inv_V=np.multiply(np.diag(K_v_inv),V)

            gd_u=-2/(sigma_r**2)*(np.multiply
            (repmat((bval-pred),1,D),V[bcol,:]))+K_u_invU[brow,:]/repmat(sumEachRow[brow],1,D)+diag_K_u_inv_U[brow,:]/repmat(sumEachRow[brow],1,D)

            gd_v=-2/(sigma_r**2)*(np.multiply(repmat((bval-pred),1,D),U[bcol,:]))+K_v_invV[bcol,:]/repmat(sumEachCol[bcol],1,D)+diag_K_v_inv_V[bcol,:]/repmat(sumEachCol[bcol],1,D)

            dU=np.zeros(N,D)
            dV=np.zeros(M,D)

            for b in range(len(brow)):
                r=brow[b]
                c=bcol[b]
                dU[r,:]=dU[r,:]+gd_u[b,:]
                dV[c,:]=dV[c,:]+gd_v[b,:]
            
            U_inc=U_inc*monmentum+dU*eta
            V_inc=V_inc*monmentum+dV*eta

            U=U-U_inc
            V=V-V_inc
        RMSE=np.zeros((1,maxepoch))
        RMSE[epoch]=np.sqrt(np.sum((R2-U*V.transpose())**2*mask2)/np.sum(mask2))
        print('epoch{:}, validation rmse={:}'.format(epoch,RMSE[epoch]))
        if (epoch>minepoch) and ((RMSE[epoch]-RMSE[epoch-1])/RMSE[epoch] <epsilon ):
            break

if __name__ == '__main__':
    # Parameters
    sigma_r = 0.4;  # Variance of entries
    D=10;           # Latent dimension
    eta = 0.003;    # Learning rate
    gamma=0.1;      # Parameter for graph kernel

    # Load data
    mat=scipy.io.loadmat('data.mat')
    Graph=mat['Graph']
    R=mat['R']
    testSet=mat['testSet']
    trainSet=mat['trainSet']
    valSet=mat['valSet']

    # Positive entries are valid entries, construct mask matricies
    mask_train=np.zeros(trainSet.shape)
    mask_train[np.nonzero(trainSet)]=1
    mask_test=np.zeros(testSet.shape)
    mask_test[np.nonzero(testSet)]=1
    mask_val=np.zeros(valSet.shape)
    mask_val[np.nonzero(valSet)]=1

    # Scale entries to be in [0,1]
    trainSet = trainSet/5.0
    testSet = testSet/5.0
    valSet = valSet/5.0

    N,M=trainSet.shape
    # Get covariance matrix for columns/movies (assuming diagonal)
    K_v = 0.2 * np.eye(M)
    K_v_inv = np.linalg.inv(K_v)

    # Get covariance matrix for rows/users based on the graph kernel (using side information) 
    # The value Graph of social network of users is the side information
    K_u = graphKernel(Graph, gamma); 
    K_u_inv = np.linalg.pinv(K_u)

    # Training
    U, V, RMSE,time = kpmf_sgd(trainSet, mask_train, D, K_u_inv, K_v_inv, sigma_r, eta, valSet, mask_val)

    rmse=np.sqrt( np.sum(((U*V.transpose()-testSet)*mask_test )**2)/np.sum(mask_test)   ) 
    print('The rmse result is: ', rmse)


