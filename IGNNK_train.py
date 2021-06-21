from __future__ import division
​
import torch
import numpy as np
import torch.optim as optim
from torch import nn
import matplotlib.pyplot as plt
from utils import *
import random
import pandas as pd
from basic_structure import IGNNK
import argparse
import sys
import os
import time
​
def parse_args(args):
    '''Parse training options user can specify in command line.
    Specify hyper parameters here
​
    Returns
    -------
    argparse.Namespace
        the output parser object
    '''
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Parse argument used when training IGNNK model.",
        epilog="python IGNNK_train.py DATASET, for example: python IGNNK_train.py 'metr' ")
​
    # Requird input parametrs
    parser.add_argument(
        'dataset',type=str,default='metr',
        help = 'Name of the datasets, select from metr, nrel, ushcn, sedata or pems'
    )
    
    # optional input parameters
    parser.add_argument(
        '--n_o',type=int,default=150,
        help='sampled space dimension'
    )
    parser.add_argument(
        '--h',type=int,default=24,
        help='sampled time dimension'
    )
    parser.add_argument(
        '--z',type=int,default=100,
        help='hidden dimension for graph convolution'
    )
    parser.add_argument(
        '--K',type=int,default=1,
        help='If using diffusion convolution, the actual diffusion convolution step is K+1'
    )
    parser.add_argument(
        '--n_m',type=int,default=50,
        help='number of mask node during training'
    )
    parser.add_argument(
        '--n_u',type=int,default=50,
        help='target locations, n_u locations will be deleted from the training data'
    )
    parser.add_argument(
        '--max_iter',type=int,default=750,
        help='max training episode'
    )
    parser.add_argument(
        '--learning_rate',type=float,default=0.0001,
        help='the learning_rate for Adam optimizer'
    )
    parser.add_argument(
        '--E_maxvalue',type=int,default=80,
        help='the max value from experience'
    )
    parser.add_argument(
        '--batch_size',type=int,default=4,
        help='Batch size'
    )                 
    parser.add_argument(
        '--to_plot',type=bool,default=True,
        help='Whether to plot the RMSE training result'
    )           
    return parser.parse_known_args(args)[0]
​
def load_data(dataset):
    '''Load dataset
    Input: dataset name
    Returns
    -------
    A: adjacency matrix 
    X: processed data
    capacity: only works for NREL, each station's capacity
    '''
    capacity = []
    if dataset == 'metr':
        A, X = load_metr_la_rdata()
        X = X[:,0,:]
    elif dataset == 'nrel':
        A, X , files_info = load_nerl_data()
        #For Nrel, We only use 7:00am to 7:00pm as the target data, because otherwise the 0-values of periods without sunshine will greatly influence the results
        time_used_base = np.arange(84,228)
        time_used = np.array([])
        for i in range(365):
            time_used = np.concatenate((time_used,time_used_base + 24*12* i))
        X=X[:,time_used.astype(np.int)]
        capacities = np.array(files_info['capacity'])
        capacities = capacities.astype('float32')
    elif dataset == 'ushcn':
        A,X,Omissing = load_udata()
        X = X[:,:,:,0]
        X = X.reshape(1218,120*12)
        X = X/100
    elif dataset == 'sedata':
        A, X = load_sedata()
        A = A.astype('float32')
        X = X.astype('float32')
    elif dataset == 'pems':
        A,X = load_pems_data()
    else:
        raise NotImplementedError('Please specify datasets from: metr, nrel, ushcn, sedata or pems')
    split_line1 = int(X.shape[1] * 0.7)
    training_set = X[:,:split_line1].transpose()
    test_set = X[:, split_line1:].transpose()       # split the training and test period
    rand = np.random.RandomState(0) # Fixed random output
    unknow_set = rand.choice(list(range(0,X.shape[0])),n_u,replace=False)
    unknow_set = set(unknow_set)
    full_set = set(range(0,X.shape[0]))        
    know_set = full_set - unknow_set
    training_set_s = training_set[:, list(know_set)]   # get the training data in the sample time period
    A_s = A[:, list(know_set)][list(know_set), :]      # get the observed adjacent matrix from the full adjacent matrix,
                                                    # the adjacent matrix are based on pairwise distance, 
                                                    # so we need not to construct it for each batch, we just use index to find the dynamic adjacent matrix  
    return A,X,training_set,test_set,unknow_set,full_set,know_set,training_set_s,A_s,capacity     
​
"""
Define the test error
"""
def test_error(STmodel, unknow_set, test_data, A_s, Missing0):
    """
    :param STmodel: The graph neural networks
    :unknow_set: The unknow locations for spatial prediction
    :test_data: The true value test_data of shape (test_num_timesteps, num_nodes)
    :A_s: The full adjacent matrix
    :Missing0: True: 0 in original datasets means missing data
    :return: NAE, MAPE and RMSE
    """  
    unknow_set = set(unknow_set)
    time_dim = STmodel.time_dimension
    
    test_omask = np.ones(test_data.shape)
    if Missing0 == True:
        test_omask[test_data == 0] = 0
    test_inputs = (test_data * test_omask).astype('float32')
    test_inputs_s = test_inputs
   
    missing_index = np.ones(np.shape(test_data))
    missing_index[:, list(unknow_set)] = 0
    missing_index_s = missing_index
    
    o = np.zeros([test_data.shape[0]//time_dim*time_dim, test_inputs_s.shape[1]]) #Separate the test data into several h period
    
    for i in range(0, test_data.shape[0]//time_dim*time_dim, time_dim):
        inputs = test_inputs_s[i:i+time_dim, :]
        missing_inputs = missing_index_s[i:i+time_dim, :]
        T_inputs = inputs*missing_inputs
        T_inputs = T_inputs/E_maxvalue
        T_inputs = np.expand_dims(T_inputs, axis = 0)
        T_inputs = torch.from_numpy(T_inputs.astype('float32'))
        A_q = torch.from_numpy((calculate_random_walk_matrix(A_s).T).astype('float32'))
        A_h = torch.from_numpy((calculate_random_walk_matrix(A_s.T).T).astype('float32'))
        
        imputation = STmodel(T_inputs, A_q, A_h)
        imputation = imputation.data.numpy()
        o[i:i+time_dim, :] = imputation[0, :, :]
    
    if dataset == 'NREL':  
        o = o*capacities[None,:]
    else:
        o = o*E_maxvalue
    truth = test_inputs_s[0:test_set.shape[0]//time_dim*time_dim]
    o[missing_index_s[0:test_set.shape[0]//time_dim*time_dim] == 1] = truth[missing_index_s[0:test_set.shape[0]//time_dim*time_dim] == 1]
    
    test_mask =  1 - missing_index_s[0:test_set.shape[0]//time_dim*time_dim]
    if Missing0 == True:
        test_mask[truth == 0] = 0
        o[truth == 0] = 0
    
    MAE = np.sum(np.abs(o - truth))/np.sum( test_mask)
    RMSE = np.sqrt(np.sum((o - truth)*(o - truth))/np.sum( test_mask) )
    MAPE = np.sum(np.abs(o - truth)/(truth + 1e-5))/np.sum( test_mask)
    
    return MAE, RMSE, MAPE
​
​
def rolling_test_error(STmodel, unknow_set, test_data, A_s, Missing0):
    """
    :It only calculates the last time points' prediction error, and updates inputs each time point
    :param STmodel: The graph neural networks
    :unknow_set: The unknow locations for spatial prediction
    :test_data: The true value test_data of shape (test_num_timesteps, num_nodes)
    :A_s: The full adjacent matrix
    :Missing0: True: 0 in original datasets means missing data
    :return: NAE, MAPE and RMSE
    """  
    
    unknow_set = set(unknow_set)
    time_dim = STmodel.time_dimension
    
    test_omask = np.ones(test_data.shape)
    if Missing0 == True:
        test_omask[test_data == 0] = 0
    test_inputs = (test_data * test_omask).astype('float32')
    test_inputs_s = test_inputs
   
    missing_index = np.ones(np.shape(test_data))
    missing_index[:, list(unknow_set)] = 0
    missing_index_s = missing_index
​
    o = np.zeros([test_set.shape[0] - time_dim, test_inputs_s.shape[1]])
    
    for i in range(0, test_set.shape[0] - time_dim):
        inputs = test_inputs_s[i:i+time_dim, :]
        missing_inputs = missing_index_s[i:i+time_dim, :]
        MF_inputs = inputs * missing_inputs
        MF_inputs = np.expand_dims(MF_inputs, axis = 0)
        MF_inputs = torch.from_numpy(MF_inputs.astype('float32'))
        A_q = torch.from_numpy((calculate_random_walk_matrix(A_s).T).astype('float32'))
        A_h = torch.from_numpy((calculate_random_walk_matrix(A_s.T).T).astype('float32'))
        
        imputation = STmodel(MF_inputs, A_q, A_h)
        imputation = imputation.data.numpy()
        o[i, :] = imputation[0, time_dim-1, :]
    
 
    truth = test_inputs_s[time_dim:test_set.shape[0]]
    o[missing_index_s[time_dim:test_set.shape[0]] == 1] = truth[missing_index_s[time_dim:test_set.shape[0]] == 1]
    
    if dataset == 'NREL':  
        o = o*capacities[None,:]
    else:
        o = o*E_maxvalue
    truth = test_inputs_s[0:test_set.shape[0]//time_dim*time_dim]
    test_mask =  1 - missing_index_s[time_dim:test_set.shape[0]]
    if Missing0 == True:
        test_mask[truth == 0] = 0
        o[truth == 0] = 0
        
    MAE = np.sum(np.abs(o - truth))/np.sum( test_mask)
    RMSE = np.sqrt(np.sum((o - truth)*(o - truth))/np.sum( test_mask) )
    MAPE = np.sum(np.abs(o - truth)/(truth + 1e-5))/np.sum( test_mask)  #avoid x/0
        
    return MAE, RMSE, MAPE  
​
def plot_res(RMSE_list,dataset,time_batch):
    """
    Draw Learning curves on testing error
    """    
    fig,ax = plt.subplots()
    ax.plot(RMSE_list,label='RMSE_on_test_set',linewidth=3.5)
    ax.set_xlabel('Training Batch (x{:})'.format(time_batch),fontsize=20)
    ax.set_ylabel('RMSE',fontsize=20)
    ax.tick_params(axis="x", labelsize=14)
    ax.tick_params(axis="y", labelsize=14)
    ax.legend(fontsize=16)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('fig/learning_curve_{:}.pdf'.format(dataset))
​
if __name__ == "__main__":
    """
    Model training
    """
    flags = parse_args(sys.argv[1:])
    dataset=flags.dataset
    n_o_n_m = flags.n_o
    h = flags.h
    z = flags.z
    K = flags.K
    n_m = flags.n_m
    n_u = flags.n_u
    max_iter = flags.max_iter
    learning_rate = flags.learning_rate
    E_maxvalue = flags.E_maxvalue
    batch_size = flags.batch_size
    to_plot = flags.to_plot
    # load dataset
    A,X,training_set,test_set,unknow_set,full_set,know_set,training_set_s,A_s,capacity = load_data(dataset)
    # Define model
    STmodel = IGNNK(h, z, K)  # The graph neural networks
​
    criterion = nn.MSELoss()
    optimizer = optim.Adam(STmodel.parameters(), lr=learning_rate)
    RMSE_list = []
    MAE_list = []
    MAPE_list = []
    for epoch in range(max_iter):
        for i in range(training_set.shape[0]//(h * batch_size)):  #using time_length as reference to record test_error
            t_random = np.random.randint(0, high=(training_set_s.shape[0] - h), size=batch_size, dtype='l')
            know_mask = set(random.sample(range(0,training_set_s.shape[1]),n_o_n_m)) #sample n_o + n_m nodes
            feed_batch = []
            for j in range(batch_size):
                feed_batch.append(training_set_s[t_random[j]: t_random[j] + h, :][:, list(know_mask)]) #generate 8 time batches
            
            inputs = np.array(feed_batch)
            inputs_omask = np.ones(np.shape(inputs))
            if not dataset == 'NREL': 
                inputs_omask[inputs == 0] = 0           # We found that there are irregular 0 values for METR-LA, so we treat those 0 values as missing data,
                                                        # For other datasets, it is not necessary to mask 0 values
                                                    
            missing_index = np.ones((inputs.shape))
            for j in range(batch_size):
                missing_mask = random.sample(range(0,n_o_n_m),n_m) #Masked locations
                missing_index[j, :, missing_mask] = 0
            if dataset == 'NREL':
                Mf_inputs = inputs * inputs_omask * missing_index / capacities[:, None]
            else:
                Mf_inputs = inputs * inputs_omask * missing_index / E_maxvalue #normalize the value according to experience
            Mf_inputs = torch.from_numpy(Mf_inputs.astype('float32'))
            mask = torch.from_numpy(inputs_omask.astype('float32'))   #The reconstruction errors on irregular 0s are not used for training
            
            A_dynamic = A_s[list(know_mask), :][:, list(know_mask)]   #Obtain the dynamic adjacent matrix
            A_q = torch.from_numpy((calculate_random_walk_matrix(A_dynamic).T).astype('float32'))
            A_h = torch.from_numpy((calculate_random_walk_matrix(A_dynamic.T).T).astype('float32'))
            
            if dataset == 'NREL':
                outputs = torch.from_numpy(inputs/capacities[:, None])
            else:
                outputs = torch.from_numpy(inputs/E_maxvalue) #The label
            
            optimizer.zero_grad()
            X_res = STmodel(Mf_inputs, A_q, A_h)  #Obtain the reconstruction
            
            loss = criterion(X_res*mask, outputs*mask)
            loss.backward()
            optimizer.step()        #Errors backward
        if not dataset == 'NREL':
            MAE_t, RMSE_t, MAPE_t = test_error(STmodel, unknow_set, test_set, A, True)
        else:
            MAE_t, RMSE_t, MAPE_t = test_error(STmodel, unknow_set, test_set, A, False)
        RMSE_list.append(RMSE_t)
        MAE_list.append(MAE_t)
        MAPE_list.append(MAPE_t)
        print(epoch, MAE_t, RMSE_t, MAPE_t)
    
    if to_plot:
        plot_res(RMSE_list,dataset,training_set.shape[0]//(h * batch_size))
    torch.save(STmodel.state_dict(), 'model/IGNNK_{:}_{:}iter_{:}.pth'.format(dataset,max_iter,time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))) # Save the model