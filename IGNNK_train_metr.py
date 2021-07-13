from __future__ import division

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
from scipy.io import loadmat
import copy


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
        # print(X.shape)
        # dis = loadmat("IGNNK_author/dist_metrla.mat")
        # dis = dis['dis']
        # dis_mx = dis / 100
        # A = np.exp(-np.power(dis_mx / 5.57, 2))
        # A[A < 0.01] = 0

    split_line1 = int(X.shape[1] * 0.7)

    training_set = X[:,:split_line1].transpose()
    print('training_set',training_set.shape)
    test_set = X[:, split_line1:].transpose()       # split the training and test period
    rand = np.random.RandomState(0) # Fixed random output
    # unknow_set = rand.choice(list(range(0,X.shape[0])),n_u,replace=False)
    unknow_set = np.load("data/metr/unknow_infer.npy")
    unknow_set = set(unknow_set)

    full_set = set(range(0,X.shape[0]))        
    know_set = full_set - unknow_set
    training_set_s = training_set[:, list(know_set)]   # get the training data in the sample time period
    A_s = A[:, list(know_set)][list(know_set), :]      # get the observed adjacent matrix from the full adjacent matrix,
                                                    # the adjacent matrix are based on pairwise distance, 
                                                    # so we need not to construct it for each batch, we just use index to find the dynamic adjacent matrix  
    return A,X,training_set,test_set,unknow_set,full_set,know_set,training_set_s,A_s,capacity     

"""
Define the test error
"""
def test_error(STmodel, unknow_set, test_data, A_s, Missing0, device):
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
        T_inputs = torch.from_numpy(T_inputs.astype('float32')).to(device)
        A_q = torch.from_numpy((calculate_random_walk_matrix(A_s).T).astype('float32')).to(device)
        A_h = torch.from_numpy((calculate_random_walk_matrix(A_s.T).T).astype('float32')).to(device)
        
        imputation = STmodel(T_inputs, A_q, A_h)
        imputation = imputation.cuda().data.cpu().numpy()
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
    
    return MAE, RMSE, MAPE, o, truth


if __name__ == "__main__":
    """
    Model training
    """

    dataset= "metr"
    n_o_n_m = 150
    h = 24
    z = 100
    K = 1
    n_m = 50
    n_u = 50
    max_iter = 200
    learning_rate = 0.001
    E_maxvalue = 80
    batch_size = 4
    to_plot = True
    device = torch.device("cuda:0")

    save_path = "./result_best/k=%d_T=%d_Z=%d/%s/" % (K, h, z, dataset)

    # load dataset
    A,X,training_set,test_set,unknow_set,full_set,know_set,training_set_s,A_s,capacity = load_data(dataset)
    # Define model
    STmodel = IGNNK(h, z, K)  # The graph neural networks
    STmodel.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(STmodel.parameters(), lr=learning_rate)
    RMSE_list = []
    MAE_list = []
    MAPE_list = []
    pred = []
    truth = []
    print('##################################    start training    ##################################')
    best_mae = 100000
    for epoch in range(max_iter):
        time_s = time.time()
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
            Mf_inputs = torch.from_numpy(Mf_inputs.astype('float32')).to(device)
            mask = torch.from_numpy(inputs_omask.astype('float32')).to(device)   #The reconstruction errors on irregular 0s are not used for training
            # print('Mf_inputs.shape = ',Mf_inputs.shape)

            A_dynamic = A_s[list(know_mask), :][:, list(know_mask)]   #Obtain the dynamic adjacent matrix
            A_q = torch.from_numpy((calculate_random_walk_matrix(A_dynamic).T).astype('float32')).to(device)
            A_h = torch.from_numpy((calculate_random_walk_matrix(A_dynamic.T).T).astype('float32')).to(device)
            
            if dataset == 'NREL':
                outputs = torch.from_numpy(inputs/capacities[:, None]).to(device)
            else:
                outputs = torch.from_numpy(inputs/E_maxvalue).to(device) #The label
            
            optimizer.zero_grad()
            X_res = STmodel(Mf_inputs, A_q, A_h)  #Obtain the reconstruction
            
            loss = criterion(X_res*mask, outputs*mask)
            loss.backward()
            optimizer.step()        #Errors backward
        if not dataset == 'NREL':
            MAE_t, RMSE_t, MAPE_t, pred, truth = test_error(STmodel, unknow_set, test_set, A, True, device)
        else:
            MAE_t, RMSE_t, MAPE_t, pred, truth = test_error(STmodel, unknow_set, test_set, A, False, device)
        time_e = time.time()
        RMSE_list.append(RMSE_t)
        MAE_list.append(MAE_t)
        MAPE_list.append(MAPE_t)
        print(epoch, MAE_t, RMSE_t, MAPE_t, 'time=',time_e - time_s)

        if MAE_t < best_mae:
            best_mae = MAE_t
            best_rmse = RMSE_t
            best_mape = MAPE_t
            best_epoch = epoch
            best_model = copy.deepcopy(STmodel.state_dict())
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            np.savez(save_path + "result.npz", pred=pred, truth=truth)

    torch.save(best_model, 'model/best_metr.pth') # Save the model
    print("###############     best_result:        ")
    print("epoch = ",best_epoch, "     mae = ",best_mae,"     rmse = ",best_rmse,"     mape = ",best_mape)