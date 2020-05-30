from __future__ import division

import torch
import numpy as np
import torch.optim as optim
from torch import nn
import matplotlib.pyplot as plt
from utils import load_metr_la_rdata, get_normalized_adj, get_Laplace, calculate_random_walk_matrix
import random
import pandas as pd
from basic_structure import D_GCN, C_GCN, K_GCN

'''
Hyper parameters
'''
n_o_n_m = 150 #sampled space dimension

h = 24 #sampled time dimension

z = 100 #hidden dimension for graph convolution

K = 1 #If using diffusion convolution, the actual diffusion convolution step is K+1

n_m = 50 #number of mask node during training

N_u = 50 #target locations, N_u locations will be deleted from the training data

Max_episode = 750 #max training episode

learning_rate = 0.0001 #the learning_rate for Adam optimizer

E_maxvalue = 80 #the max value from experience

batch_size = 8



'''
Load data
'''
A, X = load_metr_la_rdata()

split_line1 = int(X.shape[2] * 0.7)

training_set = X[:, 0, :split_line1].transpose()

test_set = X[:, 0, split_line1:].transpose()       # split the training and test period

rand = np.random.RandomState() # Fixed random output
unknow_set = rand.choice(list(range(0,X.shape[0])),N_u,replace=False)
unknow_set = set(unknow_set)
full_set = set(range(0,207))        
know_set = full_set - unknow_set
training_set_s = training_set[:, list(know_set)]   # get the sampled time period
A_s = A[:, list(know_set)][list(know_set), :]      # get the observed adjacent matrix from the full adjacent matrix,
                                                   # the adjacent matrix are based on pairwise distance, 
                                                   # so we need not to construct it for each batch, we just use index to find the dynamic adjacent matrix
                                                  

'''
Buitld the GNN
'''

class IGNNK(nn.Module):
    """
    GNN on ST datasets to reconstruct the datasets
   x_s
    |GNN_3
   H_2 + H_1
    |GNN_2
   H_1
    |GNN_1
  x^y_m     
    """
    def __init__(self, h, z, k): 
        super(IGNNK, self).__init__()
        self.time_dimension = h
        self.hidden_dimnesion = z
        self.order = K

        self.GNN1 = D_GCN(self.time_dimension, self.hidden_dimnesion, self.order)
        self.GNN2 = D_GCN(self.hidden_dimnesion, self.hidden_dimnesion, self.order)
        self.GNN3 = D_GCN(self.hidden_dimnesion, self.time_dimension, self.order)

    def forward(self, X, A_q, A_h):
        """
        :param X: Input data of shape (batch_size, num_timesteps, num_nodes)
        :A_q: The forward random walk matrix (num_nodes, num_nodes)
        :A_h: The backward random walk matrix (num_nodes, num_nodes)
        :return: Reconstructed X of shape (batch_size, num_timesteps, num_nodes)
        """  
        X_S = X.permute(0, 2, 1) # to correct the input dims 
        
        X_s1 = self.GNN1(X_S, A_q, A_h)
        X_s2 = self.GNN2(X_s1, A_q, A_h) + X_s1 #num_nodes, rank
        X_s3 = self.GNN3(X_s2, A_q, A_h) 

        X_res = X_s3.permute(0, 2, 1)
               
        return X_res
    
STmodel = IGNNK(h, z, K)  # The graph neural networks


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

"""
Model training
"""
criterion = nn.MSELoss()
optimizer = optim.Adam(STmodel.parameters(), lr=learning_rate)
RMSE_list = []
MAE_list = []
MAPE_list = []
for epoch in range(Max_episode):
    for i in range(training_set.shape[0]//(h * batch_size)):  #using time_length as reference to record test_error
        t_random = np.random.randint(0, high=(training_set_s.shape[0] - h), size=batch_size, dtype='l')
        know_mask = set(random.sample(range(0,training_set_s.shape[1]),n_o_n_m)) #sample n_o + n_m nodes
        feed_batch = []
        for j in range(batch_size):
            feed_batch.append(training_set_s[t_random[j]: t_random[j] + h, :][:, list(know_mask)]) #generate 8 time batches
        
        inputs = np.array(feed_batch)
        inputs_omask = np.ones(np.shape(inputs))
        inputs_omask[inputs == 0] = 0           # We found that there are irregular 0 values for METR-LA, so we treat those 0 values as missing data,
                                                # For other datasets, it is not necessary to mask 0 values
                                                
        missing_index = np.ones((inputs.shape))
        for j in range(batch_size):
            missing_mask = random.sample(range(0,n_o_n_m),n_m) #Masked locations
            missing_index[j, :, missing_mask] = 0
            
        Mf_inputs = inputs * inputs_omask * missing_index / E_maxvalue #normalize the value according to experience
        Mf_inputs = torch.from_numpy(Mf_inputs.astype('float32'))
        mask = torch.from_numpy(inputs_omask.astype('float32'))   #The reconstruction errors on irregular 0s are not used for training
        
        A_dynamic = A_s[list(know_mask), :][:, list(know_mask)]   #Obtain the dynamic adjacent matrix
        A_q = torch.from_numpy((calculate_random_walk_matrix(A_dynamic).T).astype('float32'))
        A_h = torch.from_numpy((calculate_random_walk_matrix(A_dynamic.T).T).astype('float32'))
        
        outputs = torch.from_numpy(inputs/E_maxvalue) #The label
        
        optimizer.zero_grad()
        X_res = STmodel(Mf_inputs, A_q, A_h)  #Obtain the reconstruction
        
        loss = criterion(X_res*mask, outputs*mask)
        loss.backward()
        optimizer.step()        #Errors backward
    
    MAE_t, RMSE_t, MAPE_t = test_error(STmodel, unknow_set, test_set, A, True)
    RMSE_list.append(RMSE_t)
    MAE_list.append(MAE_t)
    MAPE_list.append(MAPE_t)
    print(epoch, MAE_t, RMSE_t, MAPE_t)
    
    
"""
Draw Learning curves on testing error
"""    

fig,ax = plt.subplots()
ax.plot(RMSE_list,label='RMSE_on_test_set',linewidth=3.5)
ax.set_xlabel('Training Batch (x249)',fontsize=20)
ax.set_ylabel('RMSE',fontsize=20)
ax.tick_params(axis="x", labelsize=14)
ax.tick_params(axis="y", labelsize=14)
ax.legend(fontsize=16)
plt.grid(True)
plt.tight_layout()
plt.savefig('learning_curve_METR_LA.pdf')
