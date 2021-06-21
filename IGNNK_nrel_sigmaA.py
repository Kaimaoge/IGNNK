from __future__ import division
import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import torch.optim as optim
from torch import nn
import matplotlib.pyplot as plt
from utils import load_pems_data, get_normalized_adj, get_Laplace, calculate_random_walk_matrix,test_error
import random
import copy
import scipy
import scipy.io
import math
import pandas as pd
from basic_structure import D_GCN, C_GCN, K_GCN,IGNNK

n_o_n_m = 100 #sampled space dimension

h = 16 #sampled time dimension

z = 100 #hidden dimension for graph convolution

K = 1 #If using diffusion convolution, the actual diffusion convolution step is K+1

n_m = 30 #number of mask node during training

N_u = 30 #target locations, N_u locations will be deleted from the training data

Max_episode = 750 #max training episode

learning_rate = 0.0001 #the learning_rate for Adam optimizer

batch_size = 8

STmodel = IGNNK(h, z, K)

# A = np.load('NREL/nerl_A.npy')
from scipy.io import loadmat
dist_mx = loadmat('data/nrel/nrel_dist_mx_lonlat.mat')
dist_mx = dist_mx['nrel_dist_mx_lonlat']
dis = dist_mx/1e3
A = np.exp( -0.5* np.power( dis/14 ,2) )

X = np.load('data/nrel/nerl_X.npy')
files_info = pd.read_pickle('data/nrel/nerl_file_infos.pkl')

time_used_base = np.arange(84,228)
time_used = np.array([])
for i in range(365):
    time_used = np.concatenate((time_used,time_used_base + 24*12* i))
X=X[:,time_used.astype(np.int)]

# We only use 7:00am to 7:00pm as the target data, because otherwise the 0-values of periods without sunshine will greatly influence the results
#time_used_base = np.arange(84,228)
#time_used = np.array([])
#for i in range(365):
#    time_used = np.concatenate((time_used,time_used_base + 24*12* i))
#X=X[:,time_used.astype(np.int)]

capacities = np.array(files_info['capacity'])
capacities = capacities.astype('float32')
E_maxvalue = capacities.max() 

X = X.transpose()
print(X.shape,E_maxvalue)

split_line1 = int(X.shape[0] * 0.7)
training_set = X[:split_line1, :]
test_set = X[split_line1:, :]

rand = np.random.RandomState(0) # Fixed random output, just an example when seed = 0.
unknow_set = rand.choice(list(range(0,X.shape[1])),N_u,replace=False)
unknow_set = set(unknow_set)
full_set = set(range(0,X.shape[1]))        
know_set = full_set - unknow_set
training_set_s = training_set[:, list(know_set)]   # get the training data in the sample time period
A_s = A[:, list(know_set)][list(know_set), :] 

criterion = nn.MSELoss()
optimizer = optim.Adam(STmodel.parameters(), lr=0.0001)
RMSE_list = []
MAE_list = []
R2_list = []
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
            
        Mf_inputs = inputs* inputs_omask * missing_index / E_maxvalue #normalize the value according to experience
        Mf_inputs = torch.from_numpy(Mf_inputs.astype('float32'))
        mask = torch.from_numpy(inputs_omask.astype('float32'))
        
        A_dynamic = A_s[list(know_mask), :][:, list(know_mask)]   #Obtain the dynamic adjacent matrix
        A_q = torch.from_numpy((calculate_random_walk_matrix(A_dynamic).T).astype('float32'))
        A_h = torch.from_numpy((calculate_random_walk_matrix(A_dynamic.T).T).astype('float32'))
        
        outputs = torch.from_numpy((inputs/E_maxvalue).astype('float32')) #The label
        
        optimizer.zero_grad()
        X_res = STmodel(Mf_inputs, A_q, A_h)  #Obtain the reconstruction
        
        loss = criterion(X_res*mask, outputs*mask)
        loss.backward()
        optimizer.step()        #Errors backward
    
    MAE_t, RMSE_t, R2_t, metr_ignnk_res  = test_error(STmodel, unknow_set, test_set, A,E_maxvalue, True)
    RMSE_list.append(RMSE_t)
    MAE_list.append(MAE_t)
    R2_list.append(R2_t)
    if MAE_t == min(MAE_list):
        best_model = copy.deepcopy(STmodel.state_dict())

    print(epoch, MAE_t, RMSE_t, R2_t)
    
STmodel.load_state_dict(best_model)
torch.save(STmodel, "nrel_ignnk_sigmaA_20210119.pth")  
