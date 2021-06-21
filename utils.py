from __future__ import division
import os
import zipfile
import numpy as np
import scipy.sparse as sp
import pandas as pd
from math import radians, cos, sin, asin, sqrt
#from sklearn.externals import joblib
import joblib
import scipy.io
import torch
from torch import nn
"""
Geographical information calculation
"""
def get_long_lat(sensor_index,loc = None):
    """
        Input the index out from 0-206 to access the longitude and latitude of the nodes
    """
    if loc is None:
        locations = pd.read_csv('data/metr/graph_sensor_locations.csv')
    else:
        locations = loc
    lng = locations['longitude'].loc[sensor_index]
    lat = locations['latitude'].loc[sensor_index]
    return lng.to_numpy(),lat.to_numpy()

def haversine(lon1, lat1, lon2, lat2): 
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
 
    # haversine
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 
    return c * r * 1000


"""
Load datasets
"""

def load_metr_la_rdata():
    if (not os.path.isfile("data/metr/adj_mat.npy")
            or not os.path.isfile("data/metr/node_values.npy")):
        with zipfile.ZipFile("data/metr/METR-LA.zip", 'r') as zip_ref:
            zip_ref.extractall("data/metr/")

    A = np.load("data/metr/adj_mat.npy")
    X = np.load("data/metr/node_values.npy").transpose((1, 2, 0))
    X = X.astype(np.float32)

    return A, X

def generate_nerl_data():
    # %% Obtain all the file names
    filepath = 'data/nrel/al-pv-2006'
    files = os.listdir(filepath)

    # %% Begin parse the file names and store them in a pandas Dataframe
    tp = [] # Type
    lat = [] # Latitude
    lng =[] # Longitude
    yr = [] # Year
    pv_tp = [] # PV_type
    cap = [] # Capacity MW
    time_itv = [] # Time interval
    file_names =[]
    for _file in files:
        parse = _file.split('_')
        if parse[-2] == '5':
            tp.append(parse[0])
            lat.append(np.double(parse[1]))
            lng.append(np.double(parse[2]))
            yr.append(np.int(parse[3]))
            pv_tp.append(parse[4])
            cap.append(np.int(parse[5].split('MW')[0]))
            time_itv.append(parse[6])
            file_names.append(_file)
        else:
            pass

    files_info = pd.DataFrame(
        np.array([tp,lat,lng,yr,pv_tp,cap,time_itv,file_names]).T,
        columns=['type','latitude','longitude','year','pv_type','capacity','time_interval','file_name']
    )
    # %% Read the time series into a numpy 2-D array with 137x105120 size
    X = np.zeros((len(files_info),365*24*12))
    for i in range(files_info.shape[0]):
        f = filepath + '/' + files_info['file_name'].loc[i]
        d = pd.read_csv(f)
        assert d.shape[0] == 365*24*12, 'Data missing!'
        X[i,:] = d['Power(MW)']
        print(i/files_info.shape[0]*100,'%')

    np.save('data/nrel/nerl_X.npy',X)
    files_info.to_pickle('data/nrel/nerl_file_infos.pkl')
    # %% Get the adjacency matrix based on the inverse of distance between two nodes
    A = np.zeros((files_info.shape[0],files_info.shape[0]))

    for i in range(files_info.shape[0]):
        for j in range(i+1,files_info.shape[0]):
            lng1 = lng[i]
            lng2 = lng[j]
            lat1 = lat[i]
            lat2 = lat[j]
            d = haversine(lng1,lat1,lng2,lat2)
            A[i,j] = d

    A = A/7500 # distance / 7.5 km
    A += A.T + np.diag(A.diagonal())
    A = np.exp(-A)
    np.save('data/nrel/nerl_A.npy',A)

def load_nerl_data():
    if (not os.path.isfile("data/nrel/nerl_X.npy")
            or not os.path.isfile("data/nrel/nerl_A.npy")):
        with zipfile.ZipFile("data/nrel/al-pv-2006.zip", 'r') as zip_ref:
            zip_ref.extractall("data/nrel/al-pv-2006")
        generate_nerl_data()
    X = np.load('data/nrel/nerl_X.npy')
    A = np.load('data/nrel/nerl_A.npy')
    files_info = pd.read_pickle('data/nrel/nerl_file_infos.pkl')

    X = X.astype(np.float32)
    # X = (X - X.mean())/X.std()
    return A,X,files_info

def generate_ushcn_data():
    pos = []
    Utensor = np.zeros((1218, 120, 12, 2))
    Omissing = np.ones((1218, 120, 12, 2))
    with open("data/ushcn/Ulocation", "r") as f:
        loc = 0
        for line in f.readlines():
            poname = line[0:11]
            pos.append(line[13:30])
            with open("data/ushcn/ushcn.v2.5.5.20191231/"+ poname +".FLs.52j.prcp", "r") as fp:
                temp = 0
                for linep in fp.readlines():
                    if int(linep[12:16]) > 1899:
                        for i in range(12):
                            str_temp = linep[17 + 9*i:22 + 9*i]
                            p_temp = int(str_temp)
                            if p_temp == -9999:
                                Omissing[loc, temp, i, 0] = 0
                            else:
                                Utensor[loc, temp, i, 0] = p_temp
                        temp = temp + 1   
            with open("data/ushcn/ushcn.v2.5.5.20191231/"+ poname +".FLs.52j.tavg", "r") as ft:
                temp = 0
                for linet in ft.readlines():
                    if int(linet[12:16]) > 1899:
                        for i in range(12):
                            str_temp = linet[17 + 9*i:22 + 9*i]
                            t_temp = int(str_temp)
                            if t_temp == -9999:
                                Omissing[loc, temp, i, 1] = 0
                            else:
                                Utensor[loc, temp, i, 1] = t_temp
                        temp = temp + 1    
            loc = loc + 1
            
    latlon =np.loadtxt("data/ushcn/latlon.csv",delimiter=",")
    sim = np.zeros((1218,1218))

    for i in range(1218):
        for j in range(1218):
            sim[i,j] = haversine(latlon[i, 1], latlon[i, 0], latlon[j, 1], latlon[j, 0]) #RBF
    sim = np.exp(-sim/10000/10)

    joblib.dump(Utensor,'data/ushcn/Utensor.joblib')
    joblib.dump(Omissing,'data/ushcn/Omissing.joblib')
    joblib.dump(sim,'data/ushcn/sim.joblib')            

def load_udata():
    if (not os.path.isfile("data/ushcn/Utensor.joblib")
            or not os.path.isfile("data/ushcn/sim.joblib")):
        with zipfile.ZipFile("data/ushcn/ushcn.v2.5.5.20191231.zip", 'r') as zip_ref:
            zip_ref.extractall("data/ushcn/ushcn.v2.5.5.20191231/")
        generate_ushcn_data()
    X = joblib.load('data/ushcn/Utensor.joblib')
    A = joblib.load('data/ushcn/sim.joblib')
    Omissing = joblib.load('data/ushcn/Omissing.joblib')
    X = X.astype(np.float32)
    return A,X,Omissing

def load_sedata():
    assert os.path.isfile('data/sedata/A.mat')
    assert os.path.isfile('data/sedata/mat.csv')
    A_mat = scipy.io.loadmat('data/sedata/A.mat')
    A = A_mat['A']
    X = pd.read_csv('data/sedata/mat.csv',index_col=0)
    X = X.to_numpy()
    return A,X

def load_pems_data():
    assert os.path.isfile('data/pems/pems-bay.h5')
    assert os.path.isfile('data/pems/distances_bay_2017.csv')
    df = pd.read_hdf('data/pems/pems-bay.h5')
    transfer_set = df.as_matrix()
    distance_df = pd.read_csv('data/pems/distances_bay_2017.csv', dtype={'from': 'str', 'to': 'str'})
    normalized_k = 0.1

    dist_mx = np.zeros((325, 325), dtype=np.float32)

    dist_mx[:] = np.inf

    sensor_ids = df.columns.values.tolist()

    sensor_id_to_ind = {}

    for i, sensor_id in enumerate(sensor_ids):
        sensor_id_to_ind[sensor_id] = i
        
    for row in distance_df.values:
            if row[0] not in sensor_id_to_ind or row[1] not in sensor_id_to_ind:
                continue
            dist_mx[sensor_id_to_ind[row[0]], sensor_id_to_ind[row[1]]] = row[2]

    distances = dist_mx[~np.isinf(dist_mx)].flatten()
    std = distances.std()
    adj_mx = np.exp(-np.square(dist_mx / std))

    adj_mx[adj_mx < normalized_k] = 0

    A_new = adj_mx
    return transfer_set,A_new
"""
Dynamically construct the adjacent matrix
"""

def get_Laplace(A):
    """
    Returns the laplacian adjacency matrix. This is for C_GCN
    """
    if A[0, 0] == 1:
        A = A - np.diag(np.ones(A.shape[0], dtype=np.float32)) # if the diag has been added by 1s
    D = np.array(np.sum(A, axis=1)).reshape((-1,))
    D[D <= 10e-5] = 10e-5
    diag = np.reciprocal(np.sqrt(D))
    A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), A),
                         diag.reshape((1, -1)))
    return A_wave

def get_normalized_adj(A):
    """
    Returns the degree normalized adjacency matrix. This is for K_GCN
    """
    if A[0, 0] == 0:
        A = A + np.diag(np.ones(A.shape[0], dtype=np.float32)) # if the diag has been added by 1s
    D = np.array(np.sum(A, axis=1)).reshape((-1,))
    D[D <= 10e-5] = 10e-5    # Prevent infs
    diag = np.reciprocal(np.sqrt(D))
    A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), A),
                         diag.reshape((1, -1)))
    return A_wave

def calculate_random_walk_matrix(adj_mx):
    """
    Returns the random walk adjacency matrix. This is for D_GCN
    """
    adj_mx = sp.coo_matrix(adj_mx)
    d = np.array(adj_mx.sum(1))
    d_inv = np.power(d, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    random_walk_mx = d_mat_inv.dot(adj_mx).tocoo()
    return random_walk_mx.toarray()

def test_error_missing(STmodel, unknow_set, test_data, A_s, E_maxvalue, Missing0,test_truth):
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

    o = np.zeros([test_truth.shape[0]//time_dim*time_dim, test_inputs_s.shape[1]]) #Separate the test data into several h period

    for i in range(0, test_truth.shape[0]//time_dim*time_dim, time_dim):
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
    truth = test_truth[0:test_data.shape[0]//time_dim*time_dim]
    o[missing_index_s[0:test_data.shape[0]//time_dim*time_dim] == 1] = truth[missing_index_s[0:test_data.shape[0]//time_dim*time_dim] == 1]

    test_mask =  1 - missing_index_s[0:test_data.shape[0]//time_dim*time_dim]
    if Missing0 == True:
        test_mask[truth == 0] = 0
        o[truth == 0] = 0

    o_ = o[:,list(unknow_set)]
    truth_ = truth[:,list(unknow_set)]
    test_mask_ = test_mask[:,list(unknow_set)]

    MAE = np.sum(np.abs(o_ - truth_))/np.sum( test_mask_)
    RMSE = np.sqrt(np.sum((o_ - truth_)*(o_ - truth_))/np.sum( test_mask_) )
    # MAPE = np.sum(np.abs(o - truth)/(truth + 1e-5))/np.sum( test_mask)
    R2 = 1 - np.sum( (o_ - truth_)*(o_ - truth_) )/np.sum( (truth_ - truth_.mean())*(truth_-truth_.mean() ) )
    return MAE, RMSE, R2, o

def test_error(STmodel, unknow_set, test_data, A_s, E_maxvalue, Missing0):
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
    truth = test_inputs_s[0:test_data.shape[0]//time_dim*time_dim]
    o[missing_index_s[0:test_data.shape[0]//time_dim*time_dim] == 1] = truth[missing_index_s[0:test_data.shape[0]//time_dim*time_dim] == 1]
    test_mask =  1 - missing_index_s[0:test_data.shape[0]//time_dim*time_dim]
    if Missing0 == True:
        test_mask[truth == 0] = 0
        o[truth == 0] = 0
    o_ = o[:,list(unknow_set)]
    truth_ = truth[:,list(unknow_set)]
    test_mask_ = test_mask[:,list(unknow_set)]
    
    MAE = np.sum(np.abs(o_ - truth_))/np.sum( test_mask_)
    RMSE = np.sqrt(np.sum((o_ - truth_)*(o_ - truth_))/np.sum( test_mask_) )
    # MAPE = np.sum(np.abs(o - truth)/(truth + 1e-5))/np.sum( test_mask)
    R2 = 1 - np.sum( (o_ - truth_)*(o_ - truth_) )/np.sum( (truth_ - truth_.mean())*(truth_-truth_.mean() ) )
    return MAE, RMSE, R2, o


def rolling_test_error(STmodel, unknow_set, test_data, A_s, E_maxvalue,Missing0):
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

    o = np.zeros([test_data.shape[0] - time_dim, test_inputs_s.shape[1]])
    
    for i in range(0, test_data.shape[0] - time_dim):
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
    
 
    truth = test_inputs_s[time_dim:test_data.shape[0]]
    o[missing_index_s[time_dim:test_data.shape[0]] == 1] = truth[missing_index_s[time_dim:test_data.shape[0]] == 1]
    
    o = o*E_maxvalue
    truth = test_inputs_s[0:test_data.shape[0]//time_dim*time_dim]
    test_mask =  1 - missing_index_s[time_dim:test_data.shape[0]]
    if Missing0 == True:
        test_mask[truth == 0] = 0
        o[truth == 0] = 0
        
    MAE = np.sum(np.abs(o - truth))/np.sum( test_mask)
    RMSE = np.sqrt(np.sum((o - truth)*(o - truth))/np.sum( test_mask) )
    MAPE = np.sum(np.abs(o - truth)/(truth + 1e-5))/np.sum( test_mask)  #avoid x/0
        
    return MAE, RMSE, MAPE, o

def test_error_cap(STmodel, unknow_set, full_set, test_set, A,time_dim,capacities):
    unknow_set = set(unknow_set)
    
    test_omask = np.ones(test_set.shape)
    test_omask[test_set == 0] = 0
    test_inputs = (test_set * test_omask).astype('float32')
    test_inputs_s = test_inputs#[:, list(proc_set)]

    
    missing_index = np.ones(np.shape(test_inputs))
    missing_index[:, list(unknow_set)] = 0
    missing_index_s = missing_index#[:, list(proc_set)]
    
    A_s = A#[:, list(proc_set)][list(proc_set), :]
    o = np.zeros([test_set.shape[0]//time_dim*time_dim, test_inputs_s.shape[1]])
    
    for i in range(0, test_set.shape[0]//time_dim*time_dim, time_dim):
        inputs = test_inputs_s[i:i+time_dim, :]
        missing_inputs = missing_index_s[i:i+time_dim, :]
        MF_inputs = inputs*missing_inputs
        MF_inputs = MF_inputs
        MF_inputs = np.expand_dims(MF_inputs, axis = 0)
        MF_inputs = torch.from_numpy(MF_inputs.astype('float32'))
        A_q = torch.from_numpy((calculate_random_walk_matrix(A_s).T).astype('float32'))
        A_h = torch.from_numpy((calculate_random_walk_matrix(A_s.T).T).astype('float32'))
        
        imputation = STmodel(MF_inputs, A_q, A_h)
        imputation = imputation.data.numpy()
        o[i:i+time_dim, :] = imputation[0, :, :]
    
    o = o*capacities
    truth = test_inputs_s[0:test_set.shape[0]//time_dim*time_dim]
    truth = truth*capacities
    o[missing_index_s[0:test_set.shape[0]//time_dim*time_dim] == 1] = truth[missing_index_s[0:test_set.shape[0]//time_dim*time_dim] == 1]
    o[truth == 0] = 0
    
    test_mask =  1 - missing_index_s[0:test_set.shape[0]//time_dim*time_dim]
    test_mask[truth == 0] = 0
    
    o_ = o[:,list(unknow_set)]
    truth_ = truth[:,list(unknow_set)]
    test_mask_ = test_mask[:,list(unknow_set)]

    MAE = np.sum(np.abs(o_ - truth_))/np.sum( test_mask_)
    RMSE = np.sqrt(np.sum((o_ - truth_)*(o_ - truth_))/np.sum( test_mask_) )
    # MAPE = np.sum(np.abs(o - truth)/(truth + 1e-5))/np.sum( test_mask)
    R2 = 1 - np.sum( (o_ - truth_)*(o_ - truth_) )/np.sum( (truth_ - truth_.mean())*(truth_-truth_.mean() ) )
    return MAE, RMSE, R2, o
