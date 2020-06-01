from __future__ import division
import os
import zipfile
import numpy as np
import scipy.sparse as sp
import pandas as pd
from math import radians, cos, sin, asin, sqrt
from sklearn.externals import joblib
import scipy.io
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
