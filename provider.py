import h5py
import os
import numpy as np
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    B, N, C = batch_data.shape
    assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1*clip, clip)
    jittered_data += batch_data
    jittered_data = jittered_data.astype(np.float32)
    
    return jittered_data

def select_random_point(batch_data, ratio=0.9):
    point_num = batch_data.shape[1]
    select_num = int(point_num * ratio)
    
    def f(data):
        idx = np.random.choice(range(point_num), select_num, replace=False)
        result = data[idx, :]
        
        return result
        
    result = np.array([f(data) for data in batch_data])
    
    return result

def getDataFiles(list_filename):
    return [line.rstrip() for line in open(list_filename)]

def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    return (data, label)

def loadTrainModel40(path):
    data = []
    label = []
    train_files = getDataFiles(
        os.path.join(BASE_DIR, path + '/modelnet40_ply_hdf5_2048/train_files.txt'))
    for fn in range(len(train_files)):
        current_data, current_label = load_h5(train_files[fn])
        data.append(current_data)
    data = np.concatenate(data)
    return data

def loadTestModel40(path):
    data = []
    label = []
    test_files = getDataFiles(
        os.path.join(BASE_DIR, path + '/modelnet40_ply_hdf5_2048/test_files.txt'))
    for fn in range(len(test_files)):
        current_data, current_label = load_h5(test_files[fn])
        data.append(current_data)
    data = np.concatenate(data)
    return data
