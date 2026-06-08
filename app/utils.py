import math
import random
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

all_labels = ['VCC2SF1',
              'VCC2SF2',
              'VCC2SF3',
              'VCC2SF4',
              'VCC2SM1',
              'VCC2SM2',
              'VCC2SM3',
              'VCC2SM4',
              'VCC2TF1',
              'VCC2TF2',
              'VCC2TM1',
              'VCC2TM2']

def getId(label):
    return all_labels.index(label)

def getMcepSlice(mcep):
    if(len(mcep.shape) == 3):
        mcep = mcep.squeeze(0)

    max_start = max(0, mcep.shape[0] - 128)
    start = random.randint(0, max_start)
    return mcep[start:start + 128, :]

def calculateMetrics(converted, target):
    # Stripped the 0th coefficient (energy)
    conv_features = converted[:, 1:]
    targ_features = target[:, 1:]

    # Dynamic Time Warping to find the optimal alignment path
    _, path = fastdtw(conv_features, targ_features, dist = euclidean)
    
    path_conv, path_targ = zip(*path)
    
    aligned_conv = converted[list(path_conv)]
    aligned_targ = target[list(path_targ)]
    
    # MCD (using aligned sequences, dropping 0th coefficient)
    diff_mcd = aligned_conv[:, 1:] - aligned_targ[:, 1:]
    squared_diff_mcd = diff_mcd ** 2
    sum_squared_diff = np.sum(squared_diff_mcd, axis = -1)
    constant = (10.0 * math.sqrt(2.0)) / math.log(10.0)
    mcd = np.mean(np.sqrt(sum_squared_diff)) * constant
    
    # MSD (using aligned sequences, full coefficients)
    diff_msd = aligned_conv - aligned_targ
    squared_diff_msd = diff_msd ** 2
    msd = np.mean(np.sqrt(np.sum(squared_diff_msd, axis = -1)))
    
    return mcd, msd