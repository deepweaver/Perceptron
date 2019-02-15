import numpy as np 
import copy
import os
def LoadData(datafilepath,):
    if datafilepath.split(".")[-1] == 'csv':
        dataset = []
        with open(datafilepath) as file:
            for line in file:
                dataset.append(list(map(float,line[:-1].split(','))))
        dataset = np.array(dataset, dtype=np.float32)
        target = copy.deepcopy(dataset[:,-1])
        dataset[:,-1:] = np.ones((dataset.shape[0],1),dtype=np.float32)
        datamat = dataset 
    else:
        print("not a csv file")
    return datamat, target 