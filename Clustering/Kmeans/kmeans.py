import numpy as np

def kmeans(X,k,maxIt):
    
    numPoints,numDim = X.shape
    
    dataSet = np.zeros((numPoints,numDim + 1))
    dataSet[:, :-1] = X
