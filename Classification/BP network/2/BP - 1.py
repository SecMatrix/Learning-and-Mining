import numpy as np
import sys
default_encoding = 'utf-8'
if sys.getdefaultencoding() != default_encoding:
    reload(sys)
    sys.setdefaultencoding(default_encoding)
 
# sigmoid function
def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x)) 

def createDataSet():
    dataSet = [line.split() for line in open('1.dat').readlines()]
    newSet = []
    for d2 in dataSet:
        newList = []
        for d1 in d2:
            newList.append(int(d1))
        newSet.append(newList)
        
    dataList = []
    classList = []
    for data in newSet:
        dataList.append(data[:-1])
        classList.append(data[-1])
    
    return np.array(dataList),np.array([classList]).T

def bp1Train(X,y):
    np.random.seed(1)
    # initialize weights randomly with mean 0
    syn0 = 2*np.random.random((3,1)) - 1
    
    for iter in xrange(10000):
        # forward propagation
        l0 = X
        l1 = nonlin(np.dot(l0,syn0))
 
        # how much did we miss?
        l1_error = y - l1
 
        # multiply how much we missed by the 
        # slope of the sigmoid at the values in l1
        l1_delta = l1_error * nonlin(l1,True)
 
        # update weights
        syn0 += np.dot(l0.T,l1_delta)
    #print "sOutput After Training:"
    #print syn0
    return syn0
    
def main():
    dataList,classList = createDataSet()
    X = dataList[:]
    y = classList[:]
    #train
    syn0 = bp1Train(X,y)
    #run
    print np.dot(dataList,syn0)
 

 
