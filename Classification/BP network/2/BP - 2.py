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

def bp2Train(X,y):
    np.random.seed(1)
    # initialize weights randomly with mean 0
    syn0 = 2*np.random.random((3,4)) - 1
    syn1 = 2*np.random.random((4,1)) - 1
    
    for j in xrange(10000):
        # Feed forward through layers 0, 1, and 2
        l0 = X
        l1 = nonlin(np.dot(l0,syn0))
        l2 = nonlin(np.dot(l1,syn1))
 
        # how much did we miss the target value?
        l2_error = y - l2
 
        #if (j%10000) == 0:
           # print "Error:" + str(np.mean(np.abs(l2_error)))
        
        # in what direction is the target value?
        # were we really sure? if so, don't change too much.
        l2_delta = l2_error*nonlin(l2,deriv=True)
        # how much did each l1 value contribute to the l2 error (according to the weights)?
        l1_error = l2_delta.dot(syn1.T)
        # in what direction is the target l1?
        # were we really sure? if so, don't change too much.
        l1_delta = l1_error * nonlin(l1,deriv=True)
        syn1 += l1.T.dot(l2_delta)
        syn0 += l0.T.dot(l1_delta)
    
    #print "sOutput After Training:"
    #print syn0
    return syn0,syn1
    
def main():
    dataList,classList = createDataSet()
    X = dataList[:]
    y = classList[:]
    #train
    syn0,syn1 = bp2Train(X,y)
    print X,syn0,syn1
    #run
    d1 = np.dot(dataList,syn0)
    print np.dot(d1,syn1)
 

 
