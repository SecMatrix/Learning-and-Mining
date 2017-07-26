import numpy
import SVM
import sys
default_encoding = 'utf-8'
if sys.getdefaultencoding() != default_encoding:
    reload(sys)
    sys.setdefaultencoding(default_encoding)

################## test svm #####################
## step 1: load data
print ("step 1: load data...")
dataSet = []
labels = []
fileIn = open('data/bigdata.txt')
for line in fileIn.readlines():
	lineArr = line.strip().split(' ')
	dataSet.append([float(lineArr[0]), float(lineArr[1])])
	labels.append(float(lineArr[2]))

dataSet = numpy.mat(dataSet)
labels = numpy.mat(labels).T

train_x = dataSet[:1000, :]
train_y = labels[:1000, :]
test_x = dataSet[:, :]
test_y = labels[:, :]

import time
start = time.clock()
## step 2: training...
print ("step 2: training...")
C = 0.6
toler = 0.001
maxIter = 50
svmClassifier = SVM.trainSVM(train_x, train_y, C, toler, maxIter, kernelOption = ('linear', 0))

## step 3: testing
print ("step 3: testing...")
accuracy = SVM.testSVM(svmClassifier, test_x, test_y)

## step 4: show the result
print ("step 4: show the result...")	
print ('The classify accuracy is: %.3f%%' % (accuracy * 100))
end = time.clock()
print('Running  time: %s Seconds'%(end-start))
SVM.showSVM(svmClassifier)
