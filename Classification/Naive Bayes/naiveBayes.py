import numpy
import sys
default_encoding = 'utf-8'
if sys.getdefaultencoding() != default_encoding:
    reload(sys)
    sys.setdefaultencoding(default_encoding)
            
def createDataSet():
    data = [line.split() for line in open('connect-4.txt').readlines()]
    dataSet = data[1:6755]
    features = data[0]
    return dataSet,features
def createTestData():
    data = [line.split() for line in open('connect-4.txt').readlines()][1:]
    return data
    
def calcpLabels(dataSet):
    #count
    dataNum = len(dataSet)
    #count label
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    #class 占得总比例
    pLabels = {}
    for key in labelCounts:
        pLabels[key] = float(labelCounts[key])/dataNum
    return pLabels
    
def calPbyLabel(dataSet,dataNumAll,label,testVec):
    prob = 0.0
    dataNum = len(dataSet)
    for i in range(len(testVec)):
        count = 0.0000001;
        for featVec in dataSet:
            if featVec[i] == testVec[i]:
                count += 1
        prob += numpy.math.log(count/float(dataNum),2)
    prob += numpy.math.log(dataNum/float(dataNumAll),2)
    return prob
    
def classify(dataSet,pLabels,testVec):
    dataNum = len(dataSet)
    flag = 1
    maxLabel = 'no1'
    maxP = 0.0

    for key in pLabels:
        currentDataSet = []
        for featVec in dataSet:
            if featVec[-1] == key:
                currentDataSet.append(featVec)
        currentP = calPbyLabel(currentDataSet,dataNum,key,testVec)
        if flag == 1:
            maxLabel = key
            maxP = currentP
            flag = 0
        elif currentP > maxP:
            maxP = currentP
            maxLabel = key
    return maxLabel

def main():
    import time
    start = time.clock()
    
    dataSet,features = createDataSet()
    pLabels = calcpLabels(dataSet)
    #test
    #testVec = ['rain','cool','normal','strong']
    #maxLabel = classify(dataSet,pLabels,testVec)
    #print(maxLabel)
   
    testDataSet = createTestData()
    count = 0;
    num = 0;
    for td in testDataSet:
        tds = td[:-1]
        num += 1
        if classify(testDataSet,pLabels,tds) == td[-1]:
            #print('right!')
            count += 1
        #else: 
            #print('wrong!')
    print float(count)/num
    
    end = time.clock()
    print('Running kosarak500 time: %s Seconds'%(end-start))   
if __name__ == '__main__':
    main()