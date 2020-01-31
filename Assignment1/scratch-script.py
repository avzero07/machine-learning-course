# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 23:55:37 2020

@author: akshay
"""

import matplotlib.pyplot as plt
import numpy as np

def loadData():
    with np.load('notMNIST.npz') as data:
        Data, Target = data['images'], data['labels']
        posClass = 2
        negClass = 9
        dataIndx = (Target==posClass) + (Target==negClass)
        Data = Data[dataIndx]/255.0
        Target = Target[dataIndx].reshape(-1, 1)
        Target[Target==posClass] = 1
        Target[Target==negClass] = 0
        np.random.seed(1)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data, Target = Data[randIndx], Target[randIndx]
        trainData, trainTarget = Data[:3500], Target[:3500]
        validData, validTarget = Data[3500:3600], Target[3500:3600]
        testData, testTarget = Data[3600:], Target[3600:]
       
    return trainData, validData, testData, trainTarget, validTarget, testTarget

def ErrorRate (w, x, y):

    # YOUR CODE HERE
    #raise NotImplementedError()
    
    # Assuming w, x and y are NumPy Arrays
    
    # Find y_cap
    y_cap = ((np.matmul(x,w))>0).astype(int)
    # Note y_cap needs to be changed to [-1,1]
    y_cap[y_cap<1]=-1
    # Calculate Total Loss for N Samples
    loss = (((y_cap!=y).astype(float)).sum())/y.size
    
    return loss

def PLA(w, x, y, maxIter):

    # YOUR CODE HERE
    #raise NotImplementedError()
    
    # Assuming w, x and y are NumPy Arrays
    
    # Start PLA Loop
    while((maxIter>0)and(ErrorRate(w,x,y)>0)):
        
        # Pick Random Sample To Use for Weight Update
        
        # First Identify Misclassified Samples
        y_cap = ((np.matmul(x,w))>0).astype(int)
        # Note y_cap needs to be changed to [-1,1]
        y_cap[y_cap<1]=-1
        r = (y_cap!=y).astype(int)
        
        # Create an Array to Store Indices of
        # misclassified samples
        s = np.zeros(r.sum())
        
        sInd = 0 #Index for s
        i = r.shape[0] #Index for Inner Loop
        while(i>0):
            i-=1
            if r[i][0]==1:
                s[sInd]=i
                sInd+=1
        
        # Pick Random Index
        rInd = int(np.random.choice(s))
        #Weight Update
        tempx = x[rInd].reshape(x[rInd].shape[0],1)
        w = np.add(w,(y[rInd][0]*(tempx)))
        
        maxIter-=1
    return w

# YOUR CODE HERE
#raise NotImplementedError()

# Load Data
d = loadData()
trainData = d[0].reshape(3500,-1)
trainData = np.append(np.ones((trainData.shape[0],1)),trainData,axis=1)

validData = d[1].reshape(100,-1)
testData = d[2].reshape(145,-1)
testData = np.append(np.ones((testData.shape[0],1)),testData,axis=1)

trainTarget = d[3].astype(int)
trainTarget[trainTarget<1]=-1
validTarget = d[4].astype(int)
validTarget[validTarget<1]=-1
testTarget = d[5].astype(int)
testTarget[testTarget<1]=-1

# Training
w = PLA(np.zeros((785,1)),trainData,trainTarget,100) #Add 1 to w for bias

# Test
EClassTest = ErrorRate(w,testData,testTarget)
print("The Classification Error is {:f}".format(EClassTest))