# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 23:12:29 2020

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

a = np.array([[1],[2],[3],[4]])
b = np.array([[1,2,3],[4,5,6],[7,8,9]])

# Sigmoid 
def sigmoid(x):
    return (1/(1+np.exp(-x)))

# Regularized Cross Entropy Loss
def crossEntropyLoss(w, x, y, reg):
    y_cap = sigmoid(x,w)
    loss = ((np.matmul(np.log(y_cap),-1*y)-np.matmul(np.log(1-y_cap),(-1*y)+1))/y.size)+(0.5*reg*np.ndarray.item(np.matmul(w.T,w)))
    return loss

def gradCE(w, x, y, reg):
    
    return grad