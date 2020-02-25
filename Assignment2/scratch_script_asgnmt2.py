# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 15:25:48 2020

@author: akshay
"""

# Imports
import matplotlib.pyplot as plt
import numpy as np

# Load Data Function
def loadData():
    with np.load('notMNIST.npz') as data:
        Data, Target = data['images'], data['labels']
        np.random.seed(521)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data = Data[randIndx]/255.0
        Target = Target[randIndx]
        trainData, trainTarget = Data[:15000], Target[:15000]
        validData, validTarget = Data[15000:16000], Target[15000:16000]
        testData, testTarget = Data[16000:], Target[16000:]
       
    return trainData, validData, testData, trainTarget, validTarget, testTarget

# One Hot Encoding
def convertOneHot(trainTarget, validTarget, testTarget):
    trainTargetOneHot = np.zeros([trainTarget.shape[0],10])
    trainTargetOneHot[np.arange(trainTarget.size),trainTarget]=1
    
    validTargetOneHot = np.zeros([validTarget.shape[0],10])
    validTargetOneHot[np.arange(validTarget.size),validTarget] = 1
    
    testTargetOneHot = np.zeros([testTarget.shape[0],10])
    testTargetOneHot[np.arange(testTarget.size),testTarget]=1
    
    
    return trainTargetOneHot, validTargetOneHot, testTargetOneHot

# Helper Functions
    
# ReLU
def relu(x):
    return max(0,x)

# SoftMax
def softmax(x):    
    op = np.exp(x)/sum(np.exp(x))
    return op

# Compute Layer
def computeLayer(x,W):
    # Assuming x includes bias
    return np.matmul(W.T,x)

# Load Data
trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()

trainData = trainData.reshape(trainData.shape[0],-1)
validData = validData.reshape(validData.shape[0],-1)
testData = testData.reshape(testData.shape[0],-1)

trainTargetOneHot, validTargetOneHot, testTargetOneHot = convertOneHot(trainTarget, validTarget, testTarget)

# 