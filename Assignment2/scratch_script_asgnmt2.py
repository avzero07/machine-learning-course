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
    return np.maximum(0,x)

# SoftMax
def softmax(x):
    # Stability Trick to Avoid NaN
    x = x - np.max(x)
    op = np.exp(x)/np.sum(np.exp(x))
    return op

# Compute Layer
def computeLayer(x,W):
    # Assuming x includes bias
    return np.matmul(x,W)

# CE Loss
def CE(target, prediction):
    # My Implementation is n x k
    return (-1.0/target.shape[0])*np.sum(np.multiply(target,np.log(np.apply_along_axis(softmax,1,prediction))))
    
# Grad CE
def gradCE(target, prediction):
    # Returns Sensitivity Vector for Layer L
    return softmax(prediction)-target

def trainNN(trainingData, trainingTarget, weightHidd, weightOp, numIter, eta, alpha, validationData, validationTarget, testData, testTarget):
    # eta   --> Learning Rate
    # alpha --> Momentum
    
    # Grab NN Dimensions
    numHiddenUnits = weightHidd.shape[1]
    numOpUnits = weightOp.shape[1]
    numIpVectors = trainingData.shape[0]
    numInputs = trainingData.shape[1]
    
    # Initialize Matrices
    velocityHidd = 1E-5 * (np.ones([weightHidd.shape[0],weightHidd.shape[1]]))
    velocityOp = 1E-5 * (np.ones([weightOp.shape[0],weightOp.shape[1]]))
    sToHidd = np.zeros([trainingData.shape[0],numHiddenUnits])
    sToOp = np.zeros([trainingData.shape[0],numOpUnits])
    
    lossesTrain = np.zeros([numIter,1])
    lossesValid = np.zeros([numIter,1])
    lossesTest = np.zeros([numIter,1])
    
    accuracyTrain = np.zeros([numIter,1])
    accuracyValid = np.zeros([numIter,1])
    accuracyTest = np.zeros([numIter,1])
    
    # Utility Functions
    
    # A. Easy ForwardProp
    def forwardProp(inputData,targetLabel,weightHidd,weightOp):
        # 1. Hidden Layer
        # Add Bias and Multiply with Weights to get S(1)
        sToHidd = computeLayer(np.append(np.ones((inputData.shape[0],1)),inputData,axis=1),weightHidd)
        # Calculate Activation to get X(1)
        xToOp = relu(sToHidd)
        
        # 2. Output Layer
        # Add Bias and Multiply with Weights to get S(L)
        sToOp = computeLayer(np.append(np.ones((xToOp.shape[0],1)),xToOp,axis=1),weightOp)
        # Calculate Activation to get h(x)
        fpassResult = np.apply_along_axis(softmax,1,sToOp)
        # Calculate Loss
        fpassLoss  = CE(targetLabel,fpassResult)
        
        return fpassResult, fpassLoss
    
    # B. Easy Classification Accuracies
    def classAccuracy(fpassResult,targetLabel):
        # Fpass Classification
        fpassClass = np.apply_along_axis(np.argmax,1,fpassResult)
        # True Classification
        trueClass = np.apply_along_axis(np.argmax,1,targetLabel)
        return np.sum(fpassClass==trueClass)/targetLabel.shape[0]
    
    i = 1
    while (i != numIter+1):        
        # Forward Propagation
        if i==5:
            print("here")
        # 1. Hidden Layer
        # Add Bias and Multiply with Weights to get S(1)
        sToHidd = computeLayer(np.append(np.ones((trainData.shape[0],1)),trainData,axis=1),weightHidd)
        # Calculate Activation to get X(1)
        xToOp = relu(sToHidd)
        
        # 2. Output Layer
        # Add Bias and Multiply with Weights to get S(L)
        sToOp = computeLayer(np.append(np.ones((xToOp.shape[0],1)),xToOp,axis=1),weightOp)
        # Calculate Activation to get h(x)
        hx = np.apply_along_axis(softmax,1,sToOp)
        
        # Calculate Loss
        loss = CE(trainingTarget,hx)
        lossesTrain[i-1,0] = loss
        
        # Back Propagation
        
        # Part 1 : At OP
        # 1. Grad w.r.t weightOp
        dEdWL = np.matmul((np.append(np.ones((xToOp.shape[0],1)),xToOp,axis=1)).T,(hx-trainingTarget))
        # 2. Velocity OP
        velocityOp = (alpha*velocityOp) - (eta*dEdWL)
        # 3. weightOp Update
        weightOp = weightOp + velocityOp
        
        # Part 2 : At Hidden
        # 1. Grad w.r.t weightHidd
        dedxl = np.matmul((hx-trainingTarget),weightOp.T)[:,1:] # Remember to Slice
        derRelu = (sToHidd>0).astype(int) # Derivative of ReLU
        dEdWl = np.matmul((np.append(np.ones((trainData.shape[0],1)),trainData,axis=1)).T,np.multiply(derRelu,dedxl))
        # 2. Velocity Hidden
        velocityHidd = (alpha*velocityHidd) - (eta*dEdWl)
        # 3. weightHidd Update
        weightHidd = weightHidd + velocityHidd
        
        # Report Accuracies and Losses
        
        # 1. Training Accuracy
        accuracyTrain[i-1,0] = classAccuracy(hx,trainingTarget)
        
        # 2. Validation Accuracy
        fpassResValid, lossesValid[i-1,0] = forwardProp(validationData,validationTarget,weightHidd,weightOp)
        accuracyValid[i-1,0] = classAccuracy(fpassResValid,validationTarget)
        
        # 3. Testing Accuracy
        fpassResTest, lossesTest[i-1,0] = forwardProp(testData,testTarget,weightHidd,weightOp)
        accuracyTest[i-1,0] = classAccuracy(fpassResTest,testTarget)
        
        print(i)
        # Increment Index
        i = i + 1 

    # Return Updated Weight Matrices, Losses, Classification Accuracies
    return weightHidd, weightOp, lossesTrain, lossesValid, lossesTest, accuracyTrain, accuracyValid, accuracyTest

# Load Data
trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()

trainData = trainData.reshape(trainData.shape[0],-1)
validData = validData.reshape(validData.shape[0],-1)
testData = testData.reshape(testData.shape[0],-1)

trainTargetOneHot, validTargetOneHot, testTargetOneHot = convertOneHot(trainTarget, validTarget, testTarget)

alpha = 0.9                 # Momentum
eta = 1E-5                  # Learning Rate
numIter = 200               # Epochs
numHiddenNeurons = 1000     # Number of Hidden Layer Neurons
numInputNodes = 784         # Excluding Bias
numOpNodes = 10             # 10 Classes

# Weight Matrix Initialization
variance = 2/(numInputNodes+numOpNodes)
standDev = np.sqrt(variance)
centre = 0.0

weightHiddenLayer = np.random.normal(loc=centre,scale=standDev,size=(numInputNodes+1,numHiddenNeurons))
weightOpLayer = np.random.normal(loc=centre,scale=standDev,size=(numHiddenNeurons+1,numOpNodes))

wHid, wOp, ltrain, lvalid, ltest, atrain, avalid, atest = trainNN(trainData, trainTargetOneHot, weightHiddenLayer, weightOpLayer,numIter,eta,alpha,validData,validTargetOneHot,testData,testTargetOneHot)

plt.plot(atrain,"-r",label="Training Set")
plt.plot(avalid,"-b",label="Validation Set")
plt.plot(atest,"-g",label="Test Set")
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc="upper left")
plt.title("Classification Accuracy vs Number of Epochs")
plt.show()

plt.plot(ltrain,"-r",label="Training Set")
plt.plot(lvalid,"-b",label="Validation Set")
plt.plot(ltest,"-g",label="Testing Set")
plt.xlabel('Epochs')
plt.ylabel('Cross Entropy Loss')
plt.legend(loc="upper right")
plt.title("Cross Entropy Loss vs Epochs")
plt.show()