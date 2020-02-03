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
d2 = loadData()
trainData2 = d2[0].reshape(3500,-1)
trainData2 = np.append(np.ones((trainData2.shape[0],1)),trainData2,axis=1)

validData2 = d2[1].reshape(100,-1)
validData2 = np.append(np.ones((validData2.shape[0],1)),validData2,axis=1)

testData2 = d2[2].reshape(145,-1)
testData2 = np.append(np.ones((testData2.shape[0],1)),testData2,axis=1)

trainTarget2 = d2[3].astype(int)
#trainTarget[trainTarget<1]=-1
validTarget2 = d2[4].astype(int)
#validTarget[validTarget<1]=-1
testTarget2 = d2[5].astype(int)
#testTarget[testTarget<1]=-1

a = np.array([[1],[2],[3]])
b = np.array([[1,2,3],[4,5,6],[7,8,9]])

# Sigmoid 
def sigmoid(x):
    return (1/(1+np.exp(-x)))

# Regularized Cross Entropy Loss
def crossEntropyLoss(w, x, y, reg):
    theta = sigmoid(np.matmul(x,w))
    loss = ((np.matmul((np.log(theta)).T,-1.0*y).item()-np.matmul((np.log(1-theta)).T,(-1.0*y)+1).item())/y.size)+(0.5*reg*np.ndarray.item(np.matmul(w.T,w)))
    return loss

def gradCE(w, x, y, reg):
    theta = sigmoid(np.matmul(x,w))
    theta = theta - y
    grad = ((np.matmul(x.T,theta))/y.size)+(reg*w)
    return grad

def grad_descent(w, x, y, eta, iterations, reg, error_tol):
    ceLossList = list()
    # 1 - w is already initialized to 0
    # 2 - Start Loop
    while iterations>0:
        # 3 - Compute Gradient
        grad = gradCE(w,x,y,reg)
        # 4 - Set Step Direction
        stepDir = -1.0*(grad/(np.linalg.norm(grad,2)))
        wOld = w
        # 5 - Update Weights
        w = w + (eta*stepDir)
        # Calculate Loss
        ceLossList.append(crossEntropyLoss(w,x,y,reg))
        # Early Terminate - Tolerance Condition
        if np.linalg.norm((wOld-w),2)<error_tol:
            return w, ceLossList
        iterations-=1
    return w, ceLossList, iterations

# Run Batch Gradient Descent
epochs = 5000
regParam = 0 # Lambda
learningRate1 = 0.005 # eta1
learningRate2 = 0.001 # eta2
learningRate3 = 0.0001 # eta3
errTolerance = 0.0000001 # tolerance

# For Learning Rate 1
op1 = grad_descent(np.zeros((trainData2.shape[1],1)),trainData2,trainTarget2,learningRate1,epochs,regParam,errTolerance)

# For Learning Rate 2
op2 = grad_descent(np.zeros((trainData2.shape[1],1)),trainData2,trainTarget2,learningRate2,epochs,regParam,errTolerance)

# For Learning Rate 3
op3 = grad_descent(np.zeros((trainData2.shape[1],1)),trainData2,trainTarget2,learningRate3,epochs,regParam,errTolerance)

plt.plot(op1[1],"-r",label="eta = 0.005")
plt.plot(op2[1],"-b",label="eta = 0.001")
plt.plot(op3[1],"-g",label="eta = 0.0001")
plt.xlabel('Epochs')
plt.ylabel('Cross Entropy Loss')
plt.legend(loc="upper right")
plt.title("Cross Entropy Loss vs Epochs")
plt.show()

# Compute Loss With Test Data

# With Eta = 0.005
ceLossEta1 = crossEntropyLoss(op1[0],testData2,testTarget2,regParam)

# With Eta = 0.001
ceLossEta2 = crossEntropyLoss(op2[0],testData2,testTarget2,regParam)

# With Eta = 0.0001
ceLossEta3 = crossEntropyLoss(op3[0],testData2,testTarget2,regParam)

print("Cross Entropy Loss computed for Test Data at Different Eta")
print("For eta = 0.005, the Test Loss = {:f}".format(ceLossEta1))
print("For eta = 0.001, the Test Loss = {:f}".format(ceLossEta2))
print("For eta = 0.0001, the Test Loss = {:f}".format(ceLossEta3))