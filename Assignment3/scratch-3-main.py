# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 15:31:28 2020

@author: Akshay
"""

import matplotlib.pyplot as plt
import numpy as np

# K-Means Algorithm

# Implementing Functions

#testData = data2D[0:5,:]
#testMu = np.array([[1,1],[-1,-1],[1,0]])

# Function 1
def distanceFunc(x, mu):
    # Inputs  
    # x: is an NxD data matrix (N observations and D dimensions)
    # mu: is an KxD cluster center matrix (K cluster centers and D dimensions)
    # Output
    # pair_dist2: is the NxK matrix of squared pairwise distances
    
    # YOUR CODE HERE
    # Initialize pair_dist2
    pair_dist2 = np.zeros([x.shape[0],mu.shape[0]])
    
    for i in range(0,mu.shape[0]):
        pair_dist2[:,i] = np.sum(np.square(x-mu[i,:]),axis=1)
    
    return pair_dist2

# Function 2
def KMinit(x, K):
    # Inputs
    # x: is an NxD data matrix 
    # K: number of clusters
    # Output
    # mu: is the KxD matrix of initial cluster centers using the "greedy approach" described on page 6-16 in the textbook. 
    # Remark: Always pick the first entry in the data set as the first center. 
    
    # YOUR CODE HERE
    # Initialize mu
    mu = np.zeros([K,x.shape[1]])
    
    for i in range(0,K):
        if i==0:
            mu[i,:] = x[i,:]
            centroid = x[i,:]
            continue
        maxD = 0
        maxDind = 0
        
        for j in range(0,x.shape[0]):
            dist = np.linalg.norm((centroid-x[j,:]),ord=2)
            if dist > maxD:
                # Check if Point is Already in Mu
                if(x[j,:] in mu):
                    continue
                maxD = dist
                maxDind = j

        # Set Mu
        mu[i,:] = x[maxDind,:]
        
        tempCent = np.zeros([1,2])
        for k in range(0,i+1):
            tempCent = tempCent + mu[k,:]
        centroid = tempCent / (i+1)
    
    return mu

# Need to compare Distance With All other Points
def KMinit2(x, K):
    # Inputs
    # x: is an NxD data matrix 
    # K: number of clusters
    # Output
    # mu: is the KxD matrix of initial cluster centers using the "greedy approach" described on page 6-16 in the textbook. 
    # Remark: Always pick the first entry in the data set as the first center. 
    
    # YOUR CODE HERE
    # Initialize mu
    mu = np.zeros([K,x.shape[1]])
    
    for i in range(0,K):
        if i==0:
            mu[i,:] = x[i,:]
            continue
        maxD = 0
        maxDind = 0
        for j in range(0,x.shape[0]):
            dist = np.linalg.norm((mu[i-1,:]-x[j,:]),ord=2)
            if dist > maxD:
                maxD = dist
                maxDind = j
        # Set Mu
        mu[i,:] = x[maxDind,:]
    
    return mu

# Function 3
def lossFunc(pair_dist2):
    # Input 
    # pair_dist2: is an NxK matrix of squared pairwise distances
    # Output
    # loss: error as defined in (6.5) in the textbook
    
    # YOUR CODE HERE
    # Initialize Loss
    loss = 0
    # Column Sum to Get Error per Cluster
    loss = np.sum(np.min(pair_dist2,axis=1))
    return loss

def Kmeans(x,K):
    # Inputs
    # x: is an NxD data matrix 
    # K: number of clusters
    # Outputs
    # mu: is the KxD of cluster centers  
    # loss: error as defined in (6.5) in the textbook 
    
    # YOUR CODE HERE
    
    # Initialize Loss
    loss = []
    
    # 1. Generate Initial Cluster Centres
    mu = KMinit(x,K)
    
    iter = 0
    while(1):
        iter = iter + 1
        # 2. Create Clusters
        pair_dist2 = distanceFunc(x,mu) # Compute Pair-wise Distance
        clusterIndex = (np.argmin(pair_dist2,axis=1)) # Map Data to Clusters
        # Clusters can be explicitly defined as x[clusterIndex==n,:]
        loss.append(lossFunc(pair_dist2))
        
        if(iter>1):
            if loss[iter-1]==loss[iter-2]:
                break
    
        # 3. Update Cluster Center to Centroid of Cluster
        for i in range(0,mu.shape[0]):
            mu[i,:] = np.mean(x[clusterIndex==i,:],axis=0)
        
    # Repeat 2 and 3 Until Ein stops decreasing
    
    loss = np.array(loss)
    loss = loss.reshape(loss.shape[0],1)
    
    return mu, np.array(loss)

def plotClust(x,mu,loss):
    pair_dist2 = distanceFunc(x,mu)
    clusterIndex = (np.argmin(pair_dist2,axis=1))
    
    fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2,figsize=(15, 6))
    
    color=iter(plt.cm.rainbow(np.linspace(0,1,mu.shape[0])))
    for i in range(0,mu.shape[0]):
        cluster = x[clusterIndex==i,:]
        ax1.scatter(cluster[:,0],cluster[:,1],next(color),label="Cluster-{}".format(i+1))
    ax1.scatter(mu[:,0],mu[:,1],color="black",marker="x",label="Cluster Center")
    ax1.set_xlabel('X1')
    ax1.set_ylabel('X2')
    ax1.legend(loc="lower right")
    ax1.set_title("K = {}".format(mu.shape[0]))
    
    ax2.plot(loss)
    ax2.set_xlabel('Iterations')
    ax2.set_ylabel('K-Means Loss')
    ax2.set_title("K-Means Loss vs Number of Iterations | K = {}".format(mu.shape[0]))

# Integrated Test
    
# Import Data
data2D = np.load('data2D.npy')

finalLoss = np.zeros([5,1])
for i in range(1,6):
    mu, Loss = Kmeans(data2D,i)
    finalLoss[i-1] = Loss[-1]
    plotClust(data2D,mu,Loss)

# Loss vs K
plt.figure()
plt.plot(np.linspace(1,5,5),finalLoss,marker='o')
plt.xlabel("K")
plt.ylabel("K-Means Loss")
plt.title("K-Means Loss as a Function of K")
plt.show()

# Gap Statistics
np.random.seed(421)

# Init Result Vector
resErr = np.zeros([10,5])

# Determine Bounding Box Around Data2D
def boxBound(x):
    x1min = np.min(data2D[:,0])
    x1max = np.max(data2D[:,0])
    x2min = np.min(data2D[:,1])
    x2max = np.max(data2D[:,1])
    return x1min, x1max, x2min, x2max

x1min, x1max, x2min, x2max = boxBound(data2D)

# Loop Over 10 Datasets
for i in range(0,10):
    
    # Init Dataset
    dataRandX = np.random.uniform(low=x1min,high=x1max,size=(data2D.shape[0],1))
    dataRandY = np.random.uniform(low=x2min,high=x2max,size=(data2D.shape[0],1))
    dataRand2D = np.concatenate((dataRandX,dataRandY),axis=1) 
    
    # Loop Over K = 1 to 5
    for j in range(1,6):
        muRand, LossRand = Kmeans(dataRand2D,j)
        resErr[i,j-1] = LossRand[-1]

# Find Average of K-Means Error Per Cluster Size
avgKmeans = (np.mean(resErr,axis=0,keepdims=True)).T

# Plot Avg K-Means Error
plt.figure()
plt.plot(np.linspace(1,5,5),finalLoss,marker='o',label='$E_{in}(K)$')
plt.plot(np.linspace(1,5,5),avgKmeans,marker='o',label='$E_{in}^{Rand}(K)$')
plt.xlabel("K")
plt.ylabel("K-Means Error")
plt.legend(loc="upper right")
plt.title("K-Means Error as a Function of K")
plt.show()

# Compute Gap Statistic
def gapStat(randKmeansError,KmeansError):
    # Assume randKmeansError is already average of Logs
    res = (randKmeansError) - np.log(KmeansError)
    return res

# Update for Computing Gap Stats
newResErr = np.log(resErr)
newAvgKmeans = (np.mean(newResErr,axis=0,keepdims=True)).T

gap = gapStat(newAvgKmeans,finalLoss)
# Plot Gap Statistics
plt.figure()
plt.plot(np.linspace(1,5,5),gap,marker='o')
plt.xlabel("K")
plt.ylabel("Gap Statistics")
plt.title("Gap Statistics as a Function of K")
plt.show()