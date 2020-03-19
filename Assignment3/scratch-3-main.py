# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 15:31:28 2020

@author: Akshay
"""

import matplotlib.pyplot as plt
import numpy as np

# K-Means Algorithm

data2D = np.load('data2D.npy')


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
        pair_dist2[:,i] = (np.reshape(np.linalg.norm(x-mu[i,:],ord=2,axis=1),(x.shape[0],1)))[:,0]
    
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
    
    color=iter(plt.cm.rainbow(np.linspace(0,1,mu.shape[0])))
    for i in range(0,mu.shape[0]):
        cluster = x[clusterIndex==i,:]
        plt.scatter(cluster[:,0],cluster[:,1],next(color),label="Cluster-{}".format(i+1))
    plt.scatter(mu[:,0],mu[:,1],color="b",marker="x",label="Cluster Center")
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.legend(loc="lower right")
    plt.title("K = {}".format(mu.shape[0]))
    plt.show()
    
    plt.plot(loss)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title("Loss vs Number of Iterations | K = {}".format(mu.shape[0]))
    plt.show()

# Integrated Test
mu, Loss = Kmeans(data2D,3)
plotClust(data2D,mu,Loss)