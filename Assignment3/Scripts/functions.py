# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 10:54:09 2020

@author: akshay
"""
import matplotlib.pyplot as plt
import numpy as np

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
                if(np.isin(x[j,:],mu).all()):
                    continue
                maxD = dist
                maxDind = j

        # Set Mu
        mu[i,:] = x[maxDind,:]
        
        tempCent = np.zeros([1,x.shape[1]])
        for k in range(0,i+1):
            tempCent = tempCent + mu[k,:]
        centroid = tempCent / (i+1)
    
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
