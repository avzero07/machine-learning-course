# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 10:08:01 2020

@author: akshay
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.mixture import GaussianMixture

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
        
        tempCent = np.zeros([1,x.shape[1]])
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

def gaussMix(data,K):
    # Create GMM
    gmm = GaussianMixture(n_components=K)
    gmm.fit(data)
    
    return gmm


# Import Data
data100D = np.load('data100D.npy')

# K Means

finalLoss100 = np.zeros([6,1])
finBIC100 = np.zeros([6,1])

for i in range(3,9):
    mu, Loss = Kmeans(data100D,i)    
    finalLoss100[i-3,0] = Loss[-1]
    
    gmmFin = gaussMix(data100D,i)
    finBIC100[i-3,0] = gmmFin.bic(data100D)
    print(i)
    
# Plot K Means Loss vs K
fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2,figsize=(17, 6))    

ax1.plot(finalLoss100,marker='o',label='Data 100D')
ax1.set_xticks(np.arange(6), minor=False)
ax1.set_xticklabels(np.linspace(3,8,6), fontdict=None, minor=False)
ax1.set_xlabel('K')
ax1.set_ylabel('K Means Loss')
ax1.legend(loc="upper right")
titax1 = ax1.set_title('K Means Loss as a Function of K')

# Plot BIC vs K
ax2.plot(finBIC100,marker='o',label='Data 100D')
ax2.set_xticks(np.arange(6), minor=False)
ax2.set_xticklabels(np.linspace(3,8,6), fontdict=None, minor=False)
ax2.set_xlabel('K')
ax2.set_ylabel('BIC')
ax2.legend(loc="upper right")
titax2 = ax2.set_title('Bayesian Information Criterion (BIC) as a Function of K')

