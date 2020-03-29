# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 10:27:14 2020

@author: akshay
"""

import matplotlib.pyplot as plt
import numpy as np

from functions import distanceFunc, KMinit, lossFunc, Kmeans

# Plot Toy Data
def plotTestOP(toyData,toyFinalMu,realMu): 
    clusterIndexToy = (np.argmin(distanceFunc(toyData,toyFinalMu),axis=1))
    clusterIndexReal = (np.argmin(distanceFunc(toyData,realMu),axis=1))

    fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2,figsize=(15, 6))
    for i in range(0,toyFinalMu.shape[0]):
        cluster = toyData[clusterIndexToy==i,:]
        ax1.scatter(cluster[:,0],cluster[:,1],label="Cluster-{}".format(i+1))
    ax1.scatter(toyFinalMu[:,0],toyFinalMu[:,1],color="black",marker="x",label="Cluster Center")
    ax1.set_xlabel('X1')
    ax1.set_ylabel('X2')
    ax1.set_ylim(np.min(toyData[:,1])-5,np.max(toyData[:,1])+5)
    ax1.legend(loc="lower center")
    ax1.set_title("K = {} | Clustered Using K Means".format(toyFinalMu.shape[0]))
    
    # Plot Real Data
    for i in range(0,realMu.shape[0]):
        cluster = toyData[clusterIndexReal==i,:]
        ax2.scatter(cluster[:,0],cluster[:,1],label="Cluster-{}".format(i+1))
    ax2.scatter(realMu[:,0],realMu[:,1],color="black",marker="x",label="Cluster Center")
    ax2.set_xlabel('X1')
    ax2.set_ylabel('X2')
    ax2.set_ylim(np.min(toyData[:,1])-5,np.max(toyData[:,1])+5)
    ax2.legend(loc="lower center")
    ax2.set_title("K = {} | Ground Truth".format(realMu.shape[0]))

np.random.seed(7)

centre1 = (25,25)
x1 = np.array([np.random.normal(centre1[0],1,size=(20,))])
y1 = np.array([np.random.normal(centre1[1],1,size=(20,))])

centre2 = (-25,25)
x2 = np.array([np.random.normal(centre2[0],1,size=(20,))])
y2 = np.array([np.random.normal(centre2[1],1,size=(20,))])

centre3 = (0,25)
x3 = np.array([np.random.normal(centre3[0],1,size=(20,))])
y3 = np.array([np.random.normal(centre3[1],1,size=(20,))])

# Data
x = np.concatenate((x1,x2,x3),axis=1)
y = np.concatenate((y1,y2,y3),axis=1)
toyData = (np.concatenate((x,y))).T

realMu = np.array([[centre1[0],centre1[1]],[centre2[0],centre2[1]],[centre3[0],centre3[1]]])

print('\n\nCommence Test 2 with K = 3\n\n')
print('Toy Dataset 2 is drawn from 3 normal distributions centered at (25,25), (-25,25) and (0,25)\n')

plt.figure()
plt.scatter(toyData[:,0],toyData[:,1],color='b')
plt.xlim(-30,30)
plt.ylim(20,30)
plt.title('Toy Dataset 2')
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()

# Testing KMinit

toySetMu = KMinit(toyData,3)

plt.figure()
plt.scatter(realMu[:,0],realMu[:,1],color='black',marker='x',label='Ground Truth')
plt.scatter(toySetMu[:,0],toySetMu[:,1],color='red',marker='x',label='Predicted by KMinit')
plt.title('Cluster Centres')
plt.legend(loc='lower center')
plt.xlabel('X1')
plt.ylabel('X2')
plt.show

# Test 2 : Testing distanceFunc

toyPairWiseDist = distanceFunc(toyData,toySetMu)
toyPairWiseDistReal = distanceFunc(toyData,realMu)

# Test 3 : lossFunc

toyLoss = lossFunc(toyPairWiseDist)
realLoss = lossFunc(toyPairWiseDistReal)

# Test 4 : KMeans

toyFinalMu, toyFinalLoss = Kmeans(toyData,3)

# Plot

plotTestOP(toyData,toyFinalMu,realMu)

plt.figure()
plt.bar(1,toyLoss,label='Before Running KMeans = {:.2f}'.format(toyLoss))
plt.bar(3,realLoss,label='Ground Truth Loss = {:.2f}'.format(realLoss))
plt.bar(2,toyFinalLoss[-1],label='After Running KMeans = {:.2f}'.format(toyFinalLoss[-1,0]))
plt.legend(loc='upper right')
plt.ylabel('Loss')
plt.title('Losses')
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
plt.show()
