# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 12:53:12 2020

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
    
# Generate Data
    
centre1 = (-5,-5)
disp = 2
x1 = np.array([[centre1[0]-disp,centre1[0],centre1[0]+2]])
y1 = np.array([[centre1[1]-disp,centre1[1],centre1[1]+2]])

centre2 = (5,5)
x2 = np.array([[centre2[0]-disp,centre2[0],centre2[0]+2]])
y2 = np.array([[centre2[1]-disp,centre2[1],centre2[1]+2]])

# Data
x = np.concatenate((x1,x2),axis=1)
y = np.concatenate((y1,y2),axis=1)
toyData1 = (np.concatenate((x,y))).T

realMu1 = np.array([[centre1[0],centre1[1]],[centre2[0],centre2[1]]])

print('\n\nCommence Test 1 with K = 2\n\n')
print('Toy Dataset 1 is hand crafted, made of 2 clusters with centres at ({},{}) and ({},{})\n'.format(centre1[0],centre1[1],centre2[0],centre2[1]))

plt.figure()
plt.scatter(toyData1[:,0],toyData1[:,1],color='b')
plt.xlim(-10,10)
plt.ylim(-10,10)
plt.title('Toy Dataset 1')
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()

# Testing KMinit

toySetMu1 = KMinit(toyData1,2)

plt.figure()
plt.scatter(realMu1[:,0],realMu1[:,1],color='black',marker='x',label='Ground Truth')
plt.scatter(toySetMu1[:,0],toySetMu1[:,1],color='red',marker='x',label='Predicted by KMinit')
plt.xlim(-10,10)
plt.ylim(-10,10)
plt.title('Cluster Centres')
plt.legend(loc='lower center')
plt.xlabel('X1')
plt.ylabel('X2')
plt.show

# Testing distanceFunc

toyPairWiseDist1 = distanceFunc(toyData1,toySetMu1)
toyPairWiseDistReal1 = distanceFunc(toyData1,realMu1)

# Testin lossFunc

toyLoss1 = lossFunc(toyPairWiseDist1)
realLoss1 = lossFunc(toyPairWiseDistReal1)

# Testing KMeans

toyFinalMu1, toyFinalLoss1 = Kmeans(toyData1,2)

# Plot

plotTestOP(toyData1,toyFinalMu1,realMu1)

plt.figure()
plt.bar(1,toyLoss1,label='Before Running KMeans = {:.2f}'.format(toyLoss1))
plt.bar(2,toyFinalLoss1[-1],label='After Running KMeans = {:.2f}'.format(toyFinalLoss1[-1,0]))
plt.bar(3,realLoss1,label='Ground Truth Loss = {:.2f}'.format(realLoss1))
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