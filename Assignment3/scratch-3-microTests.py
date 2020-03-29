# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 14:15:23 2020

@author: akshay
"""

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from functions import distanceFunc, KMinit, lossFunc, Kmeans

def plotTestOP(toyData,toyFinalMu,realMu,toyLoss1,realLoss1,toyFinalLoss1,figNum): 
    clusterIndexToy = (np.argmin(distanceFunc(toyData,toyFinalMu),axis=1))
    clusterIndexReal = (np.argmin(distanceFunc(toyData,realMu),axis=1))

    fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2,figsize=(15, 6))
    for i in range(0,toyFinalMu.shape[0]):
        cluster = toyData[clusterIndexToy==i,:]
        ax1.scatter(cluster[:,0],cluster[:,1],label="Cluster-{}".format(i+1))
    ax1.scatter(toyFinalMu[:,0],toyFinalMu[:,1],color="black",marker="x",label="Cluster Center",s=50)
    ax1.set_xlabel('X1')
    ax1.set_ylabel('X2')
    ax1.set_ylim(np.min(toyData[:,1])-5,np.max(toyData[:,1])+5)
    ax1.legend(loc="lower center")
    ax1.set_title("Fig {}: K = {} | Clustered Using K Means".format(figNum,toyFinalMu.shape[0]))
    
    # Plot Real Data
    for i in range(0,realMu.shape[0]):
        cluster = toyData[clusterIndexReal==i,:]
        ax2.scatter(cluster[:,0],cluster[:,1],label="Cluster-{}".format(i+1))
    ax2.scatter(realMu[:,0],realMu[:,1],color="black",marker="x",label="Cluster Center",s=50)
    ax2.set_xlabel('X1')
    ax2.set_ylabel('X2')
    ax2.set_ylim(np.min(toyData[:,1])-5,np.max(toyData[:,1])+5)
    ax2.legend(loc="lower center")
    ax2.set_title("Fig {}: K = {} | Ground Truth".format(figNum+1,realMu.shape[0]))
    
    plt.figure()
    plt.bar(1,toyLoss1,label='Before Running KMeans = {:.2f}'.format(toyLoss1))
    plt.bar(2,toyFinalLoss1[-1],label='After Running KMeans = {:.2f}'.format(toyFinalLoss1[-1,0]))
    plt.bar(3,realLoss1,label='Ground Truth Loss = {:.2f}'.format(realLoss1))
    plt.legend(loc='upper right')
    plt.ylabel('Loss')
    plt.title('Fig {}: Losses'.format(figNum+2))
    plt.tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=False) # labels along the bottom edge are off
    plt.show()

# Testing KMinit - Straight Line in 2D (Defined Forward)
testData1 = np.array([np.linspace(1,10,10)]).T
testData1 = np.concatenate((testData1,np.ones((10,1))),axis=1)

KMinitTestData1 = KMinit(testData1,2)
print("\n\nTest 1 : Testing KMinit - Straight Line in 2D (Points Defined in Forward Order)")
# Plot
plt.figure(figsize=(9,6))
plt.scatter(testData1[:,0],testData1[:,1],marker='x',color='black',label='TestData1')
plt.scatter(KMinitTestData1[:,0],KMinitTestData1[:,1],marker='x',color='red',label='Center Determined by KMinit')
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend(loc='lower center')
plt.title('Fig 1: TestData1 and Initial Cluster Centers')
plt.show()

# Testing KMinit - 45 Degree Line in 3D (Defined Backwards)
testData2 = np.array([np.linspace(10,1,10)]).T
testData2 = np.concatenate((testData2,testData2,testData2),axis=1)

KMinitTestData2 = KMinit(testData2,2)
print("\n\nTest 2 : Testing KMinit - 45 Degree Line in 3D (Points Defined in Reverse Order)")
# Plot
fig = plt.figure(figsize=(9,6))
ax = fig.add_subplot(111,projection='3d')
ax.scatter(testData2[:,0],testData2[:,1],testData2[:,2],marker='x',color='black',label='TestData2')
ax.scatter(KMinitTestData2[:,0],KMinitTestData2[:,1],KMinitTestData2[:,2],marker='x',color='red',label='Center Determined by KMinit')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.legend(loc='upper left', numpoints=1, ncol=3, fontsize=8, bbox_to_anchor=(0, 0))
ax.set_title('Fig 2: TestData2 and Initial Cluster Centers')
plt.show()

# Testing DistanceFunc - Zero Distance
testData3 = np.ones((10,3)) # 3D Points
testMu3 = np.array([[1,1,1]])

pair_distTestData3 = distanceFunc(testData3,testMu3)
print("\n\nTest 3: Testing DistanceFunc - Zero Distance")
print("\nGiven centre ({},{},{})".format(testMu3[0,0],testMu3[0,1],testMu3[0,2]))
print("\nGiven data,")
print(testData3)
print('\nPairwise Distances should all be 0, the computed pairwise distances are,')
print(pair_distTestData3)

# Testing DistanceFunc - 3D Distance
testData4 = testData2
testMu4 = KMinitTestData2

pair_distTestData4 = distanceFunc(testData4,testMu4)
print("\n\nTest 4: Testing DistanceFunc - 3D Distance")
# Plot
plt.figure(figsize=(9,6))
ax = plt.subplot(111)
w = 0.3
xtickLabels = ['(10,10,10)','(9,9,9)','(8,8,8)','(7,7,7)','(6,6,6)','(5,5,5)','(4,4,4)','(3,3,3)','(2,2,2)','(1,1,1)']
rect1 = ax.bar(np.linspace(1,10,10),pair_distTestData4[:,0], width=w, color='g', align='center',label='Squared Distance from ({:1.0f},{:1.0f},{:1.0f})'.format(testMu4[0,0],testMu4[0,1],testMu4[0,2]))
rect2 = ax.bar(np.linspace(1,10,10)+w,pair_distTestData4[:,1], width=w, color='r', align='center',label='Squared Distance from ({:1.0f},{:1.0f},{:1.0f})'.format(testMu4[1,0],testMu4[1,1],testMu4[1,2]))
ax.set_xticks(np.arange(1,len(xtickLabels)+1),)
ax.set_xticklabels(xtickLabels,rotation=45)

def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
autolabel(rect1)
autolabel(rect2)

ax.legend(loc='upper center')
ax.set_xlabel('Data Points')
ax.set_ylabel('Squared Distance from Centers')
ax.set_title('Fig 3 : Pair-wise Distance of Datapoints From Centers')
plt.show()

# Testing lossFunc - for 3D Data
testData5 = testData4
pair_distTestData5 = distanceFunc(testData4,testMu4)  # Same Data from Previous Test
testLoss5 = lossFunc(pair_distTestData5)
print('\n\nTest 5: Testing lossFunc With Data From Previous Test')
print('\nGiven the pairwise-distances from the previous test (Fig.3), the output computed by lossFun should be the sum of shortest squared distance bars across all data points.')
print('\nComputing by hand, the loss is 180.')
print('The output from lossFunc = {:1.1f}'.format(testLoss5))

# Testing Kmeans - With Ordered Dataset

# Generate Data
    
testData6Center1 = (-5,-5)
disp = 2
x1 = np.array([[testData6Center1[0]-disp,testData6Center1[0],testData6Center1[0]+2]])
y1 = np.array([[testData6Center1[1]-disp,testData6Center1[1],testData6Center1[1]+2]])

testData6Center2 = (5,5)
x2 = np.array([[testData6Center2[0]-disp,testData6Center2[0],testData6Center2[0]+2]])
y2 = np.array([[testData6Center2[1]-disp,testData6Center2[1],testData6Center2[1]+2]])

# Data
x = np.concatenate((x1,x2),axis=1)
y = np.concatenate((y1,y2),axis=1)
testData6 = (np.concatenate((x,y))).T

realMu6 = np.array([[testData6Center1[0],testData6Center1[1]],[testData6Center2[0],testData6Center2[1]]])

print('\n\nTest 6: Test K Means for a simple 2D Dataset (testData6) with K = 2')
print('\ntestData6 is hand crafted, made of 2 clusters with centres at ({},{}) and ({},{})\n'.format(testData6Center1[0],testData6Center1[1],testData6Center2[0],testData6Center2[1]))

# Plot testData6
plt.figure()
plt.scatter(testData6[:,0],testData6[:,1],color='b')
plt.xlim(-10,10)
plt.ylim(-10,10)
plt.title('Fig 4: testData6')
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()

testData6Mu = KMinit(testData6,2) # Run KMinit
pair_distanceData6 = distanceFunc(testData6,testData6Mu) # Compute Pairwise-Distance
pair_distanceData6Real = distanceFunc(testData6,realMu6) # Ground Truth Pairwise-Distance

Loss6 = lossFunc(pair_distanceData6) # Compute Loss before First Iteration
realLoss6 = lossFunc(pair_distanceData6Real) # Compute Ground Truth Loss

# Testing KMeans
testData6FinalMu, testData6FinalLoss = Kmeans(testData6,2)
plotTestOP(testData6,testData6FinalMu,realMu6,Loss6,realLoss6,testData6FinalLoss,5)

# Testing Kmeans - With Randomly Generated Dataset

# Generate Data

np.random.seed(7)

testData7Center1 = (25,25)
x1 = np.array([np.random.normal(testData7Center1[0],1,size=(20,))])
y1 = np.array([np.random.normal(testData7Center1[1],1,size=(20,))])

testData7Center2 = (-25,25)
x2 = np.array([np.random.normal(testData7Center2[0],1,size=(20,))])
y2 = np.array([np.random.normal(testData7Center2[1],1,size=(20,))])

testData7Center3 = (0,25)
x3 = np.array([np.random.normal(testData7Center3[0],1,size=(20,))])
y3 = np.array([np.random.normal(testData7Center3[1],1,size=(20,))])

# Data
x = np.concatenate((x1,x2,x3),axis=1)
y = np.concatenate((y1,y2,y3),axis=1)
testData7 = (np.concatenate((x,y))).T

realMu7 = np.array([[testData7Center1[0],testData7Center1[1]],[testData7Center2[0],testData7Center2[1]],[testData7Center3[0],testData7Center3[1]]])

print('\n\nTest 7: Test K Means for a 2D Dataset (testData7) with K = 3')
print('\ntestData7 is drawn from 3 normal distributions with means at ({},{}), ({},{}) and ({},{})\n'.format(testData7Center1[0],testData7Center1[1],testData7Center2[0],testData7Center2[1],testData7Center3[0],testData7Center3[1]))

# Plot testData7
plt.figure()
plt.scatter(testData7[:,0],testData7[:,1],color='b')
plt.xlim(-30,30)
plt.ylim(20,30)
plt.title('Fig 8: testData7')
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()

testData7Mu = KMinit(testData7,3) # Run KMinit
pair_distanceData7 = distanceFunc(testData7,testData7Mu) # Compute Pairwise-Distance
pair_distanceData7Real = distanceFunc(testData7,realMu7) # Ground Truth Pairwise-Distance

Loss7 = lossFunc(pair_distanceData7) # Compute Loss before First Iteration
realLoss7 = lossFunc(pair_distanceData7Real) # Compute Ground Truth Loss

# Testing KMeans
testData7FinalMu, testData7FinalLoss = Kmeans(testData7,3)
plotTestOP(testData7,testData7FinalMu,realMu7,Loss7,realLoss7,testData7FinalLoss,9)