# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 18:54:43 2020

@author: akshay
"""

import matplotlib.pyplot as plt
import numpy as np

np.random.seed(421)

# Import GMM
from sklearn.mixture import GaussianMixture

def gaussMix(data,K):
    # Create GMM
    gmm = GaussianMixture(n_components=K)
    gmm.fit(data2D)
    
    return gmm

# Import Data
data2D = np.load('data2D.npy')

# Set K
K = 5

# Create GMM
gmm = gaussMix(data2D,K)

# Plotting Contours of Density
X, Y = np.meshgrid(np.linspace(-4,5),np.linspace(-6,3))
XY = np.array([X.ravel(), Y.ravel()]).T
Z = gmm.score_samples(XY)
Z = Z.reshape((50,50))

# Visualize
plt.figure()
plt.contour(X,Y,Z,K-1)
plt.scatter(data2D[:,0],data2D[:,1], c = gmm.predict(data2D), s=1)
plt.show()

# Loop over K = 1 to 5 and Plot BIC vs K
finBIC = np.zeros([5,1])

for i in range(1,6):
    gmmFin = gaussMix(data2D,i)
    finBIC[i-1,0] = gmmFin.bic(data2D)

# Plot BIC vs K
plt.figure()
plt.plot(finBIC,marker='o')
plt.xticks(np.arange(5),np.linspace(1,5,5))
plt.xlabel('K')
plt.ylabel('BIC')
plt.title('Bayesian Information Criterion (BIC) as a Function of K')
plt.show()