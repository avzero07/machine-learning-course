# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 18:54:43 2020

@author: akshay
"""

import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from matplotlib.colors import LogNorm, Normalize

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
K = 3

# Create GMM
gmm = gaussMix(data2D,K)

# Plotting Contours of Density
X, Y = np.meshgrid(np.linspace(-4,5),np.linspace(-6,3))
XY = np.array([X.ravel(), Y.ravel()]).T
Z = -gmm.score_samples(XY)
Z = Z.reshape(X.shape)
# Visualize

fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2,figsize=(17, 6))
cont = ax1.contour(X,Y,Z,norm=Normalize(vmin=0, vmax=30),levels=np.logspace(0, 1.47, 15))
ax1.scatter(data2D[:,0],data2D[:,1], s=1)
ax1.set_title("Contours of Density")
ax1.set_xlabel('X1')
ax1.set_ylabel('X2')

ax2.scatter(data2D[:,0],data2D[:,1], c = gmm.predict(data2D), s=1)
ax2.set_xlabel('X1')
ax2.set_ylabel('X2')
ax2.set_title('Decision Boundaries of Clustering')
colors = ['red', 'turquoise', 'darkorange']
for n, color in enumerate(colors):
    covariances = gmm.covariances_[n][:2, :2]
    v, w = np.linalg.eigh(covariances)
    u = w[0] / np.linalg.norm(w[0])
    angle = np.arctan2(u[1], u[0])
    angle = 180 * angle / np.pi  # convert to degrees
    v = 2. * np.sqrt(2.) * np.sqrt(v)
    ell = mpl.patches.Ellipse(gmm.means_[n, :2], v[0], v[1],180 + angle, color=color)
    ell.set_clip_box(ax2.bbox)
    ell.set_alpha(0.2)
    ax2.add_artist(ell)
    ax2.set_aspect('equal', 'datalim')
plt.show()

# Decision Regions
X2, Y2 = np.meshgrid(np.linspace(-4,5,1000),np.linspace(-6,3,1000))
XY2 = np.array([X2.ravel(), Y2.ravel()]).T
Z2 = gmm.predict(XY2)
Z2 = Z2.reshape(X2.shape)

plt.figure(figsize=(17, 6))
plt.imshow(Z2,origin='lower')
plt.title('Decision Region Colored Over a Fine Meshgrid')
plt.xticks([])
plt.yticks([])
plt.locator_params(axis='both', nbins=10)
plt.xlabel('X1')
plt.ylabel('X2')
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