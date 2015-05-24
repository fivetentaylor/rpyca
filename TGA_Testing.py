
# coding: utf-8

# # Robust PCA Example

# Robust PCA is an awesome relatively new method for factoring a matrix into a low rank component and a sparse component.  This enables really neat applications for outlier detection, or models that are robust to outliers.

# ### Make Some Toy Data

# In[49]:

import numpy as np


# In[50]:

def mk_rot_mat(rad=np.pi / 4):
    rot = np.array([[np.cos(rad),-np.sin(rad)], [np.sin(rad), np.cos(rad)]])
    return rot


# In[51]:

rot_mat = mk_rot_mat( np.pi / 4)
x = np.random.randn(100) * 5
y = np.random.randn(100)
points = np.vstack([y,x])


# In[52]:

rotated = np.dot(points.T, rot_mat).T


# ### Add Some Outliers to Make Life Difficult

# In[53]:

outliers = np.tile([15,-10], 10).reshape((-1,2))


# In[54]:

pts = np.vstack([rotated.T, outliers]).T


# ### Compute SVD on both the clean data and the outliery data

# In[55]:

U,s,Vt = np.linalg.svd(rotated)
U_n,s_n,Vt_n = np.linalg.svd(pts)


# ### Just 10 outliers can really screw up our line fit!

# ### Now the robust pca version!

# In[57]:

import tga


# In[58]:

reload(tga)


# In[59]:

import logging
logger = logging.getLogger(tga.__name__)
logger.setLevel(logging.INFO)


# ### Factor the matrix into L (low rank) and S (sparse) parts

# In[60]:

v = tga.tga(pts.T, eps=0.0000001, k=1)

