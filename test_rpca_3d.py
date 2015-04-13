from rpca import rpca
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import sys
import logging

logger = logging.getLogger('rpca')
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stderr))

def randrange(n, vmin, vmax):
    return (vmax-vmin)*np.random.randn(n) + vmin

fig = plt.figure()
ax = fig.add_subplot(131, projection='3d')
n = 100
xa,ya,za = [],[],[]
for c, m, zl, zh in [('r', 'o', 25, -25), ('b', '^', 5, -5)]:
    xs = randrange(n, -100, 100)
    ys = randrange(n, -50, 50)
    zs = randrange(n, zl, zh)
    xa.append(xs); ya.append(ys); za.append(zs);
    ax.scatter(xs, ys, zs, c=c, marker=m)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

xa = np.hstack(xa); ya = np.hstack(ya); za = np.hstack(za);
data = np.vstack([xa,ya,za])
L,S = rpca(data.T, eps=0.00001, r=2)
ax.scatter(*(L.T), c='b', marker='*')

ax = fig.add_subplot(132, projection='3d')
ax.scatter(*L.T, c='r', marker='o')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

ax = fig.add_subplot(133, projection='3d')
ax.scatter(*(S.T), c='r', marker='o')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()

