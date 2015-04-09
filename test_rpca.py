#!/usr/bin/env python

import numpy as np
from rpca import rpca

x = np.random.randn(100) * 5
y = np.random.randn(100)
points = np.vstack([y,x])

L,S = rpca(points.T, k=1)

print L
print S
