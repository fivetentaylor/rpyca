#!/usr/bin/env python

import numpy as np
import scipy as sp
from scipy import sparse
#from scipy.sparse import linalg
from scipy import linalg
from scipy import stats

# Set default logging handler to avoid "No handler found" warnings.
import logging
try:  # Python 2.7+
    from logging import NullHandler
except ImportError:
    class NullHandler(logging.Handler):
        def emit(self, record):
            pass

logger = logging.getLogger(__name__)
logger.addHandler(NullHandler())


def tga(X, p=0.5, k=1, eps=1e-5):
    '''
    An implementation of the trimmed grassman average
    for robust principal component analysis
    outlined in "Grassmann Averages for Scalable Robust PCA"

    Parameters
    ----------
        X:  Input data matrix
        p:  Argument to scipy.stats.trim_mean
        k:  The number of components to compute

    Returns
    -------
        k robust components
    '''
    assert(len(X.shape) == 2)

    m,n = X.shape

    # Calculate the means and subtract them
    # from the data matrix
    means = np.mean(X, axis=0)
    X -= means

    vectors = np.zeros(k*n, dtype=X.dtype).reshape((k,n))

    for i in xrange(k):
        mu = np.rand(n) - 0.5
        mu /= np.linalg.norm(mu)

        for _ in xrange(3):
            dots = np.dot(X, mu)
            mu = np.dot(dots.T, X)
            mu /= np.linalg.norm(mu)

        for j in xrange(M):
            prev_mu = mu

            dot_signs = np.sign(np.dot(X, mu))

            mu = sp.stats.trim_mean(X * dot_signs, p, axis=1)
            mu /= np.linalg.norm(mu)

            if np.max(np.abs(mu - prev_mu)) < eps:
                break

        if i == 0:
            vectors[i] = mu
            X = X - np.dot(mu, np.dot(X, mu).T)
        elif i < k:
            # should reorthogonalize mu to the existing basis like:
            # mu = reorth(vectors[:i], mu)
            # mu /= np.linalg.norm(mu)
            # for numerical stability (but mu should already be orthogonal)
            vectors[i] = mu
            X = X - np.dot(mu, np.dot(X, mu).T)
        else i == k:
            # should reorthogonalize mu to the existing basis like:
            # mu = reorth(vectors[:i], mu)
            # mu /= np.linalg.norm(mu)
            # for numerical stability (but mu should already be orthogonal)
            vectors[i] = mu
            
    return vectors
            
        
