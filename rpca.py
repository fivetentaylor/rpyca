#!/usr/bin/env python

import ipdb

import numpy as np
import scipy as sp
from scipy import sparse
#from scipy.sparse import linalg
from scipy import linalg

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


def HT(M, t):
    '''
    Hard Threshold Function:

        Given matrix M

        M_ij = M_ij where M_ij >= t

        else

        M_ij = 0 where M_ij < t
    '''
    Mt = M.copy()
    Mt[Mt < t] = 0.0
    return Mt

def frob_norm(M):
    '''
    Frobenius norm of a sparse matrix
    '''
    return np.linalg.norm(M.data)

def error(A, B):
    '''
    The error function
    '''
    return frob_norm(A - B)

def rpca(M, eps=0.001, r=1):
    '''
    An implementation of the robust pca algorithm as
    outlined in [need reference]
    '''
    assert(len(M.shape) == 2)

    m,n = M.shape

    s = linalg.svd(M, compute_uv=False)
    B = 1 / np.sqrt(n) # threshold parameter
    thresh = B * s[0]

    # Initial Low Rank Component
    L = np.zeros(M.shape)
    # Initial Sparse Component
    S = HT(M - L, thresh)

    iterations = range(int(10 * np.log(n * B * frob_norm(M - S) / eps)))
    logger.info('Number of iterations: %d to achieve eps = %f' % (len(iterations), eps))

    for k in xrange(1, r+1):
        for t in iterations:

            U,s,Vt = linalg.svd(M - S, full_matrices=False)
            thresh = B * ( s[k] + s[k-1] * (1/2)**t )

            # Best rank k approximation of M - S
            ipdb.set_trace()
            L = np.dot(np.dot(U[:,:k], np.diag(s[:k])), Vt[:k])
            S = HT(M - L, thresh)

        if (B * s[k]) < (eps / (2*n)):
            break

    return L,S


