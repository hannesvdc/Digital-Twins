import numpy as np
import scipy.linalg as lg
import scipy.sparse.linalg as slg
import matplotlib.pyplot as plt

def shiftInvertArnoldi(A, sigma, v0, tolerance):
    M = v0.size
    B = slg.LinearOperator(shape=(M, M), matvec=lambda _v: A(_v) - sigma * _v)
    v = np.copy(v0)

    n = M-1
    v = np
    h = np.zeros((n + 1, n))
    Q = np.zeros((M, n + 1))
    Q[:, 0] = v / lg.norm(v) 
    for k in range(1, n + 1):
        v = slg.gmres(B, x0=v, atol=tolerance)

        for j in range(k):
            h[j, k - 1] = np.dot(Q[:, j].conj(), v)
            v = v - h[j, k - 1] * Q[:, j]
        h[k, k - 1] = lg.norm(v)

        if h[k, k - 1] < tolerance:  # Add the produced vector to the list, unless
            break
        Q[:, k] = v / h[k, k - 1]
    
    # Compute the eigenvalues of H using the QR-algorithm (I'm guessing this is what scipy.eig does)
    Hk = h[0:k, 0:k]
    lambdas, eigvs = lg.eig(Hk)
    min_index = np.argmin(np.real(lambdas))

    # Shift back with sigma and return the eigenvector as well (Should be v?)
    return sigma + lambdas[min_index], eigvs[:,min_index]

