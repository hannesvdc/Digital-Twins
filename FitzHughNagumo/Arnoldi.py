import numpy as np
import scipy.linalg as lg
import scipy.sparse.linalg as slg
import matplotlib.pyplot as plt

def rayleigh(A, q):
    return np.vdot(q, A(q)) / np.vdot(q, q)

def shiftInvertArnoldi(A, sigma, v0, tolerance, n=1000):
    M = v0.size
    B = slg.LinearOperator(shape=(M, M), matvec=lambda w: A(w) - sigma * w)
    q = np.copy(v0)

    h = np.zeros((n + 1, n))
    Q = np.zeros((M, n + 1))
    Q[:, 0] = q / np.sqrt(np.vdot(q, q))
    for k in range(1, M):
        q = slg.gmres(B, Q[:,k-1], atol=tolerance)[0]

        for j in range(k):
            h[j, k - 1] = np.vdot(Q[:, j], q)
            q = q - h[j, k - 1] * Q[:, j]
        h[k, k - 1] = np.sqrt(np.vdot(q, q))
        q = q / h[k, k-1]

        if h[k, k - 1] < tolerance:
            break
        Q[:, k] = q
    
    # Compute the eigenvalues of H using the QR-algorithm (I'm guessing this is what scipy.eig does)
    Hk = h[0:k, 0:k]
    lams_H, vecs_H = lg.eig(Hk)
    lams_A = sigma + 1.0 / lams_H
    index = np.argmin(np.real(lams_A))
    vec = np.dot(Q[:,0:k], vecs_H[:,index])

    return lams_A[index], vec
