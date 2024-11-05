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

    h = np.zeros((n + 1, n), dtype=np.complex64)
    Q = np.zeros((M, n + 1), dtype=np.complex64)
    Q[:, 0] = q / np.sqrt(np.vdot(q, q))
    for k in range(1, M):
        q = slg.gmres(B, Q[:,k-1], x0=Q[:,k-1], atol=tolerance)[0]

        for j in range(k):
            h[j, k - 1] = np.vdot(Q[:, j], q)
            q = q - h[j, k - 1] * Q[:, j]
        h[k, k - 1] = np.sqrt(np.vdot(q, q))
        q = q / h[k, k-1]

        if h[k, k - 1] < tolerance:
            break
        Q[:, k] = q
    
    # Compute the eigenvalues of H using the QR-algorithm
    Hk = h[0:k, 0:k]
    lams_H, vecs_H = lg.eig(Hk)
    lams_A = sigma + 1.0 / lams_H

    index = np.argmin(np.real(lams_A))
    lam_A = lams_A[index]
    vec = np.dot(Q[:,0:k], vecs_H[:,index])
    if np.abs(lam_A - sigma) > np.abs(lam_A.conj() - sigma):
        lam_A = np.conjugate(lam_A)
        vec = np.conjugate(np.dot(Q[:,0:k], vecs_H[:,index]))

    return lam_A, vec

def shiftInvertArnoldiSimple(A, sigma, v0, tolerance):
    M = v0.size
    B = slg.LinearOperator(shape=(M, M), matvec=lambda w: A(w) - sigma * w)
    q = np.copy(v0)
    prev_coef = rayleigh(A, q)

    for _ in range(1, M):
        q = slg.gmres(B, q, x0=q, atol=tolerance)[0]
        q = q / np.vdot(q, q)

        coef = rayleigh(A, q)
        print(coef, np.abs((coef - prev_coef) / prev_coef))
        if np.abs((coef - prev_coef) / prev_coef) < tolerance:
            return coef, q
        prev_coef = coef
        
    return coef, q