import numpy as np
import scipy.linalg as lg
import scipy.sparse.linalg as slg

import warnings
warnings.filterwarnings("error")

# Calculate the Rayleigh Coefficient of a vector
def rayleigh(A, q):
    return np.vdot(q, A(q)) / np.vdot(q, q)

# Shift sigma with -0.01 to ensure we find an eigenvalue to the left of sigma
# because we are looking for Hopf eigenvalues that cross the imaginary axis from
# the right to the left. This corresponds to the Largest Magnitude shifted eigenvale
# 1 / (lambda - shift + 0.01).
def shiftInvertArnoldiSimple(A, sigma, v0, tolerance, report_tolerance=1.e-5):
    M = v0.size

    for log_shift in range(10):
        shift = sigma - (log_shift + 1.0) * 0.01
        B = slg.LinearOperator(shape=(M, M), matvec=lambda w: A(w) - shift * w)
        q = np.copy(v0)

        try:
            for iter in range(1, M):
                q = slg.gmres(B, q, x0=q, atol=tolerance)[0]
                q = q / np.sqrt(np.vdot(q, q))

                coef = rayleigh(A, q)
                error = lg.norm(A(q) - coef * q)
                print(coef, error)
                if error < report_tolerance:
                    return coef, q

                if iter > 10:
                    return coef, q
                
            return coef, q
        except:
            print('Trying again with a different shift')
            log_shift += 1.0
            pass

# Shift sigma with -0.01 to ensure we find an eigenvalue to the left of sigma
# because we are looking for Hopf eigenvalues that cross the imaginary axis from
# the right to the left. This corresponds to the Largest Magnitude shifted eigenvale
# 1 / (lambda - shift + 0.01).
def shiftInvertArnoldiScipy(A, sigma, v0, tolerance):
    M = v0.size
    _A_complex = lambda w : A(w).astype(dtype=np.complex128)
    A_complex = slg.LinearOperator(shape=(M,M), matvec=_A_complex)
    eig_vals, eig_vecs = slg.eigs(A_complex, k=1, sigma=sigma - 0.1, v0=v0, which='LM', return_eigenvectors=True, tol=tolerance)

    return eig_vals[0], eig_vecs[:,0]

def continueArnoldi(G_x, x_path, eps_path, sigma, q, tolerance):
    eigenvalues = [sigma]
    eigenvectors = [q]

    n_points = x_path.shape[0]
    M = x_path.shape[1]
    for i in range(1, n_points):
        print('i =', i)
        x = x_path[i,:]
        eps = eps_path[i]

        A = slg.LinearOperator(shape=(M, M), matvec=lambda w: G_x(x, w, eps))
        sigma, q = shiftInvertArnoldiSimple(A, sigma, q, tolerance)

        eigenvalues.append(sigma)
        eigenvectors.append(q)

    return np.array(eigenvalues, dtype=np.complex128), np.array(eigenvectors)

def continueArnoldiScipy(G_x, x_path, eps_path, sigma, q, tolerance):
    eigenvalues = [sigma]
    eigenvectors = [q]

    n_points = x_path.shape[0]
    M = x_path.shape[1]
    for i in range(1, n_points):
        print('i =', i)
        x = x_path[i,:]
        eps = eps_path[i]

        A = slg.LinearOperator(shape=(M, M), matvec=lambda w: G_x(x, w, eps))
        sigma, q = shiftInvertArnoldiScipy(A, sigma, q, tolerance)

        eigenvalues.append(sigma)
        eigenvectors.append(q)

    return np.array(eigenvalues, dtype=np.complex128), np.array(eigenvectors)