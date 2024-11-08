import numpy as np
import scipy.linalg as lg
import scipy.sparse.linalg as slg

import warnings
warnings.filterwarnings("error")

def rayleigh(A, q):
    return np.vdot(q, A(q)) / np.vdot(q, q)

def shiftInvertArnoldiSimple(A, sigma, v0, tolerance, report_tolerance=1.e-3):
    M = v0.size

    for log_shift in range(10):
        shift = sigma - (log_shift + 1.0) * 0.1
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

def continueArnoldi(G_x, x_path, eps_path, sigma, q, tolerance):
    eigenvalues = [sigma]
    eigenvectors = [q]

    n_points = x_path.shape[0]
    M = x_path.shape[1]
    for i in range(1, n_points):
        x = x_path[i,:]
        eps = eps_path[i]

        A = slg.LinearOperator(shape=(M, M), matvec=lambda w: G_x(x, w, eps))
        sigma, q = shiftInvertArnoldiSimple(A, sigma, q, tolerance)

        eigenvalues.append(sigma)
        eigenvectors.append(q)

    return eigenvalues, np.array(eigenvectors)