import numpy as np
import numpy.linalg as lg
import numpy.random as rd
import scipy.sparse.linalg as slg
import scipy.optimize as opt
import matplotlib.pyplot as plt

from EulerTimestepper import psi, sigmoid
from Arnoldi import shiftInvertArnoldiSimple

N = 200
L = 20.0
dx = L / N
dt = 0.001
T = 1.0
params = {'delta': 4.0, 'a0': -0.03, 'a1': 2.0, 'eps': 0.0}

def G(x, eps):
    params['eps'] = eps
    return psi(x, T, dx, dt, params)

def dGdx_v(x, v, eps):
    rdiff = 1.e-8
    return (G(x + rdiff * v, eps) - G(x, eps)) / rdiff

def dGdeps(x, eps):
    rdiff = 1.e-8
    return (G(x, eps + rdiff) - G(x, eps)) / rdiff

# Calculates the tangent to the path at the current point as (Gx^{-1} G_eps, -1).
def computeTangent(Gx_v, G_eps, prev_tangent, M, tolerance):
    x0 = prev_tangent[0:M] / prev_tangent[M]
    A = slg.LinearOperator(matvec=Gx_v, shape=(M,M))

    _tangent = slg.gmres(A, G_eps, x0=x0, atol=tolerance)[0]
    tangent = np.append(_tangent, -1.0)
    tangent = tangent / lg.norm(tangent)

    if np.dot(tangent, prev_tangent) > 0:
        return tangent
    else:
        return -tangent

def numericalContinuation(x0, eps0, initial_tangent, M, max_steps, ds, ds_min, ds_max, tolerance):
    x = np.copy(x0)
    eps = eps0
    prev_tangent = np.copy(initial_tangent)

    x_path = [np.copy(x)]
    eps_path = [eps]
    print_str = 'Step {0:3d}:\t <u>: {1:4f}\t eps: {2:4f}\t ds: {3:6f}'.format(0, np.mean(x_path[0][0:N]), eps, ds)
    print(print_str)

    #eig_vals = [sigma]
    #q = q0 / np.sqrt(np.vdot(q0, q0))
    for n in range(1, max_steps+1):
		# Calculate the tangent to the curve at current point 
        Gx_v = lambda v: dGdx_v(x, v, eps)
        Geps = dGdeps(x, eps)
        tangent = computeTangent(Gx_v, Geps, prev_tangent, M, tolerance)

		# Create the extended system for corrector: z = (x, eps) = (u, v, eps)
        N_opt = lambda z: np.dot(tangent, z - np.append(x, eps)) + ds
        F = lambda z: np.append(G(z[0:M], z[M]), N_opt(z))

		# Our implementation uses adaptive timetepping
        while ds > ds_min:
			# Predictor: Extrapolation
            x_p = x + ds * tangent[0:M]
            eps_p = eps + ds * tangent[M]
            z_p = np.append(x_p, eps_p)

			# Corrector: Newton - Krylov
            try:
                z_new = opt.newton_krylov(F, z_p, f_tol=tolerance)
                x = z_new[0:M]
                eps = z_new[M]
                x_path.append(np.copy(x))
                eps_path.append(eps)

				# Updating the arclength step and tangent vector
                ds = min(1.2*ds, ds_max)
                prev_tangent = np.copy(tangent)

                break
            except:
                # Decrease arclength if the corrector fails.
                ds = max(0.5*ds, ds_min)
        else:
            print('Minimal Arclength Size is too large. Aborting.')
            return x_path, eps_path#, eig_vals
        
        # Calculate the eigenvalue of Gx_v with minimal real part
        #print('calculating eigenvalue')
        #A = lambda w: dGdx_v(x, w, eps)
        #sigma, q = shiftInvertArnoldiSimple(A, sigma, q, tolerance)
        #eig_vals.append(sigma)
        #print('sigma', sigma)
		
        print_str = 'Step {0:3d}:\t <u>: {1:4f}\t eps: {2:4f}\t ds: {3:6f}'.format(n, np.mean(x_path[-1][0:N]), eps, ds)
        print(print_str)

    return x_path, eps_path#, eig_vals

def calculateBifurcationDiagram():
    eps0 = 0.1
    x_array = np.linspace(0.0, L, N)
    u0 = sigmoid(x_array, 14.0, -1, 1.0, 2.0)
    v0 = sigmoid(x_array, 15, 0.0, 2.0, 0.1)
    x0 = np.concatenate((u0, v0))

    # Calculate a good initial condition x0 on the path
    M = 2 * N
    tolerance = 1.e-6
    F = lambda x: G(x, eps0)
    x0 = opt.newton_krylov(F, x0, rdiff=1.e-8, f_tol=tolerance)

    # Continuation Parameters
    max_steps = 2000
    ds_min = 1.e-6
    ds_max = 0.01
    ds = 0.001

    # Calculate the tangent to the path at the initial condition x0
    rng = rd.RandomState()
    random_tangent = rng.normal(0.0, 1.0, M+1)
    initial_tangent = computeTangent(lambda v: dGdx_v(x0, v, eps0), dGdeps(x0, eps0), random_tangent / lg.norm(random_tangent), M, tolerance)
    initial_tangent = initial_tangent / lg.norm(initial_tangent)

    # Do actual numerical continuation in both directions
    if initial_tangent[-1] < 0.0: # Decreasing eps
        print('Increasing eps first')
        sign = 1.0
    else:
        sign = -1.0
    x1_path, eps1_path = numericalContinuation(x0, eps0,  sign * initial_tangent, M, max_steps, ds, ds_min, ds_max, tolerance)
    x2_path, eps2_path = numericalContinuation(x0, eps0, -sign * initial_tangent, M, max_steps, ds, ds_min, ds_max, tolerance)

    # Store the full path
    x1_path = np.array(x1_path)
    x2_path = np.array(x2_path)
    eps1_path = np.array(eps1_path)
    eps2_path = np.array(eps2_path)
    directory = '/Users/hannesvdc/OneDrive - Johns Hopkins/Research_Data/Digital Twins/FitzhughNagumo/'
    np.save(directory + 'bf_diagram.npy', np.hstack((x1_path, eps1_path[:,np.newaxis], x2_path, eps2_path[:,np.newaxis])))

    # Plot both branches
    plot_x1_path = np.average(x1_path[:, 0:N], axis=1)
    plot_x2_path = np.average(x2_path[:, 0:N], axis=1)
    plt.plot(eps1_path, plot_x1_path, color='blue', label='Branch 1')
    plt.plot(eps2_path, plot_x2_path, color='red', label='Branch 2')
    plt.xlabel(r'$\varepsilon$')
    plt.ylabel(r'$<u>$')
    plt.show()

def calculateEigenvaluesArnoldi():
    M = 2 * N

    directory = '/Users/hannesvdc/OneDrive - Johns Hopkins/Research_Data/Digital Twins/FitzhughNagumo/'
    bf_data = np.load(directory + 'bf_diagram.npy')
    x1_data = bf_data[:,0:M]
    eps1_data = bf_data[:, M]
    x2_data = bf_data[:, M+1 : 2*M+1]
    eps2_data = bf_data[:, 2*M+1]

    # Calculate the initial eigenvalue to high precision to start the process
    x0 = x1_data[0,:]
    eps0 = eps1_data[0]
    A = slg.LinearOperator(shape=(M, M), matvec=lambda w: dGdx_v(x0, w, eps0))
    A_matrix = np.zeros((M, M))
    for k in range(M):
        A_matrix[:,k] = A(np.eye(M)[:,k])
    eig_vals, eig_vecs = lg.eig(A_matrix)

    # Select the smallest real eigenvalue and do Arnoldi along x1_path
    sigma = np.inf
    q = np.zeros(M)
    for index in range(len(eig_vals)):
        if np.abs(np.imag(eig_vals[index])) < 1.e-8 and np.real(eig_vals[index]) < sigma:
            sigma = eig_vals[index]
            q = eig_vecs[:,index]
    eig1_path = continueEigenvalues(x1_path, eps1_path, sigma, q)

    # Select the smallest non-real eigenvalue and do Arnoldi along x2_path
    sigma = np.inf
    q = np.zeros(M)
    for index in range(len(eig_vals)):
        if np.abs(np.imag(eig_vals[index])) > 1.e-8 and np.real(eig_vals[index]) < sigma:
            sigma = eig_vals[index]
            q = eig_vecs[:,index]
    eig2_path = continueEigenvalues(x2_path, eps2_path, sigma, q)

    # Plot the results
    plt.plot(np.linspace(0, eps1_data[-1], len(eps1_data)), np.real(eig1_path), color='blue', label='Branch 1')
    plt.xlabel('Continuation Step')
    plt.ylabel('Eigenvalue')
    plt.legend()
    
    plt.figure()
    plt.scatter(np.real(eig2_path), np.imag(eig2_path), color='red', label='Branch 2')
    plt.ylabel('Imaginary') 
    plt.xlabel('Real')
    plt.legend()
    plt.show()

def plotBifurcationDiagram():
    

    # Calculate eigenvalues on x1_path and x2_path
    plot_x1_path = []
    plot_x2_path = []
    correct_eig1_vals = []
    correct_eig2_vals = []
    for i in range(len(x1_path)):
        print('i = ', i)
        plot_x1_path.append(np.mean(x1_path[i][0:N]))
        plot_x2_path.append(np.mean(x2_path[i][0:N]))

        if i % 10 != 0:
            continue

        A1 = slg.LinearOperator(shape=(M, M), matvec=lambda w: dGdx_v(x1_path[i], w, eps1_path[i]))
        A1_matrix = np.zeros((M, M))
        for k in range(M):
            A1_matrix[:,k] = A1(np.eye(M)[:,k])
        eig_vals1 = lg.eigvals(A1_matrix)
        correct_eig1_vals.append(eig_vals1[np.argmin(np.real(eig_vals1))])

        A2 = slg.LinearOperator(shape=(M, M), matvec=lambda w: dGdx_v(x2_path[i], w, eps2_path[i]))
        A2_matrix = np.zeros((M, M))
        for k in range(M):
            A2_matrix[:,k] = A2(np.eye(M)[:,k])
        eig_vals2 = np.sort(lg.eigvals(A2_matrix)) # Sort in ascending order
        for j in range(len(eig_vals2)):
            if np.imag(eig_vals2[j]) != 0.0:
                correct_eig2_vals.append(eig_vals2[j])
                break

    # Plot both branches
    plt.plot(eps1_path, plot_x1_path, color='blue', label='Branch 1')
    plt.plot(eps2_path, plot_x2_path, color='red', label='Branch 2')
    plt.xlabel(r'$\varepsilon$')
    plt.ylabel(r'$<u>$')

    plt.figure()
    plt.plot(np.linspace(0, max_steps, len(eig_vals1)), np.real(eig_vals1), color='blue', label='Branch 1')
    plt.plot(np.linspace(0, max_steps, len(correct_eig1_vals)), np.real(correct_eig1_vals), color='black', label='Exact Eigenvalues')
    plt.xlabel('Continuation Step')
    plt.ylabel('Eigenvalue')
    plt.legend()
    
    plt.figure()
    plt.scatter(np.real(eig_vals2), np.abs(np.imag(eig_vals2)), color='red', label='Branch 2')
    plt.scatter(np.real(correct_eig2_vals), np.abs(np.imag(correct_eig2_vals)), color='black', label='Exact Eigenvalues')
    plt.ylabel('Imaginary') 
    plt.xlabel('Real')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    calculateBifurcationDiagram()