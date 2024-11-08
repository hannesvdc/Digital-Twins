import numpy as np
import numpy.linalg as lg
import numpy.random as rd
import scipy.sparse.linalg as slg
import scipy.optimize as opt
import matplotlib.pyplot as plt

N = 200
L = 20.0
dx = L / N
dt = 0.001
T = 1.0

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

