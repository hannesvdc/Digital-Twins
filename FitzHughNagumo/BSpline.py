import numpy as np
import numpy.linalg as lg
import scipy.sparse as sp
import scipy.linalg as sclg

class ClampedCubicSpline:
    lu_exists = False
    lu_piv = None

    @staticmethod
    def createSystem(x, n, left_bc, right_bc):
        A = np.zeros((4 * n, 4 * n))

        # interpolation conditions
        for i in range(n):
            A[2*i,i] = 1.0 # Left point
            A[2*i+1, i] = 1.0 # Right point ->
            A[2*i+1, n + i] = x[i+1] - x[i]
            A[2*i+1, 2*n + i] = (x[i+1] - x[i])**2
            A[2*i+1, 3*n + i] = (x[i+1] - x[i])**3

        # Derivative continuity conditions
        for i in range(n-1):
            A[2*n+i, n + i] = 1.0
            A[2*n+i, 2*n + i] = 2.0 * (x[i+1] - x[i])
            A[2*n+i, 3*n + i] = 3.0 * (x[i+1] - x[i])**2
            A[2*n+i, n + i + 1] = -1.0

        # Second derivative continuity conditions
        for i in range(n-1):
            A[3*n+i-1, 2*n + i] = 2.0
            A[3*n+i-1, 3*n + i] = 6.0 * (x[i+1] - x[i])
            A[3*n+i-1, 2*n + i + 1] = -2.0

        # Clamped boundary conditions
        A[-2, n] = 1.0
        A[-2, 2*n] = 2.0 * (left_bc - x[0])
        A[-2, 3*n] = 3.0 * (left_bc - x[0])**2
        A[-1, 2*n-1] = 1.0
        A[-1, 3*n-1] = 2.0 * (right_bc - x[n-1])
        A[-1, 4*n-1] = 3.0 * (right_bc - x[n-1])**2

        ClampedCubicSpline.lu_piv = sclg.lu_factor(A)
        ClampedCubicSpline.lu_exists = True

    def __init__(self, x, f, left_bc=0.0, right_bc=1.0, solver='direct'):
        self.x = np.copy(x)
        self.f = np.copy(f)
        self.n = len(self.x) - 1
        self.x[-1] = self.x[-1] + 1.e-12

        if solver == 'lu_direct' and ClampedCubicSpline.lu_exists is False:
            ClampedCubicSpline.createSystem(self.x, self.n, left_bc, right_bc)
        if solver != 'lu_direct':
            A = np.zeros((4 * self.n, 4 * self.n))

            # interpolation conditions
            for i in range(self.n):
                A[2*i,i] = 1.0 # Left point
                A[2*i+1, i] = 1.0 # Right point ->
                A[2*i+1, self.n + i] = self.x[i+1] - self.x[i]
                A[2*i+1, 2*self.n + i] = (self.x[i+1] - self.x[i])**2
                A[2*i+1, 3*self.n + i] = (self.x[i+1] - self.x[i])**3

            # Derivative continuity conditions
            for i in range(self.n-1):
                A[2*self.n+i, self.n + i] = 1.0
                A[2*self.n+i, 2*self.n + i] = 2.0 * (self.x[i+1] - self.x[i])
                A[2*self.n+i, 3*self.n + i] = 3.0 * (self.x[i+1] - self.x[i])**2
                A[2*self.n+i, self.n + i + 1] = -1.0

            # Second derivative continuity conditions
            for i in range(self.n-1):
                A[3*self.n+i-1, 2*self.n + i] = 2.0
                A[3*self.n+i-1, 3*self.n + i] = 6.0 * (self.x[i+1] - self.x[i])
                A[3*self.n+i-1, 2*self.n + i + 1] = -2.0

            # Clamped boundary conditions
            A[-2, self.n] = 1.0
            A[-2, 2*self.n] = 2.0 * (left_bc - self.x[0])
            A[-2, 3*self.n] = 3.0 * (left_bc - self.x[0])**2
            A[-1, 2*self.n-1] = 1.0
            A[-1, 3*self.n-1] = 2.0 * (right_bc - self.x[self.n-1])
            A[-1, 4*self.n-1] = 3.0 * (right_bc - self.x[self.n-1])**2

        # Right-hand side
        b = np.zeros(4 * self.n)
        b[0] = self.f[0]
        b[2*self.n - 1] = self.f[-1]
        for i in range(self.n - 1):
            b[2*i + 1] = self.f[i+1]
            b[2*i + 2] = self.f[i+1]

        # Solve for the coefficients y = [a, b, c, d]
        if solver == 'direct':
            y = lg.solve(A, b) # Idea: Use Krylov method (GMRES, L-GMRES, ...)
        elif solver == 'krylov':
            A_csc = sp.csc_matrix(A)
            y = sp.linalg.spsolve(A_csc, b=b)
        elif solver == 'lu_direct':
            y = sclg.lu_solve(ClampedCubicSpline.lu_piv, b)
        self.a = y[0:self.n]
        self.b = y[self.n:2*self.n]
        self.c = y[2*self.n:3*self.n]
        self.d = y[3*self.n:]

    def __call__(self, x):
        return self.evaluate(x)
    
    def evaluate(self, x):
        val_array = np.zeros_like(x)
        for i in range(len(x)):
            index = np.searchsorted(self.x, x[i], side='right') - 1
            if index < 0:
                index = 0
            if index > len(self.a) - 1:
                index = len(self.a) - 1

            val_array[i] = self.a[index] + self.b[index] * (x[i] - self.x[index]) + self.c[index] * (x[i] - self.x[index])**2 + self.d[index] * (x[i] - self.x[index])**3

        return val_array
    
    def derivative(self, x):
        index = np.searchsorted(self.x, x, side='right') - 1
        if index < 0:
            index = 0
        if index > len(self.a)-1:
            index = len(self.a)-1
        
        return self.b[index]  + 2.0 * self.c[index] * (x - self.x[index]) + 3.0 * self.d[index] * (x - self.x[index])**2

