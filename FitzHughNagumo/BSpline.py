import numpy as np
import numpy.linalg as lg

class ClampedCubicSpline:
    def __init__(self, x, f, x_left=0.0, x_right=1.0):
        self.x = np.copy(x)
        self.f = np.copy(f)
        self.n = len(self.x) - 1
        self.x[-1] = self.x[-1] + 1.e-12

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
        A[-2, 2*self.n] = 2.0 * (x_left - self.x[0])
        A[-2, 3*self.n] = 3.0 * (x_left - self.x[0])**2
        A[-1, 2*self.n-1] = 1.0
        A[-1, 3*self.n-1] = 2.0 * (x_right - self.x[self.n-1])
        A[-1, 4*self.n-1] = 3.0 * (x_right - self.x[self.n-1])**2

        # Right-hand side
        b = np.zeros(4 * self.n)
        b[0] = self.f[0]
        b[2*self.n - 1] = self.f[-1]
        for i in range(self.n - 1):
            b[2*i + 1] = self.f[i+1]
            b[2*i + 2] = self.f[i+1]

        # Solve for the coefficients y = [a, b, c, d]
        y = lg.solve(A, b)
        self.a = y[0:self.n]
        self.b = y[self.n:2*self.n]
        self.c = y[2*self.n:3*self.n]
        self.d = y[3*self.n:]

    def __call__(self, x):
        return self.evaluate(x)
    
    def evaluate(self, x):
        index = np.searchsorted(self.x, x, side='right') - 1
        print('index', index, x, self.x[index])
        return self.a[index] + self.b[index] * (x - self.x[index]) + self.c[index] * (x - self.x[index])**2 + self.d[index] * (x - self.x[index])**3
    
    def derivative(self, x):
        index = np.searchsorted(self.x, x, side='right') - 1
        return self.b[index] + 2.0 * self.c[index] * (x - self.x[index]) + 3.0 * self.d[index] * (x - self.x[index])**2

