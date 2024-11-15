import numpy as np
import numpy.linalg as lg

class ClampedCubicSpline:
    def __init__(self, x, f):
        super(ClampedCubicSpline).__init__(self)

        self.x = x
        self.f = f
        self.n = len(self.x) - 1

        A = np.zeros((4 * self.n, 4 * self.n))

        # interpolation conditions
        for i in range(self.n):
            A[2*i,0] = 1.0
            A[2*i+1, 0] = 1.0
            A[2*i+1, self.n] = self.x[i+1] - self.x[i]
            A[2*i+1, 2*self.n] = (self.x[i+1] - self.x[i])**2
            A[2*i+1, 3*self.n] = (self.x[i+1] - self.x[i])**3

        # Derivative continuity conditions
        for i in range(self.n-1):
            A[2*self.n+i, self.n + i] = 1.0
            A[2*self.n+i, 2*self.n + i] = 2.0 * (self.x[i+1] - self.x[i])
            A[2*self.n+i, 3*self.n + i] = 3.0 * (self.x[i+1] - self.x[i])**2
            A[2*self.n+i, self.n + i + 1] = -1.0

        # Second derivative continuity conditions
        for i in range(self.n-1):
            A[3*self.n+i, 2*self.n + i] = 2.0
            A[3*self.n+i, 3*self.n + i] = 6.0 * (self.x[i+1] - self.x[i])
            A[3*self.n+i, 2*self.n + i + 1] = -2.0

        # Clamped boundary conditions
        A[-2, self.n] = 1.0
        A[-1, 2*self.n-1] = 1.0
        A[-1, 3*self.n-1] = 2.0 * (self.x[-1] - self.x[-2])
        A[-1, 4*self.n-1] = 3.0 * (self.x[-1] - self.x[-2])**2

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
        index = np.searchsorted(self.x, x, side='left')
        return self.a[index] + self.b[index] * (x - self.x[index]) + self.c[index] * (x - self.x[index])**2 + self.d[index] * (x - self.x[index])**3
    
    def derivative(self, x):
        index = np.searchsorted(self.x, x, side='left')
        return self.b[index] + 2.0 * self.c[index] * (x - self.x[index]) + 3.0 * self.d[index] * (x - self.x[index])

