import sys
sys.path.append('../')

from Arnoldi import *

import numpy.random as rd

def test1():
    _A = np.array([[2,0,0],[0,2,1], [0,1,2]])
    A = lambda w: np.dot(_A, w)
    sigma = 0.9

    rng = rd.RandomState()
    v0 = rng.normal(0.0, 1.0, 3)
    lam, v = shiftInvertArnoldi(A, sigma, v0 / lg.norm(v0), 1.e-12)

    print(lam, v)

if __name__ == '__main__':
    test1()