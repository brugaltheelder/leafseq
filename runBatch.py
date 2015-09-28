__author__ = 'troy'


try:
    import mkl
    have_mkl = True
    print("Running with MKL Acceleration")
except ImportError:
    have_mkl = False
    print("Running with normal backends")


from runFunctions import *
from fluenceFunctionBuilders import *
import numpy as np
import time



# #data testing for CG + Explicit models
res = 0.01
numAper = 30
sigma = 0.075
width = 10.
b = 4 / math.sqrt(width)
c = 3.
alphas = np.ones(int(width / res + 1))

dat = dataObj([c, b], res, numAper, sigma, width, alphas, [2. / width, width / 2., 1. / width], 'TestRun')

# Generate data vectors
