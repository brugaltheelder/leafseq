__author__ = 's162195'

from erfMidpointModel import *
import math

res = 0.01
numAper = 12
sigma = 0.075
width = 10.
b = 2 / math.sqrt(width)
c = 3.
alphas = np.ones(int(width / res + 1))

dat = dataObj([c, b], res, numAper, sigma, width, alphas, [2. / width, width / 2., 1. / width])

mod = erfMidModel(dat, realTimePlotting=True, realTimePlotSaving=False, initializationStringAndParams=['unifmixed', 2])
# mod = erfMidModel(dat)

# mod = erfMidModel('sin', 0.1, 30, 5, 3,0.1)
mod.solve()
mod.plotSolution()
mod.output('testout.mat')
