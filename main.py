__author__ = 's162195'

from erfMidpointModel import *
from cgModel import *
from dataObj import *
import math


res = 0.01
numAper = 12
sigma = 0.075
width = 10.
b = 2 / math.sqrt(width)
c = 3.
alphas = np.ones(int(width / res + 1))




def runerf(data, initParams,RTplot = True, RTplotsaving = False):
    mod = erfMidModel(dat, realTimePlotting=True, realTimePlotSaving=False, initializationStringAndParams=['unifmixed', 2])
    mod.solve()
    mod.plotSolution()
    mod.output('testout.mat')

def runcg(data, RTplot = False, RTplotsaving = False,  trueF = None):
    mod = cgSolver(data)
    mod.solve(data.numAper)




dat = dataObj([c, b], res, numAper, sigma, width, alphas, [2. / width, width / 2., 1. / width])

runcg(dat)