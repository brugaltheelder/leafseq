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




def runerf(data, initParams,RTplot = True, RTplotsaving = False, startVec = None):
    mod = erfMidModel(dat, realTimePlotting=RTplot, realTimePlotSaving=RTplotsaving, initializationStringAndParams=initParams, startingSolutionVector=startVec)
    mod.solve()
    mod.plotSolution()
    mod.output('testout.mat')

def runcg(data, RTplot = False, RTplotsaving = False,  trueF = None):
    mod = cgSolver(data, realTimePlotting=RTplot, realTimePlotSaving=RTplotsaving, trueFluenceVector= trueF)
    mod.solve(data.numAper)
    mod.printSolution()
    return mod.getErfInput()





dat = dataObj([c, b], res, numAper, sigma, width, alphas, [2. / width, width / 2., 1. / width])

runerf(dat, ['unifmixed', 2], RTplot=True)
runcg(dat, RTplot=False)