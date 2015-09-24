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




def runerf(data, initParams,RTplot = True, RTplotsaving = False, startVec = None, finalShow = False):
    mod = erfMidModel(dat, realTimePlotting=RTplot, realTimePlotSaving=RTplotsaving, initializationStringAndParams=initParams, startingSolutionVector=startVec)
    mod.solve()
    mod.plotSolution(finalShow = finalShow)
    mod.output('ERFout.mat')
    print 'bestObj', mod.obj

def runcg(data, RTplot = False, RTplotsaving = False,  trueF = None,  finalShow = False):
    mod = cgSolver(data, realTimePlotting=RTplot, realTimePlotSaving=RTplotsaving, trueFluenceVector= trueF)
    mod.solve(data.numAper)
    mod.printSolution(finalShow = finalShow)
    mod.output('CGout.mat')
    print 'bestObj', mod.obj
    return mod.getErfInput()





#dat = dataObj([c, b], res, numAper, sigma, width, alphas, [0, width / 2., 0], 'TestRun')
dat = dataObj([c, b], res, numAper, sigma, width, alphas, [2. / width, width / 2., 1. / width], 'TestRun')

erfInputVec = np.zeros(3*dat.numAper)
erfInputVec = np.copy(runcg(dat, RTplot=False))

runerf(dat, ['unifmixed', 2], RTplot=False, startVec=erfInputVec, finalShow=False)