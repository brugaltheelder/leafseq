__author__ = 's162195'

from erfMidpointModel import *
from cgModel import *
from dataObj import *
import math


def runerf(data, initParams, RTplot=False, RTplotsaving=False, startVec=None, finalShow=False, trueFlu=None):
    mod = erfMidModel(data, realTimePlotting=RTplot, realTimePlotSaving=RTplotsaving,
                      initializationStringAndParams=initParams, startingSolutionVector=startVec,
                      trueFluenceVector=trueFlu)
    mod.solve()
    mod.plotSolution(finalShow=finalShow)
    mod.output('ERFout.mat')
    print 'bestObj', mod.obj


def runcg(data, RTplot=False, RTplotsaving=False, trueF=None, finalShow=False):
    mod = cgSolver(data, realTimePlotting=RTplot, realTimePlotSaving=RTplotsaving, trueFluenceVector=trueF)
    mod.solve(data.numAper)
    mod.printSolution(finalShow=finalShow)
    mod.output('CGout.mat')
    print 'bestObj', mod.obj
    return mod.getErfInput()
