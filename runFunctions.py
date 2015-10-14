__author__ = 's162195'

from erfMidpointModel import *
from cgModel import *
from dataObj import *
import math
from itertools import product


def runerf(data, initParams, RTplot=False, RTplotsaving=False, startVec=None, finalShow=False, trueFlu=None, outputName='out.mat', closePlots = False, dispFreq = False, pTag = ''):
    mod = erfMidModel(data, realTimePlotting=RTplot, realTimePlotSaving=RTplotsaving,
                      initializationStringAndParams=initParams, startingSolutionVector=startVec,
                      trueFluenceVector=trueFlu, displayFreq=dispFreq, plotTag=pTag)
    mod.solve()
    mod.plotSolution(finalShow=finalShow)
    mod.output(outputName)
    if closePlots:
        mod.closePlots()
    
    return mod.obj, np.sum(mod.finalX[0:mod.K])


def runcg(data, RTplot=False, RTplotsaving=False, trueF=None, finalShow=False, outputName='out.mat', closePlots=False,
          dispFreq=False, pTag='', simpG=False):
    mod = cgSolver(data, realTimePlotting=RTplot, realTimePlotSaving=RTplotsaving, trueFluenceVector=trueF,
                   displayFreq=dispFreq, plotTag=pTag, simpleG=simpG)
    mod.solve(data.numAper)
    mod.printSolution(finalShow=finalShow)
    mod.output(outputName)    
    if closePlots:
        mod.closePlots()
    #print 'bestObj', mod.obj
    obj = mod.obj
    if simpG:
        obj = mod.finalObjEval()
    return obj, mod.getErfInput(), np.sum(mod.y[0:mod.K]), mod.obj


class paramTesting:
    def __init__(self):
        self.paramValues = []
        self.paramName = []
        self.paramRanges = []
        self.obj = []
        self.paramList = []
        self.runTimes = []
        self.totalMUs = []

    def addParam(self, pName, pValues):
        self.paramName.append(pName)
        self.paramValues.append(pValues)
        self.paramRanges.append(range(len(pValues)))

    def genCombination(self):
        self.combination = product(*self.paramRanges)

    def addObjList(self,obj):
        self.obj.append(obj)

    def addRuntimeList(self, rTimes):
        self.runTimes.append(rTimes)

    def addMUList(self, MUs):
        self.totalMUs.append(MUs)

    def getAndSaveParams(self, *indices):
        params = [self.paramValues[i][indices[i]] for i in range(len(indices))]
        self.paramList.append(params)
        return [self.paramValues[i][indices[i]] for i in range(len(indices))]

    def getFilename(self, *indices):
        params = [self.paramValues[i][indices[i]] for i in range(len(indices))]
        filename = ''
        for p in range(len(params)):
            filename+= str(self.paramName[p]) + str(params[p]) + '_'
        filename = filename[:-1]
        return filename.translate(None,'.')

    def writeRuns(self, filename):
        import scipy.io as io
        io.savemat(filename, {'paramNames':self.paramName, 'paramRanges':self.paramRanges, 'paramValues':self.paramValues,
                              'objectives': self.obj, 'runTimes': self.runTimes, 'paramList': self.paramList,
                              'totalMU': self.totalMUs})
