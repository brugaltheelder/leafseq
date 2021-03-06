__author__ = 's162195'

from itertools import product

from cgModel import *
from erfMidpointModel import *
from stratGreedy import *


def runclingreed(data,trueFlu, Lestimate, desiredK, outputName = 'out.mat', pTag = '', closePlots = False):
    '''Runs the clinical greedy model, returns objective, seed vector, MUs, number of apertures

    :return a,b,c,d: Objective function, seed vector, total MUs, number of apertures used
    '''

    mod = stratGreedy(trueFlu, Lestimate, data, K=desiredK, plotTag=pTag)
    erfReturn, obj, kout = mod.runClinicalReturnErfSeed()
    mod.plotStrat()
    mod.output(outputName)
    if closePlots:
        mod.closePlots()
    return obj, erfReturn, np.sum(mod.y), kout


def runerf(data, initParams, RTplot=False, RTplotsaving=False, startVec=None, finalShow=False, trueFlu=None, outputName='out.mat', closePlots = False, dispFreq = False, pTag = '', plotSeed = False):
    """Runs Explicit model, plots solution, outputs matlab file, closes plots, returns objective and MUs
    
    :return a,b,c,d: Objective function, total MUs
    """

    mod = erfMidModel(data, realTimePlotting=RTplot, realTimePlotSaving=RTplotsaving,
                      initializationStringAndParams=initParams, startingSolutionVector=startVec,
                      trueFluenceVector=trueFlu, displayFreq=dispFreq, plotTag=pTag, plotSeed=plotSeed)

    initialObj, grad = mod.objGradEval(mod.varArray)
    seedMUs = np.sum(mod.varArray[0:mod.K])
    mod.solve()
    mod.plotSolution(finalShow=finalShow)
    mod.output(outputName)
    if closePlots:
        mod.closePlots()

    return mod.obj, np.sum(mod.finalX[0:mod.K]), initialObj, seedMUs


def runcg(data, RTplot=False, RTplotsaving=False, trueF=None, finalShow=False, outputName='out.mat', closePlots=False,
          dispFreq=False, pTag='', simpG=False):
    """Runs CG model, plots solution, outputs matlab file, closes plots, returns objective and MUs and original obj

    :return a,b,c,d: Objective function, seed vector, total MUs, objective from unit calc
    """

    mod = cgSolver(data, realTimePlotting=RTplot, realTimePlotSaving=RTplotsaving, trueFluenceVector=trueF,
                   displayFreq=dispFreq, plotTag=pTag, simpleG=simpG)
    mod.solve(mod.K)
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
    '''Class for building a parameter set through which to iterate when doing batch runs'''
    def __init__(self):
        '''initialize lists'''
        self.paramValues = []
        self.paramName = []
        self.paramRanges = []
        self.obj = []
        self.paramList = []
        self.runTimes = []
        self.totalMUs = []
        self.realKs = []
        self.runTags = []

    def addParam(self, pName, pValues):
        """Adds a param and its values"""
        self.paramName.append(pName)
        self.paramValues.append(pValues)
        self.paramRanges.append(range(len(pValues)))

    def genCombination(self):
        """gets combination"""
        self.combination = product(*self.paramRanges)

    def addRealK(self,k):
        """appends obj value to output list"""
        self.realKs.append(k)

    def addObjList(self,obj):
        """appends obj value list to output list"""
        self.obj.append(obj)

    def addRuntimeList(self, rTimes):
        """appends runtime value list to output list"""
        self.runTimes.append(rTimes)

    def addTagList(self, tags):
        self.runTags.append(tags)

    def addMUList(self, MUs):
        """appends MU value list to output list"""
        self.totalMUs.append(MUs)

    def getAndSaveParams(self, *indices):
        params = [self.paramValues[i][indices[i]] for i in range(len(indices))]
        self.paramList.append(params)
        return [self.paramValues[i][indices[i]] for i in range(len(indices))]

    def getFilename(self, *indices):
        '''builds a filename for a particular parameter set'''
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
                              'totalMU': self.totalMUs, 'realKs': self.realKs, 'runTags': self.runTags})
