__author__ = 'troy'

try:
    import mkl

    have_mkl = True
    print("Running with MKL Acceleration")
except ImportError:
    have_mkl = False
    print("Running with normal backends")

import matplotlib

matplotlib.use('Agg')

from runFunctions import *
from fluenceFunctionBuilders import *
import numpy as np
import time
import pandas

# #data testing for CG + Explicit models
res = 0.01
sigma = 0.075
width = 10.
b = 4 / math.sqrt(width)
c = 3.
alphas = np.ones(int(width / res + 1))


printEveryThisManyIterations = 10


# Set initial input data
res = 0.01
width = 10.
sigma = 0.075
alphas = np.ones(int(width / res + 1))
minAperWidth = 0.
maxAperWidth = width
minAperEdgeGap = 0.
# minAperWidth = 1./width
# maxAperWidth = width/2.
# minAperEdgeGap = 0.1/width



# Generate data vectors - sin
outfilename = 'sinout.mat'
directory = 'singlesin'
kParam = np.arange(10, 40 + 1, 5)
cParam = np.arange(2,5+1, 3)
bParam = np.arange(1./width, 13./width+1./width, 3./width)
print kParam, cParam, bParam
params = paramTesting()
params.addParam('maxAper',kParam.tolist())
params.addParam('sinOffset',cParam.tolist())
params.addParam('sinScalar',bParam.tolist())
params.genCombination()


# Generate data vectors - double sin
# outfilename = 'doublesinout.mat'
# directory = 'doublesin'
# kParam = np.arange(10, 40 + 1, 5)
# cParam = np.arange(2.,6.+1., 2.)
# bParam = np.arange(1./5., 1.+1/6., 2./5.)
# params = paramTesting()
# params.addParam('maxAper',kParam.tolist())
# params.addParam('sin2scale',cParam.tolist())
# params.addParam('sin2multiplier',bParam.tolist())
# params.genCombination()
# print kParam, cParam, bParam


# Generate data vectors - sum of erfs
# outfilename = 'erfsout.mat'
# directory = 'erfs'
# kParam = np.arange(3, 15 + 1, 3) 
# cParam = np.arange(0.0,1.+0.1, 0.5)
# bParam = np.arange(0.0,1.+0.1, 0.5)
# params = paramTesting()
# params.addParam('nAper',kParam.tolist())
# params.addParam('centerScalar',cParam.tolist())
# params.addParam('widthScalar',bParam.tolist())
# params.genCombination()
# print kParam, cParam, bParam


# Generate data vectors - random step
# # outfilename, order, directory = 'stepFunctionOutOrder0.mat', 0, 'stepOrder0' 
# outfilename, order, directory = 'stepFunctionOutOrder2.mat', 2, 'stepOrder2'
# kParam = np.arange(10, 40 + 1, 5)
# cParam = np.arange(5, 15 + 1, 5)
# bParam = np.arange(3., 6 + 1, 1)
# params = paramTesting()
# params.addParam('maxAper', kParam.tolist())
# params.addParam('numBins', cParam.tolist())
# params.addParam('minRange', bParam.tolist())
# params.genCombination()
# print kParam, cParam, bParam



# todo do a run batch with width bounds
fGetter = functionGetter(width, res)

for kInd, cInd, bInd in params.combination:

    # doublesine
    # kP, cP, bP = params.getAndSaveParams(kInd, cInd, bInd)
    # runName = params.getFilename(kInd, cInd, bInd)
    # dat = dataObj([0, 0], res, kP, sigma, width, alphas,[minAperWidth, maxAperWidth, minAperEdgeGap], runName, directory)
    # fVec = fGetter.doubleSinfunction(1.,1.,bP, cP,4.)


    # single sine
    kP, cP, bP = params.getAndSaveParams(kInd, cInd, bInd)
    runName = params.getFilename(kInd, cInd, bInd)
    dat = dataObj([cP, bP], res, kP, sigma, width, alphas,[minAperWidth, maxAperWidth, minAperEdgeGap], runName, directory)
    fVec = fGetter.sinFunction(cP,bP)

    # random step
    # kP, cP, bP = params.getAndSaveParams(kInd, cInd, bInd)
    # runName = params.getFilename(kInd, cInd, bInd)
    # dat = dataObj([0, 0], res, kP, sigma, width, alphas, [minAperWidth, maxAperWidth, minAperEdgeGap], runName, directory)
    # fVec = fGetter.unitStep(cP, bP, 7., order=order)

    #erf functions
    # kP, cP, bP = params.getAndSaveParams(kInd, cInd, bInd)
    # runName = params.getFilename(kInd, cInd, bInd)
    # dat = dataObj([0, 0], res, kP, sigma, width, alphas, [minAperWidth, maxAperWidth, minAperEdgeGap], runName, directory)
    # fVec = fGetter.erfSumRand(kP, cP, bP, sigma, width)    


    iterObj = []
    iterRunTime = []
    iterMUs = []

    # run naive techniques
    # random
    start = time.time()
    obj, MUs = runerf(dat, ['random', 3], RTplot=False, finalShow=False, outputName='random_out.mat',
                      closePlots=True, pTag='random', trueFlu=fVec, plotSeed=True)
    iterObj.append(obj)
    iterMUs.append(MUs)
    iterRunTime.append(time.time() - start)

    # sliding window
    start = time.time()
    obj, MUs = runerf(dat, ['slidingwindow', 3], RTplot=False, finalShow=False, outputName='slidingwindow_out.mat',
                      closePlots=True, pTag='slidingwindow', trueFlu=fVec, plotSeed=True)
    iterObj.append(obj)
    iterMUs.append(MUs)
    iterRunTime.append(time.time() - start)

    # centered
    start = time.time()
    obj, MUs = runerf(dat, ['centered', 3], RTplot=False, finalShow=False, outputName='centered_out.mat',
                      closePlots=True, pTag='centered', trueFlu=fVec, plotSeed=True)
    iterObj.append(obj)
    iterMUs.append(MUs)
    iterRunTime.append(time.time() - start)

    # peaks
    start = time.time()
    obj, MUs = runerf(dat, ['peaks', 3], RTplot=False, finalShow=False, outputName='peaks_out.mat',
                      closePlots=True, pTag='peaks', trueFlu=fVec, plotSeed=True)
    iterObj.append(obj)
    iterMUs.append(MUs)
    iterRunTime.append(time.time() - start)

    # run CG
    start = time.time()
    obj, erfInputVec, MUs, otherObj = runcg(dat, RTplot=False, finalShow=False, outputName='cgseed_out.mat',
                                            closePlots=True, pTag='cgseed', trueF=fVec)
    iterMUs.append(MUs)
    iterObj.append(obj)
    iterRunTime.append(time.time() - start)


    # run explicit based on cg
    start = time.time()
    obj, MUs = runerf(dat, ['unifmixed', 3], RTplot=False, finalShow=False, outputName='cg_out.mat',
                      closePlots=True, startVec=erfInputVec, pTag='cg', trueFlu=fVec, plotSeed=True)
    iterObj.append(obj)
    iterMUs.append(MUs)
    iterRunTime.append(time.time() - start)

    # run CG with simple objective
    start = time.time()
    obj, erfInputVec, MUs, otherObj = runcg(dat, RTplot=False, finalShow=False, outputName='cgsimpseed_out.mat',
                                            closePlots=True, pTag='cgsimpseed', simpG=True, trueF=fVec)
    iterMUs.append(MUs)
    iterObj.append(obj)
    iterRunTime.append(time.time() - start)

    # run explicit based on cg with simple objective
    # run explicit based on cg
    start = time.time()
    obj, MUs = runerf(dat, ['unifmixed', 3], RTplot=False, finalShow=False, outputName='cgsimp_out.mat',
                      closePlots=True, startVec=erfInputVec, pTag='cgsimp', trueFlu=fVec, plotSeed=True)
    iterObj.append(obj)
    iterMUs.append(MUs)
    iterRunTime.append(time.time() - start)

    params.addObjList(iterObj)
    params.addRuntimeList(iterRunTime)
    params.addMUList(iterMUs)

    if len(params.obj) % printEveryThisManyIterations == 0:
        print 'Objective Function Values'
        print pandas.DataFrame(params.obj, [i for i in range(len(params.obj))],
                               ['random','slidingwindow', 'centered', 'peaks', 'cg', 'cgSeeded', 'cgSimp', 'cgSimpSeeded'])
        #print 'Run Times'
        #print pandas.DataFrame(params.runTimes, [i for i in range(len(params.obj))],['random','slidingwindow', 'centered', 'peaks', 'cg', 'cgSeeded', 'cgSimp', 'cgSimpSeeded'])


params.writeRuns(directory + "/" + outfilename)
