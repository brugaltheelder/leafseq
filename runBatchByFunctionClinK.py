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

np.random.seed(seed=1)

# #data testing for CG + Explicit models

plotSeedSwitch = True

printEveryThisManyIterations = 1
fullPrintEveryThisManyIterations = 10


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
# outfilename = 'sinout.mat'
# directory = 'singlesinClinK'
# kParam = np.arange(10, 40 + 1, 10)
# cParam = np.arange(2,5+1, 3)
# bParam = np.arange(1./width, 13./width+1./width, 3./width)
# print kParam, cParam, bParam
# params = paramTesting()
# params.addParam('maxAper',kParam.tolist())
# params.addParam('sinOffset',cParam.tolist())
# params.addParam('sinScalar',bParam.tolist())
# params.genCombination()


# Generate data vectors - double sin
# outfilename = 'doublesinout.mat'
# directory = 'doublesinClinK'
# kParam = np.arange(10, 40 + 1, 10)
# cParam = np.arange(2.,6.+1., 2.)
# bParam = np.arange(1./5., 1.+1/6., 2./5.)
# params = paramTesting()
# params.addParam('maxAper',kParam.tolist())
# params.addParam('sin2scale',cParam.tolist())
# params.addParam('sin2multiplier',bParam.tolist())
# params.genCombination()
# print kParam, cParam, bParam


# Generate data vectors - sum of erfs
outfilename = 'erfsout.mat'
directory = 'erfsClinK'
kParam = np.arange(10, 40 + 1, 10)
cParam = np.arange(0.0, 1. + 0.1, 0.5)
bParam = np.arange(0.0, 1. + 0.1, 0.5)
params = paramTesting()
params.addParam('nAper', kParam.tolist())
params.addParam('centerScalar', cParam.tolist())
params.addParam('widthScalar', bParam.tolist())
params.genCombination()
print kParam, cParam, bParam

# Generate data vectors - random step
# # outfilename, order, directory = 'stepFunctionOutOrder0.mat', 0, 'stepOrder0ClinK'
# outfilename, order, directory = 'stepFunctionOutOrder2.mat', 2, 'stepOrder2ClinK'
# kParam = np.arange(10, 40 + 1, 10)
# cParam = np.arange(10, 20 + 1, 5)
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




    # single sine
    # kP, cP, bP = params.getAndSaveParams(kInd, cInd, bInd)
    # runName = params.getFilename(kInd, cInd, bInd)
    # dat = dataObj([cP, bP], res, kP, sigma, width, alphas,[minAperWidth, maxAperWidth, minAperEdgeGap], runName, directory)
    # fVec = fGetter.sinFunction(cP,bP)

    # doublesine
    # kP, cP, bP = params.getAndSaveParams(kInd, cInd, bInd)
    # runName = params.getFilename(kInd, cInd, bInd)
    # dat = dataObj([0, 0], res, kP, sigma, width, alphas,[minAperWidth, maxAperWidth, minAperEdgeGap], runName, directory)
    # fVec = fGetter.doubleSinfunction(1.,1.,bP, cP,4.)

    #erf functions
    kP, cP, bP = params.getAndSaveParams(kInd, cInd, bInd)
    runName = params.getFilename(kInd, cInd, bInd)
    dat = dataObj([0, 0], res, kP, sigma, width, alphas, [minAperWidth, maxAperWidth, minAperEdgeGap], runName,
                  directory)
    fVec = fGetter.erfSumRand(kP, cP, bP, sigma, width)

    # random step
    # kP, cP, bP = params.getAndSaveParams(kInd, cInd, bInd)
    # runName = params.getFilename(kInd, cInd, bInd)
    # dat = dataObj([0, 0], res, kP, sigma, width, alphas, [minAperWidth, maxAperWidth, minAperEdgeGap], runName,
    #               directory)
    # fVec = fGetter.unitStep(cP, bP, 7., order=order)

 



    


    iterObj = []
    iterRunTime = []
    iterMUs = []
    iterTag = []

    # get the k goal, then set kReal 

    start = time.time()
    tag = 'Conv'
    obj, erfInputVec, MUs, kReal = runclingreed(dat, fVec, int(dat.numAper * 1.5), dat.numAper,
                                                outputName=tag + '_out.mat', closePlots=True, pTag=tag)
    iterObj.append(obj)
    iterMUs.append(MUs)
    iterTag.append(tag)
    iterRunTime.append(time.time()-start)

    dat.setKreal(kReal)
    params.addRealK(kReal)


    # run explicit based on cg
    start = time.time()
    tag = 'CLO_Conv'
    obj, MUs, seedObj, seedMUs = runerf(dat, [tag, 3], RTplot=False, finalShow=False, outputName=tag + '_out.mat',
                                        closePlots=True, startVec=erfInputVec, pTag=tag, trueFlu=fVec,
                                        plotSeed=plotSeedSwitch)
    iterObj.append(obj)
    iterMUs.append(MUs)
    iterTag.append(tag)
    iterRunTime.append(time.time() - start)



    # run naive techniques
    # random
    start = time.time()
    tag = 'CLO_Rand'
    obj, MUs, seedObj, seedMUs = runerf(dat, [tag, 3], RTplot=False, finalShow=False, outputName=tag + '_out.mat',
                                        closePlots=True, pTag=tag, trueFlu=fVec, plotSeed=plotSeedSwitch)
    iterObj.append(seedObj)
    iterTag.append(tag[4:])
    iterMUs.append(seedMUs)
    iterObj.append(obj)
    iterMUs.append(MUs)
    iterTag.append(tag)
    iterRunTime.append(time.time() - start)
    iterRunTime.append(time.time() - start)


    # sliding window
    start = time.time()
    tag = 'CLO_Unif'
    obj, MUs, seedObj, seedMUs = runerf(dat, [tag, 3], RTplot=False, finalShow=False, outputName=tag + '_out.mat',
                                        closePlots=True, pTag=tag, trueFlu=fVec, plotSeed=plotSeedSwitch)

    iterObj.append(seedObj)
    iterTag.append(tag[4:])
    iterMUs.append(seedMUs)
    iterObj.append(obj)
    iterMUs.append(MUs)
    iterTag.append(tag)
    iterRunTime.append(time.time() - start)
    iterRunTime.append(time.time() - start)

    # centered
    start = time.time()
    tag = 'CLO_Cent'
    obj, MUs, seedObj, seedMUs = runerf(dat, [tag, 3], RTplot=False, finalShow=False, outputName=tag + '_out.mat',
                                        closePlots=True, pTag=tag, trueFlu=fVec, plotSeed=plotSeedSwitch)
    iterObj.append(seedObj)
    iterTag.append(tag[4:])
    iterMUs.append(seedMUs)
    iterObj.append(obj)
    iterMUs.append(MUs)
    iterTag.append(tag)
    iterRunTime.append(time.time() - start)
    iterRunTime.append(time.time() - start)

    # peaks
    start = time.time()
    tag = 'CLO_Peak'
    obj, MUs, seedObj, seedMUs = runerf(dat, [tag, 3], RTplot=False, finalShow=False, outputName=tag + '_out.mat',
                                        closePlots=True, pTag=tag, trueFlu=fVec, plotSeed=plotSeedSwitch)
    iterObj.append(seedObj)
    iterTag.append(tag[4:])
    iterMUs.append(seedMUs)
    iterObj.append(obj)
    iterMUs.append(MUs)
    iterTag.append(tag)
    iterRunTime.append(time.time() - start)
    iterRunTime.append(time.time() - start)

    # run CG
    start = time.time()
    tag = 'DLO'
    obj, erfInputVec, MUs, otherObj = runcg(dat, RTplot=False, finalShow=False, outputName=tag + '_out.mat',
                                            closePlots=True, pTag=tag, trueF=fVec)
    iterMUs.append(MUs)
    iterObj.append(obj)
    iterTag.append(tag)
    iterRunTime.append(time.time() - start)


    # run explicit based on cg
    start = time.time()
    tag = 'CLO_DLO'
    obj, MUs, seedObj, seedMUs = runerf(dat, [tag, 3], RTplot=False, finalShow=False, outputName=tag + '_out.mat',
                                        closePlots=True, startVec=erfInputVec, pTag=tag, trueFlu=fVec,
                                        plotSeed=plotSeedSwitch)
    iterObj.append(obj)
    iterMUs.append(MUs)
    iterTag.append(tag)
    iterRunTime.append(time.time() - start)

    # run CG with simple objective
    start = time.time()
    tag = 'DLO-U'
    obj, erfInputVec, MUs, otherObj = runcg(dat, RTplot=False, finalShow=False, outputName=tag + '_out.mat',
                                            closePlots=True, pTag=tag, trueF=fVec, simpG=True)
    iterMUs.append(MUs)
    iterObj.append(obj)
    iterTag.append(tag)
    iterRunTime.append(time.time() - start)

    # run explicit based on cg with simple objective
    # run explicit based on cg
    start = time.time()
    tag = 'CLO_DLO-U'
    obj, MUs, seedObj, seedMUs = runerf(dat, [tag, 3], RTplot=False, finalShow=False, outputName=tag + '_out.mat',
                                        closePlots=True, startVec=erfInputVec, pTag=tag, trueFlu=fVec,
                                        plotSeed=plotSeedSwitch)
    iterObj.append(obj)
    iterMUs.append(MUs)
    iterTag.append(tag)
    iterRunTime.append(time.time() - start)

    params.addObjList(iterObj)
    params.addRuntimeList(iterRunTime)
    params.addMUList(iterMUs)
    params.addTagList(iterTag)

    if len(params.obj) % printEveryThisManyIterations == 0:
        print 'Iteration', len(params.obj)
        if len(params.obj) % fullPrintEveryThisManyIterations == 0:
            print 'Objective Function Values'
            print pandas.DataFrame(params.obj, [i for i in range(len(params.obj))],
                                   params.runTags[-1])
            # print 'Run Times'
            # print pandas.DataFrame(params.runTimes, [i for i in range(len(params.obj))],['random','slidingwindow', 'centered', 'peaks', 'cg', 'cgSeeded', 'cgSimp', 'cgSimpSeeded'])


params.writeRuns(directory + "/" + outfilename)
