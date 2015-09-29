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
import pandas



# #data testing for CG + Explicit models
res = 0.01
numAper = 30
sigma = 0.075
width = 10.
b = 4 / math.sqrt(width)
c = 3.
alphas = np.ones(int(width / res + 1))

dat = dataObj([c, b], res, numAper, sigma, width, alphas, [2. / width, width / 2., 1. / width], 'TestRun')


# Set initial input data
res = 0.01
width = 10.
sigma = 0.075
alphas = np.ones(int(width / res + 1))
minAperWidth = 1./width
maxAperWidth = width/2.
minAperEdgeGap = 0.1/width

# Generate data vectors
kParam = np.arange(5,45+1, 10)
cParam = np.arange(2,5+1, 3)
bParam = np.arange(1./width, 13./width+1./width, 3./width)

print kParam, cParam, bParam
params = paramTesting()
params.addParam('maxAper',kParam.tolist())
params.addParam('sinOffset',cParam.tolist())
params.addParam('sinScalar',bParam.tolist())
params.genCombination()



for kInd,cInd,bInd in params.combination:
    kP, cP, bP = params.getAndSaveParams(kInd, cInd, bInd)
    runName = params.getFilename(kInd, cInd, bInd)
    dat = dataObj([cP, bP], res, kP, sigma, width, alphas,[minAperWidth, maxAperWidth, minAperEdgeGap], runName)
    iterObj = []
    iterRunTime = []

    # run naive techniques
    #unifcent
    start = time.time()
    iterObj.append(runerf(dat, ['unifcent', 3], RTplot=False, finalShow=False, outputName='unifcent_out.mat', closePlots = True, pTag='unifcent'))
    iterRunTime.append(time.time() - start)

    #unifwidth
    start = time.time()
    iterObj.append(runerf(dat, ['unifwidth', 3], RTplot=False, finalShow=False, outputName='unifwidth_out.mat', closePlots = True, pTag='unifwidth'))
    iterRunTime.append(time.time() - start)

    #unifmixed
    start = time.time()
    iterObj.append(runerf(dat, ['unifmixed', 3], RTplot=False, finalShow=False, outputName='unifmixed_out.mat', closePlots = True, pTag='unifmixed'))
    iterRunTime.append(time.time() - start)

    # run CG
    start = time.time()
    erfInputVec, obj = runcg(dat,RTplot=False, finalShow=False, outputName= 'cg_out.mat', closePlots = True, pTag='cg')
    iterObj.append(obj)
    iterRunTime.append(time.time()-start)


    # run explicit based on cg
    start = time.time()
    iterObj.append(runerf(dat, ['unifmixed', 3], RTplot=False, finalShow=False, outputName='cgSeeded_out.mat', closePlots = True, startVec=erfInputVec, pTag='cgSeeded'))
    iterRunTime.append(time.time() - start)

    params.addObjList(iterObj)
    params.addRuntimeList(iterRunTime)

    if len(params.obj)%10==0:
        print pandas.DataFrame(params.obj, [i for i in range(len(params.obj))], ['unifcent','unifwidth','unifmixed', 'cg','cgSeeded'])
        print pandas.DataFrame(params.runTimes, [i for i in range(len(params.obj))], ['unifcent','unifwidth','unifmixed', 'cg','cgSeeded'])

params.writeRuns('k_b_c_runsout.mat')