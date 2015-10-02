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
minAperWidth = 0.
maxAperWidth = width
minAperEdgeGap = 0.
#minAperWidth = 1./width
#maxAperWidth = width/2.
#minAperEdgeGap = 0.1/width

# Generate data vectors
kParam = np.arange(5, 45 + 1, 10)
# kParam = np.arange(5,10+1, 10)
cParam = np.arange(2,5+1, 3)
bParam = np.arange(1./width, 13./width+1./width, 3./width)

print kParam, cParam, bParam
params = paramTesting()
params.addParam('maxAper',kParam.tolist())
params.addParam('sinOffset',cParam.tolist())
params.addParam('sinScalar',bParam.tolist())
params.genCombination()


# todo do a run batch with width bounds


for kInd,cInd,bInd in params.combination:
    kP, cP, bP = params.getAndSaveParams(kInd, cInd, bInd)
    runName = params.getFilename(kInd, cInd, bInd)
    dat = dataObj([cP, bP], res, kP, sigma, width, alphas,[minAperWidth, maxAperWidth, minAperEdgeGap], runName)
    iterObj = []
    iterRunTime = []
    iterMUs = []

    # run naive techniques
    #unifcent
    start = time.time()
    obj, MUs = runerf(dat, ['unifcent', 3], RTplot=False, finalShow=False, outputName='unifcent_out.mat',
                      closePlots=True, pTag='unifcent')
    iterObj.append(obj)
    iterMUs.append(MUs)
    iterRunTime.append(time.time() - start)

    #unifwidth
    start = time.time()
    obj, MUs = runerf(dat, ['unifwidth', 3], RTplot=False, finalShow=False, outputName='unifwidth_out.mat',
                      closePlots=True, pTag='unifwidth')
    iterObj.append(obj)
    iterMUs.append(MUs)
    iterRunTime.append(time.time() - start)

    #unifmixed
    start = time.time()
    obj, MUs = runerf(dat, ['unifmixed', 3], RTplot=False, finalShow=False, outputName='unifmixed_out.mat',
                      closePlots=True, pTag='unifmixed')
    iterObj.append(obj)
    iterMUs.append(MUs)
    iterRunTime.append(time.time() - start)

    # run CG
    start = time.time()
    obj, erfInputVec, MUs, otherObj = runcg(dat, RTplot=False, finalShow=False, outputName='cg_out.mat',
                                            closePlots=True, pTag='cg')
    iterMUs.append(MUs)
    iterObj.append(obj)
    iterRunTime.append(time.time()-start)


    # run explicit based on cg
    start = time.time()
    obj, MUs = runerf(dat, ['unifmixed', 3], RTplot=False, finalShow=False, outputName='cgSeeded_out.mat',
                      closePlots=True, startVec=erfInputVec, pTag='cgSeeded')
    iterObj.append(obj)
    iterMUs.append(MUs)
    iterRunTime.append(time.time() - start)

    # run CG with simple objective
    start = time.time()
    obj, erfInputVec, MUs, otherObj = runcg(dat, RTplot=False, finalShow=False, outputName='cgSimp_out.mat',
                                            closePlots=True, pTag='cgSimp', simpG=True)
    iterMUs.append(MUs)
    iterObj.append(obj)
    iterRunTime.append(time.time() - start)

    # run explicit based on cg with simple objective
    # run explicit based on cg
    start = time.time()
    obj, MUs = runerf(dat, ['unifmixed', 3], RTplot=False, finalShow=False, outputName='cgSimpSeeded_out.mat',
                      closePlots=True, startVec=erfInputVec, pTag='cgSimpSeeded')
    iterObj.append(obj)
    iterMUs.append(MUs)
    iterRunTime.append(time.time() - start)


    params.addObjList(iterObj)
    params.addRuntimeList(iterRunTime)
    params.addMUList(iterMUs)

    if len(params.obj)%10==0:
        print 'Objective Function Values'
        print pandas.DataFrame(params.obj, [i for i in range(len(params.obj))],
                               ['unifcent', 'unifwidth', 'unifmixed', 'cg', 'cgSeeded', 'cgSimp', 'cgSimpSeeded'])
        print 'Run Times'
        print pandas.DataFrame(params.runTimes, [i for i in range(len(params.obj))],
                               ['unifcent', 'unifwidth', 'unifmixed', 'cg', 'cgSeeded', 'cgSimp', 'cgSimpSeeded'])


# Do data analysis here with pandas...figure out what to do for data first



params.writeRuns('k_b_c_runsout.mat')
