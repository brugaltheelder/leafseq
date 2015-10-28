__author__ = 's162195'


try:
    import mkl
    have_mkl = True
    print("Running with MKL Acceleration")
except ImportError:
    have_mkl = False
    print("Running with normal backends")

# import matplotlib
# matplotlib.use('Agg')
import time

import numpy as np

from runFunctions import *
from fluenceFunctionBuilders import *
from stratGreedy import *


# #data testing for CG + Explicit models
res = 0.01
numAper = 3
sigma = 0.075
width = 10.
b = 4 / math.sqrt(width)
c = 3.
alphas = np.ones(int(width / res + 1))

directory = 'testout'


# data testing for 2-aper ERF objective
# res = 0.01
# numAper = 2
# sigma = 0.075
# width = 10.
# b = 2 / math.sqrt(width)
# c = 3.
# alphas = np.ones(int(width / res + 1))



#dat = dataObj([c, b], res, numAper, sigma, width, alphas, [0, width / 2., 0], 'TestRun')
#dat = dataObj([c, b], res, numAper, sigma, width, alphas, [2. / width, width / 2., 1. / width], 'TestRun')
dat = dataObj([c, b], res, numAper, sigma, width, alphas, [0., width , 0.], 'TestRun', directory)

fGetter = functionGetter(width, res)
y = np.array([1,1])
m = np.array([width/2, width/2])
a = np.array([width/3, width/6])




# erfInputVec = np.zeros(3*dat.numAper)
# erfInputVec = np.copy(runcg(dat, RTplot=False))
# runerf(dat, ['unifmixed', 2], RTplot=False, startVec=erfInputVec, finalShow=True)

#fErfVec = fGetter.erfSum(y,m,a,sigma, truncate=0.0)
#fErfVec = fGetter.sinFunction(c,b)
fErfVec = fGetter.unitStep(10, 4, 7, 2)
#fErfVec = fGetter.doubleSinfunction(1., 1., 1. / 3., 4., 4)
#fErfVec = fGetter.erfSumRand(numAper, 0.5, 0.5, sigma, width)


stratifiedGreedy = stratGreedy(fErfVec, 10, width)
y,m,a,k = stratifiedGreedy.runStratGreedy(10)
stratifiedGreedy.plotStrat(y, m, a)


exit()
fGetter.functionPlotter(fErfVec,1,1,1, color = 'r', blockVar=False)
#erfInputVec = np.zeros(3*dat.numAper)
# obj, erfInputVec = runcg(dat, RTplot=False, trueF=fErfVec)
# print runcg(dat, RTplot=False, trueF=fErfVec)
obj, erfInputVec, mus, poorobj = runcg(dat, RTplot=False, simpG=True, trueF=fErfVec)
print obj, poorobj, mus


start = time.time()
# print runerf(dat, ['unifcent', 2], RTplot=False, finalShow=True, startVec=erfInputVec)
obj,Mus = runerf(dat, ['random', 2], RTplot=False, finalShow=True, startVec=erfInputVec, trueFlu=fErfVec, plotSeed=True)
#obj,Mus = runerf(dat, ['peaks', 2], RTplot=False, finalShow=True, trueFlu=fErfVec, plotSeed = True)
print obj, Mus
print 'finished in {} seconds'.format(str(time.time()-start))
#runerf(dat, ['unifcent', 2], RTplot=True, finalShow=True, trueFlu=fErfVec )



#runerf(dat, ['unifmixed', 3], RTplot=False, finalShow=True )

