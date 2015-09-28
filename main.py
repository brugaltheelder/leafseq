__author__ = 's162195'


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


# #data testing for CG + Explicit models
# res = 0.01
# numAper = 12
# sigma = 0.075
# width = 10.
# b = 2 / math.sqrt(width)
# c = 3.
# alphas = np.ones(int(width / res + 1))


# data testing for 2-aper ERF objective
res = 0.01
numAper = 2
sigma = 0.075
width = 10.
b = 2 / math.sqrt(width)
c = 3.
alphas = np.ones(int(width / res + 1))




#dat = dataObj([c, b], res, numAper, sigma, width, alphas, [0, width / 2., 0], 'TestRun')
dat = dataObj([c, b], res, numAper, sigma, width, alphas, [2. / width, width / 2., 1. / width], 'TestRun')

fGetter = functionGetter(width, res)
y = np.array([1,1])
m = np.array([width/2, width/2])
a = np.array([width/3, width/6])

# erfInputVec = np.zeros(3*dat.numAper)
# erfInputVec = np.copy(runcg(dat, RTplot=False))
# runerf(dat, ['unifmixed', 2], RTplot=False, startVec=erfInputVec, finalShow=True)

fErfVec = fGetter.erfSum(y,m,a,sigma, truncate=0.0)
fGetter.functionPlotter(fErfVec,1,1,1, color = 'r', blockVar=False)
erfInputVec = np.zeros(3*dat.numAper)
erfInputVec = np.copy(runcg(dat, RTplot=False, trueF=fErfVec))

start = time.time()
runerf(dat, ['unifcent', 2], RTplot=False, finalShow=False,startVec=erfInputVec, trueFlu=fErfVec )
print 'finished in {} seconds'.format(str(time.time()-start))
#runerf(dat, ['unifcent', 2], RTplot=True, finalShow=True, trueFlu=fErfVec )



#runerf(dat, ['unifmixed', 4], RTplot=False, finalShow=False )

