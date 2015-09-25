__author__ = 's162195'

from runFunctions import *
from fluenceFunctionBuilders import *
import numpy as np

#data testing for CG + Explicit models
res = 0.01
numAper = 12
sigma = 0.075
width = 10.
b = 2 / math.sqrt(width)
c = 3.
alphas = np.ones(int(width / res + 1))

# data testing for 2-aper ERF objective
# res = 0.01
# numAper = 2
# sigma = 0.075
# width = 10.
# b = 2 / math.sqrt(width)
# c = 3.
# alphas = np.ones(int(width / res + 1))



#dat = dataObj([c, b], res, numAper, sigma, width, alphas, [0, width / 2., 0], 'TestRun')
dat = dataObj([c, b], res, numAper, sigma, width, alphas, [2. / width, width / 2., 1. / width], 'TestRun')

fGetter = functionGetter(width, res)
y = np.array([1,1])
m = np.array([width/2, width/2])
a = np.array([width/3, width/6])

# erfInputVec = np.zeros(3*dat.numAper)
# erfInputVec = np.copy(runcg(dat, RTplot=False))
# runerf(dat, ['unifmixed', 2], RTplot=False, startVec=erfInputVec, finalShow=True)

fErfVec = fGetter.erfSum(y,m,a,sigma, truncate=0.01)
#erfInputVec = np.zeros(3*dat.numAper)
#erfInputVec = np.copy(runcg(dat, RTplot=False, trueF=fErfVec))

#runerf(dat, ['unifcent', 2], RTplot=False, finalShow=True,startVec=erfInputVec, trueFlu=fErfVec )
#runerf(dat, ['unifcent', 2], RTplot=True, finalShow=True, trueFlu=fErfVec )

runerf(dat, ['unifmixed', 4], RTplot=False, finalShow=True )
