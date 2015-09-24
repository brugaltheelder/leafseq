__author__ = 's162195'

import numpy as np
import scipy.optimize as spo
import scipy.special as sps
import scipy.sparse as spsparse
from scipy import io
import matplotlib.pyplot as plt
from dataObj import *


class rmpSolution():
    def __init__(self, numApproxPoints, alphas, f):

        #initialize objective elements (f, alphas)
        self.f = np.copy(f)
        self.alphas = np.copy(alphas)
        # initialize y vector (intensities)
        self.y = np.array([])
        self.nApprox = numApproxPoints
        # initiailze D sparse matrix of dimension (0,num points)
        self.D = spsparse.csc_matrix((self.nApprox,0))
        # initialize solution D vector
        self.g = np.zeros(self.nApprox)
        self.lVec = []
        self.rVec = []


    def calcObjGrad(self,y):
        self.g = self.D.dot(y)
        diff = self.f-self.g
        return self.evalObj(y, diff=diff), self.evalGrad(y, diff=diff)

    def evalGrad(self, y, diff=None):
        # if diff==None, then calc diff, otherwise return gradient
        if diff==None:
            self.g = self.D.dot(y)
            diff = self.f-self.g
        return -2 * self.alphas * diff

    def evalObj(self,y, diff=None):
        # if diff==None, then calc diff, otherwise return objective
        if diff==None:
            self.g = self.D.dot(y)
            diff = self.f-self.g
        return np.sum(self.alphas * (diff**2))


    def solvePP(self, y):
        # calc gradient

        # do search

        # get best row

        # add aperture to model

    def addAper(self, l,r):
        # calculate row of D, add to sparse matrix

        # generate y-value for y-vector

        # save l and r positions

        pass

    def solveRMP(self):
        pass
        # plug into scipy.optimize

        # save solution to self







class cgModel():
    def __init__(self, runData, realTimePlotting=False, realTimePlotSaving = False, startingSolutionVector = None, trueFluenceVector = None):

        #Initialize Data

        self.realTimePlotting, self.realTimePlotSaving = realTimePlotting, realTimePlotSaving
        self.K, self.width, self.resolution = runData.numAper, 1.0 * runData.width, 1.0 * runData.resolution  # num apers, width of row, spacing of approx
        self.sinGap, self.sigma, self.sinScalar = 1.0 * runData.objParams[0], 1.0 * runData.sigma, 1.0 * \
                                                  runData.objParams[
                                                      1]  # shift for sin obj, scacling factor for erf, sin scaling factor
        self.minAperWidth, self.maxAperWidth, self.aperCenterOffset = runData.minAperWidth, runData.maxAperWidth, runData.aperCenterOffset  # min/max bounds for aper width, aper center offset from edges
        self.objCalls = 0
        # Initialize values
        self.numApproxPoints = int(self.width / self.resolution + 1)  # number of approximation points
        self.alphas = np.copy(runData.alphas)  # copy over objective function weights

        # generate approximation points, including endpoints
        self.approxPoints = np.arange(0, self.width + self.resolution / 2, self.resolution)


        # check if there is a seed target fluence, if not, default to sin
        if trueFluenceVector != None and len(self.approxPoints) == len(trueFluenceVector):
            self.fTarget = trueFluenceVector
        else:
            self.fTarget = np.array(np.sin(self.sinScalar * self.approxPoints) + self.sinGap)


        # build solution object


        # seed initial fluence solution

    def solve(self):
        pass

        # build while loop for np.size(solution.y)
            # solve PP
                # get best aper
                # add best aper to model

            # solve RMP
                # execute the solve command
                # update the solution (part of the solve command)

    def outputSolution(self):
        pass

        # output solution

    def plotSolution(self):
        pass
        # plot target fluence

        # for each aperture, plot onto graph

        # plot sum of apertures


