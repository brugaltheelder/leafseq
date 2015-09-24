__author__ = 's162195'

import numpy as np
import scipy.optimize as spo
import scipy.special as sps
import scipy.sparse as spsparse
from scipy import io
import matplotlib.pyplot as plt
from dataObj import *


class cgSolver():
    def __init__(self, runData, realTimePlotting=False, realTimePlotSaving=False, trueFluenceVector=None):
        # def __init__(self, numApproxPoints, alphas, f, sigma, approxPoints):
        self.realTimePlotting, self.realTimePlotSaving = realTimePlotting, realTimePlotSaving
        self.K, self.width, self.resolution = runData.numAper, 1.0 * runData.width, 1.0 * runData.resolution  # num apers, width of row, spacing of approx
        self.sinGap, self.sigma, self.sinScalar = 1.0 * runData.objParams[0], 1.0 * runData.sigma, 1.0 * \
                                                  runData.objParams[
                                                      1]  # shift for sin obj, scacling factor for erf, sin scaling factor
        self.objCalls = 0
        # Initialize values
        self.nApprox = int(self.width / self.resolution + 1)  # number of approximation points
        self.alphas = np.copy(runData.alphas)  # copy over objective function weights

        # generate approximation points, including endpoints
        self.approxPoints = np.arange(0, self.width + self.resolution / 2, self.resolution)


        # check if there is a seed target fluence, if not, default to sin
        if trueFluenceVector != None and len(self.approxPoints) == len(trueFluenceVector):
            self.f = trueFluenceVector
        else:
            self.f = np.array(np.sin(self.sinScalar * self.approxPoints) + self.sinGap)

        # initialize y vector (intensities)
        self.y = np.array([])

        # initiailze D sparse matrix of dimension (0,num points)
        self.D = spsparse.csc_matrix((self.nApprox, 0))
        # initialize solution D vector
        self.g = np.zeros(self.nApprox)
        self.lVec = []
        self.rVec = []
        self.Dtolerance = 0.0001
                # start ongoing graph if necessary
        if (self.realTimePlotting):
            plt.figure(0)
            self.figCounter = 0
            plt.ion()
            plt.show()


    def calcObjGrad(self, y):
        self.g = self.D.dot(y)
        diff = self.f - self.g
        return self.evalObj(y, diff=diff), self.D.transpose().dot(self.evalGrad(y, diff=diff))

    def evalGrad(self, y, diff=None):
        # if diff==None, then calc diff, otherwise return gradient
        if diff is None:
            if np.size(y) > 0:
                self.g = self.D.dot(y)
            else:
                self.g = np.zeros(self.nApprox)
            diff = self.f - self.g
        return -2. * self.alphas * diff

    def evalObj(self, y, diff=None):
        # if diff==None, then calc diff, otherwise return objective
        if diff is None:
            self.g = self.D.dot(y)
            diff = self.f - self.g
        return np.sum(self.alphas * (diff ** 2))

    def solvePP(self, y):
        # calc gradient
        grad = self.evalGrad(y)
        print grad


        # do search
        # helper variables
        maxSoFar, maxEndingHere, lE, rE = 0, 0, 0, 0
        for i in range(self.nApprox):
            maxEndingHere += grad[i]
            if maxEndingHere > 0:
                maxEndingHere, lE, rE = 0, i + 1, i + 1
            if maxSoFar > maxEndingHere:
                maxSoFar, rE = maxEndingHere, i + 1
                lBest, rBest = lE, rE
        # add aperture to model
        return self.addAper(lBest, rBest)  # rE is non-inclusive

    def solve(self, aperMax):
        self.aperMax = aperMax
        while np.size(self.y) < self.aperMax:
            lPos,rPos = self.solvePP(self.y)
            self.solveRMP()
            print 'Aperures {} of {} added, lPos: {}, rPos: {}, Obj: {}'.format(str(np.size(self.y)), str(self.aperMax), lPos, rPos,str(self.obj))

    def addAper(self, l, r):
        # calculate col of D, add to sparse matrix using error function...make midpoint then write out function...truncate at very small
        # calc Dcol along entire axis, then truncate
        Dcol = np.zeros(self.nApprox)
        midpoint = self.approxPoints[l] + 1.0 * (self.approxPoints[(r - 1)] - self.approxPoints[l]) / 2.
        idx = self.approxPoints <= midpoint
        Dcol[idx] += 1 + sps.erf((self.approxPoints[idx] - self.approxPoints[l]) / self.sigma)
        idx = self.approxPoints > midpoint
        Dcol[idx] += sps.erfc((self.approxPoints[idx] - self.approxPoints[(r - 1)]) / self.sigma)
        Dcol[Dcol < self.Dtolerance] = 0
        Dcol_sparse = spsparse.csc_matrix(Dcol).transpose()

        if np.size(self.y) > 0:
            self.D = spsparse.hstack([self.D, Dcol_sparse])
        else:
            self.D = Dcol_sparse.copy()

        # generate y-value for y-vector
        self.y = np.resize(self.y, (1, np.size(self.y) + 1))

        # save l and r positions
        self.lVec.append(self.approxPoints[l])
        self.rVec.append(self.approxPoints[r - 1])
        return self.approxPoints[l], self.approxPoints[r - 1]

    def solveRMP(self):

        self.res = spo.minimize(self.calcObjGrad, x0=self.y.copy(), method='L-BFGS-B', jac=True,
                                bounds=np.array([(0, None) for i in range(np.size(self.y))]),
                                options={'ftol': 1e-8, 'disp': False})
        self.y = self.res['x']
        self.obj = self.res['fun']


    def getErfInput(self):
        return None


    def printSolution(self, ongoingfig=False, intermediateY=None):
        # plot main function

        if ongoingfig:
            plt.figure(0)
            plt.clf()
            yVec = np.copy(intermediateY)
        else:
            plt.figure(0)
            plt.close()
            plt.figure(1)
            plt.ioff()
            yVec = np.copy(self.y)

        # Objective function evaluation and plotting

        plt.plot(self.approxPoints, self.f, 'r')

        # Sets coherent limits to plot
        plt.ylim(0, 1.2 * max(np.max(yVec), self.sinGap + 1))
        plt.xlim(0, self.width)

        # plots total sequenced fluence
        g = self.D.dot(yVec)
        plt.plot(self.approxPoints, g, 'g')

        # plots each individual aperture
        for k in range(np.size(yVec)):
            # plot left error function up to center
            plt.plot(self.approxPoints, yVec[k] * self.D.getcol(k).todense(), 'b')
            # updates figure
        if ongoingfig:
            plt.draw()
            if self.realTimePlotSaving:
                plt.savefig('iterPlotOut_' + str(self.figCounter) + '_.png')
                self.figCounter += 1
        else:
            plt.show()