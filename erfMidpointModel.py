__author__ = 'Troy Long'

import numpy as np
import scipy.optimize as spo
import scipy.special as sps
from scipy import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dataObj import *


class erfMidModel:
    # class initialization
    def __init__(self, runData, realTimePlotting=False, realTimePlotSaving=False, startingSolutionVector=None,
                 trueFluenceVector=None, initializationStringAndParams=None, displayFreq = False, plotTag = ''):

        # Read in parameters
        assert (isinstance(runData, dataObj))
        self.realTimePlotting, self.realTimePlotSaving = realTimePlotting, realTimePlotSaving
        self.K, self.width, self.resolution = runData.numAper, 1.0 * runData.width, 1.0 * runData.resolution  # num apers, width of row, spacing of approx
        self.sinGap, self.sigma, self.sinScalar = 1.0 * runData.objParams[0], 1.0 * runData.sigma, 1.0 * \
                                                  runData.objParams[
                                                      1]  # shift for sin obj, scacling factor for erf, sin scaling factor
        self.minAperWidth, self.maxAperWidth, self.aperCenterOffset = runData.minAperWidth, runData.maxAperWidth, runData.aperCenterOffset  # min/max bounds for aper width, aper center offset from edges
        self.runTag = runData.runTag
        self.objCalls = 0
        self.displayFreq = displayFreq
        self.plotTag = plotTag
        # Initialize values
        self.numApproxPoints = int(self.width / self.resolution + 1)  # number of approximation points
        self.alphas = np.copy(runData.alphas)  # copy over objective function weights


        # generate approximation points, including endpoints
        self.approxPoints = np.arange(0, self.width + self.resolution / 2, self.resolution)
        # preallocate other often-used numpy arrays
        self.g = np.zeros(self.numApproxPoints)
        self.gUnweighted = np.zeros(self.numApproxPoints)
        self.diff = np.zeros(self.numApproxPoints)
        self.grad = np.zeros(3*self.K)

        # build solution vector bounds - intensity, center, width
        # todo build in upper bounds on fluence
        self.bounds = np.array(
            [(0, None) for i in range(self.K)] + [(0 + self.aperCenterOffset, self.width - self.aperCenterOffset) for i
                                                  in range(self.K)] + [(self.minAperWidth, self.maxAperWidth) for i in
                                                                       range(self.K)])

        # allocate solution vector - [intensities, centers, widths]
        self.varArray = np.zeros(3 * self.K)


        # check if there is a seed target fluence, if not, default to sin
        if trueFluenceVector is not None and len(self.approxPoints) == len(trueFluenceVector):
            self.fTarget = trueFluenceVector
        else:
            self.fTarget = np.array(np.sin(self.sinScalar * self.approxPoints) + self.sinGap)


        # initialize solution vector
        if startingSolutionVector is not None and len(self.varArray) == len(startingSolutionVector):
            self.varArray = np.copy(startingSolutionVector)
        elif initializationStringAndParams[0] == 'unifcent':
            self.initializeVarsUniformCenter()
        elif initializationStringAndParams[0] == 'unifwidth':
            self.initializeVarsUniformWidth()
        elif initializationStringAndParams[0] == 'unifmixed':
            self.initializeVarsUniformMixed(initializationStringAndParams[1])
        else:
            self.initializeVarsUniformMixed(2)


        # start ongoing graph if necessary
        if (self.realTimePlotting):
            plt.figure(0)
            self.figCounter = 0
            plt.ion()
            plt.show()

    # calculates derivative of erf
    def derf(self, x):
        return 2 / np.pi * np.exp(-1 * x * x)

    # generates initial apertures with uniformly spaced centers and similar widths
    def initializeVarsUniformCenter(self):
        # initialize intensities
        self.varArray[0:self.K] = np.ones(self.K) * self.sinGap / self.K
        # initialize centers
        self.varArray[self.K:2 * self.K] = np.arange(0 + 1. / 2. / (self.K + 1), self.width, 1. * self.width / (self.K))
        # initialize widths
        self.varArray[2 * self.K:3 * self.K] = self.width / 2 * np.ones(self.K) * np.random.rand(self.K)

    # generates initial apertures with similar centers and uniformly varied widths
    def initializeVarsUniformWidth(self):
        # initialize intensities
        self.varArray[0:self.K] = np.ones(self.K) * self.sinGap / self.K
        # initialize centers
        self.varArray[self.K:2 * self.K] = self.width / 2. + 2 * (np.random.rand(self.K) - 0.5) / max(10., self.width)
        # initialize widths
        self.varArray[2 * self.K:3 * self.K] = np.arange(0 + 1. / 2. / (self.K + 1), self.width,
                                                         1. * self.width / (self.K))

    # this is a mix of initializeVarsUniformCenter and initializeVarsUniformWidth
    def initializeVarsUniformMixed(self, ratio):
        # initialize intensities
        self.varArray[0:self.K] = np.ones(self.K) * self.sinGap / self.K
        # initialize centers
        self.varArray[self.K:self.K + self.K / ratio] = self.width / 2. + 2 * (
            np.random.rand(self.K / ratio) - 0.5) / max(10., self.width)
        self.varArray[self.K + self.K / ratio:2 * self.K] = np.arange(0 + 1. / 2. / ((self.K - self.K / ratio) + 1),
                                                                      self.width,
                                                                      1. * self.width / (self.K - self.K / ratio))
        # initialize widths
        self.varArray[2 * self.K:2 * self.K + self.K / ratio] = np.arange(0 + 1. / 2. / (self.K / ratio + 1),
                                                                          self.width,
                                                                          1. * self.width / (self.K / ratio))
        self.varArray[2 * self.K + self.K / ratio:3 * self.K] = self.width / 2 * np.ones(
            self.K - self.K / ratio) * np.random.rand(self.K - self.K / ratio)

    # this one has very little thought put into it and should be ignored
    def initializeVarsRandomShitty(self):
        self.varArray = np.ones(3 * self.K) * np.random.rand(self.K * 3)
        self.varArray[0:self.K] *= self.sinGap
        self.varArray[self.K:3 * self.K] *= self.width
        print self.varArray

    # returns both the objective function and the derivative for the solver
    def objGradEval(self, x):

        grad = np.zeros(3*self.K)
        # set g and gUnweighted to zero
        self.g.fill(0)
        self.gUnweighted.fill(0)
        self.grad.fill(0)
        gHolder = np.zeros((self.numApproxPoints, self.K))
        gLeft = np.zeros(self.numApproxPoints)
        gRight = np.zeros(self.numApproxPoints)

        # calc unweighted g
        for k in range(self.K):
            gHolder[:,k] = 0.5 * (sps.erf((self.approxPoints-(x[self.K+k] - x[2*self.K+k]))/self.sigma) + sps.erfc((self.approxPoints-(x[self.K+k] + x[2*self.K+k]))/self.sigma) - 1)
            self.g += x[k] * gHolder[:,k]

        self.diff = self.fTarget - self.g  # difference in original and sequenced fluence


        alphaDiff = -2. * self.alphas * self.diff
        for k in range(self.K):
            self.grad[k] =  np.sum(alphaDiff * gHolder[:,k])
            gLeft = 1./self.sigma * self.derf((self.approxPoints-(x[self.K+k] - x[2*self.K+k]))/self.sigma)
            gRight = 1./self.sigma * self.derf((self.approxPoints-(x[self.K+k] + x[2*self.K+k]))/self.sigma)
            self.grad[self.K+k] = x[k]/2. * np.sum(alphaDiff * (-gLeft+gRight))
            self.grad[2*self.K+k] = x[k]/2. * np.sum(alphaDiff * (gLeft+gRight))



        # plot if necessary
        if self.realTimePlotting and self.objCalls % 5 == 0:
            self.plotSolution(True, x)
        self.objCalls += 1
        return np.sum(self.alphas * (self.diff ** 2)), self.grad

    # invokes solver
    def solve(self):
        self.res = spo.minimize(self.objGradEval, x0=self.varArray.copy(), method='L-BFGS-B', jac=True,
                                bounds=self.bounds,
                                options={'ftol': 1e-4, 'disp': self.displayFreq})
        self.finalX = self.res['x']
        self.obj = self.res['fun']

    # plotter (can handle a single ongoing plot (0) or the single final plot (1))
    def plotSolution(self, ongoingfig=False, intermediateX=None, finalShow = False):
        # plot main function

        if ongoingfig:
            plt.figure(0)
            plt.clf()
            self.finalX = intermediateX
        else:
            plt.figure(0)
            plt.close()
            plt.figure(1)
            plt.ioff()

        # Objective function evaluation and plotting

        plt.plot(self.approxPoints, self.fTarget, 'r')

        # Sets coherent limits to plot
        plt.ylim(0, 1.2 * max(np.max(self.finalX[0:self.K]), self.sinGap + 1))
        plt.xlim(0, self.width)

        g = np.zeros(self.numApproxPoints)

        # set g and gUnweighted to zero
        gHolder = np.zeros((self.numApproxPoints, self.K))
        gLeft = np.zeros(self.numApproxPoints)
        gRight = np.zeros(self.numApproxPoints)

        # calc unweighted g
        for k in range(self.K):
            gHolder[:,k] = 0.5 * (sps.erf((self.approxPoints-(self.finalX[self.K+k] - self.finalX[2*self.K+k]))/self.sigma) + sps.erfc((self.approxPoints-(self.finalX[self.K+k] + self.finalX[2*self.K+k]))/self.sigma) - 1)
            g += self.finalX[k] * gHolder[:,k]


        plt.plot(self.approxPoints, g, 'g')

        # plots each individual aperture
        for k in range(self.K):
            # plot left error function up to center

            idx = self.approxPoints <= self.finalX[self.K + k]
            plt.plot(self.approxPoints[idx], self.finalX[k] * 1./2. * (1 + sps.erf(
                (self.approxPoints[idx] - (self.finalX[self.K + k] - self.finalX[2 * self.K + k])) / self.sigma)), 'b')
            # plot right error function
            idx = self.approxPoints > self.finalX[self.K + k]
            plt.plot(self.approxPoints[idx], self.finalX[k] / 2. * (
                sps.erfc(
                    (self.approxPoints[idx] - (self.finalX[self.K + k] + self.finalX[2 * self.K + k])) / self.sigma)),
                     'b')
        # updates figure
        if ongoingfig:
            plt.draw()
            if self.realTimePlotSaving:
                plt.savefig(self.runTag + '_ERFiterPlotOut_' + str(self.figCounter) + '.png')
                self.figCounter += 1
        else:
            plt.title('Method: ERF, obj: '+str(self.obj) + ', nAper: ' + str(self.K))
            plt.xlabel('Position along MLC opening')
            plt.ylabel('Fluence')
            plt.savefig(self.runTag + '_' + self.plotTag + '.png')
            if finalShow:
                plt.show()
            else:
                plt.show(block = False)

    def closePlots(self):
        plt.close('all')



    # outputs values to a .mat file incase of MATLAB integration
    def output(self, filename):
        io.savemat(self.runTag + '_' + filename, {'y': self.finalX[0:self.K], 'm': self.finalX[self.K:2 * self.K],
                              'a': self.finalX[2 * self.K:3 * self.K], 'obj': self.obj,
                              'sinGap': self.sinGap, 'K': self.K, 'width': self.width,
                              'numApprox': self.numApproxPoints, 'sinScalar': self.sinScalar, 'sigma': self.sigma,
                              'alphas': self.alphas, 'maxAperWidth': self.maxAperWidth,
                              'minAperWidth': self.minAperWidth,
                              'aperCenterOffset': self.aperCenterOffset})

    #todo have a close figure function