__author__ = 'Troy Long'

import scipy.optimize as spo
import scipy.signal as spsignal
import scipy.special as sps
from scipy import io

# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dataObj import *


class erfMidModel:
    """
    Takes an arbitrary fluence and some starting vector, descends to a local minimum 
    """
    def __init__(self, runData, realTimePlotting=False, realTimePlotSaving=False, startingSolutionVector=None,
                 trueFluenceVector=None, initializationStringAndParams=None, displayFreq = False, plotTag = '', plotSeed = False):
        # Read in parameters
        assert (isinstance(runData, dataObj))
        self.realTimePlotting, self.realTimePlotSaving = realTimePlotting, realTimePlotSaving
        if runData.kReal is None:
            self.K = runData.numAper
        else:
            self.K = runData.kReal

        self.width, self.resolution = 1.0 * runData.width, 1.0 * runData.resolution  # num apers, width of row, spacing of approx
        self.sinGap, self.sigma, self.sinScalar = 1.0 * runData.objParams[0], 1.0 * runData.sigma, 1.0 * \
                                                  runData.objParams[
                                                      1]  # shift for sin obj, scacling factor for erf, sin scaling factor
        self.directory = runData.directory
        self.minAperWidth, self.maxAperWidth, self.aperCenterOffset = runData.minAperWidth, runData.maxAperWidth, runData.aperCenterOffset  # min/max bounds for aper width, aper center offset from edges
        self.runTag = runData.runTag
        self.plotSeed = plotSeed
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
            self.seedY = np.copy(startingSolutionVector[0:self.K])
            self.seedM = np.copy(startingSolutionVector[self.K:2*self.K])
            self.seedA = np.copy(startingSolutionVector[2 * self.K:3 * self.K])
        elif 'Rand' in initializationStringAndParams[0]:
            m,a = self.initializeVarsRandom()            
            y = self.getY(m,a)
            self.setVarArray(y,m,a)
        elif 'Unif' in initializationStringAndParams[0]:
            m,a = self.initializeVarsSW()            
            y = self.getY(m,a)
            self.setVarArray(y,m,a)
        elif 'Cent' in initializationStringAndParams[0]:
            m,a = self.initializeVarsCentered()            
            y = self.getY(m,a)
            y = np.ones(self.K) * y.sum()/self.K
            self.setVarArray(y,m,a)
        elif 'Peak' in initializationStringAndParams[0]:
            m,a = self.initializeVarsPeaks()
            y = self.getY(m, a)
            self.setVarArray(y, m, a)
        else:
            print 'error'
            exit()
            # self.initializeVarsUniformMixed(2)


        # start ongoing graph if necessary
        if (self.realTimePlotting):
            plt.figure(0)
            self.figCounter = 0
            plt.ion()
            plt.show()

    def setVarArray(self,y,m,a):
        """Reads in seed, saves seed"""
        self.varArray[0:self.K] = y.copy()        
        self.varArray[self.K:2 * self.K] = m.copy()        
        self.varArray[2 * self.K:3 * self.K] = a.copy()
        self.seedY, self.seedM, self.seedA = y.copy(), m.copy(), a.copy()


    def getY(self,m,a):
        """Solves RMP for DLO

        :return seedRes['x']: returns the solution vector 
        """

        y = np.zeros(self.K)       
        # get l,r
        l = np.maximum(0,np.round(1.0*self.numApproxPoints/self.width*(m-a),0))
        r = np.minimum(self.numApproxPoints-1,np.round(1.0*self.numApproxPoints/self.width*(m+a),0))
        
        # build D
        self.seedD = np.zeros((self.numApproxPoints, self.K))
        for k in range(self.K):
            self.seedD[l[k]:r[k],k] = 1        

        #solve for y
        seedRes = spo.minimize(self.seedCalcObjGrad, x0 = y.copy(), method = 'L-BFGS-B',jac = True, bounds = np.array([(0,None) for i in range(self.K)]), options = {'ftol':1e-6, 'disp': self.displayFreq})

        return seedRes['x']


    
    def seedCalcObjGrad(self,y):
        """calculates seed obj and grad

        :return a,b: returns objective value, gradient vector of seed for getting seed intensities
        """
        diff = self.fTarget - self.seedD.dot(y)
        return np.sum(self.alphas * (diff ** 2)), self.seedD.transpose().dot(-2. * self.alphas * diff)



    def initializeVarsRandom(self):
        """Generates random centers and widths, see writeup for details

        :return m,a: returns centers vector, widths vector
        """
        m = np.random.rand(self.K) * self.width
        a = np.random.rand(self.K) * self.width / 2.0
        return m,a        

    def initializeVarsSW(self):
        """Generates beamlet-style centers and widths, see writeup for details

        :return m,a: returns centers vector, widths vector
        """
        m = np.array([1.0*(k*self.width)/(self.K+1) for k in range(1,self.K+1)])
        a = np.array([1.0*self.width/self.K for k in range(self.K)])
        return m,a

    def initializeVarsCentered(self):
        """Generates relatively centered centers and widths, see writeup for details

        :return m,a: returns centers vector, widths vector
        """
        m = np.array([1.0*self.width/2 for k in range(self.K)])
        m = self.width * (np.random.rand(self.K)-0.5)/2.0 + self.width/2.0
        a = np.array([1.0*(1.0* k*self.width)/(2.0*(self.K+1)) for k in range(1,self.K+1)])
        return m,a

    def initializeVarsPeaks(self):
        """Generates centers and widths for each peak, then random others, see writeup for details

        :return m,a: returns centers vector, widths vector
        """
        m,a = [self.width/2],[self.width/2]
        peaks = spsignal.find_peaks_cwt(self.fTarget, np.arange(10,20))    
        nPeaks= len(peaks)
        nAperPerPeak = int(np.floor(1.0*(self.K-1)/nPeaks))

        for kp in range(nAperPerPeak):
            for p in range(nPeaks):
                m.append(self.width*peaks[p]/self.numApproxPoints)
                a.append(1.0*(kp+1)*nPeaks/self.K)

        while len(m)<self.K:
            m.append(self.width/2)
            a.append(self.width/4 + 2.0*(np.random.rand(1)-0.5)*self.width/4)

        return np.array(m[:self.K]), np.array(a[:self.K])



    
    def derf(self, x):
        """calculates derivative of erf

        :returns derf: derivative of error function at x
        """
        return 2 / np.sqrt(np.pi) * np.exp(-1 * x * x)
        # return 2 / np.pi * np.exp(-1 * x * x)

    # returns both the objective function and the derivative for the solver
    def objGradEval(self, x):
        '''calculates helper arrays and then returns objective function and gradient for the explicit model

        :return a,b: returns objective function, gradient vector
        '''

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

    
    def solve(self):
        """Invokes solver for explicit model"""
        self.res = spo.minimize(self.objGradEval, x0=self.varArray.copy(), method='L-BFGS-B', jac=True,
                                bounds=self.bounds,
                                options={'ftol': 1e-4, 'disp': self.displayFreq})
        self.finalX = self.res['x']
        self.obj = self.res['fun']

    def plotSolution(self, ongoingfig=False, intermediateX=None, finalShow=False, fontsize=20):
        """Plotter (can handle a single ongoing plot (0) or the single final plot (1)). Plots target and generated fluence as well as aperture fluences"""

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

        plt.plot(self.approxPoints, self.fTarget, 'r', linestyle='dotted', zorder=2, linewidth=2)

        # Sets coherent limits to plot
        plt.ylim(0, 1.2 * max(np.max(self.finalX[0:self.K]), np.max(self.fTarget)))
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
                plt.savefig(self.directory + '/' + self.runTag + '_ERFiterPlotOut_' + str(self.figCounter) + '.png')
                plt.savefig(self.directory + '/' + self.runTag + '_ERFiterPlotOut_' + str(self.figCounter) + '.eps')
                self.figCounter += 1
        else:
            plt.title('Method: ' + self.plotTag + ', obj: ' + str(round(self.obj, 5)) + ', K: ' + str(self.K),
                      fontsize=fontsize)
            plt.xlabel('Position along MLC opening', fontsize=fontsize)
            plt.ylabel('Fluence', fontsize=fontsize)
            plt.savefig(self.directory + '/' + self.runTag + '_' + self.plotTag + '.png')
            plt.savefig(self.directory + '/' + self.runTag + '_' + self.plotTag + '.eps')
            if self.plotSeed:
                for k in range(self.K):
                # plot left error function up to center

                    idx = self.approxPoints <= self.seedM[k]
                    plt.plot(self.approxPoints[idx], self.seedY[k] * 1./2. * (1 + sps.erf(
                        (self.approxPoints[idx] - (self.seedM[k] - self.seedA[k])) / self.sigma)), 'k',
                             linestyle='dashed', zorder=3)
                    # plot right error function
                    idx = self.approxPoints > self.seedM[k]
                    plt.plot(self.approxPoints[idx], self.seedY[k] / 2. * (
                        sps.erfc(
                            (self.approxPoints[idx] - (self.seedM[k] + self.seedA[k])) / self.sigma)),
                             'k', linestyle='dashed', zorder=3)

                plt.savefig(self.directory + '/' + self.runTag + '_' + self.plotTag + '_withSeed.png')
                plt.savefig(self.directory + '/' + self.runTag + '_' + self.plotTag + '_withSeed.eps')

            if finalShow:
                plt.show()
            else:
                plt.show(block = False)

    def closePlots(self):
        """Closes all open plots from matplotlib"""
        plt.close('all')



    
    def output(self, filename):
        """outputs values to a .mat file incase of MATLAB integration"""
        io.savemat(self.directory + '/' + self.runTag + '_' + filename, {'y': self.finalX[0:self.K], 'm': self.finalX[self.K:2 * self.K],
                              'a': self.finalX[2 * self.K:3 * self.K], 'obj': self.obj,
                              'sinGap': self.sinGap, 'K': self.K, 'width': self.width,
                              'numApprox': self.numApproxPoints, 'sinScalar': self.sinScalar, 'sigma': self.sigma,
                              'alphas': self.alphas, 'maxAperWidth': self.maxAperWidth,
                              'minAperWidth': self.minAperWidth,
                              'aperCenterOffset': self.aperCenterOffset})

