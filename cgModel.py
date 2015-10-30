import scipy.optimize as spo
import scipy.special as sps
from scipy import io
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dataObj import *


class cgSolver:
    """
    Class for running the direct leaf optimization column generation solution methodology

    Initialization: reads in runData along with other run tags, pre-allocates values        
        
    :param self.y: output fluences
    :param self.lVec,self.rVec: left and right leaf positions  
    """

    def __init__(self, runData, realTimePlotting=False, realTimePlotSaving=False, trueFluenceVector=None, displayFreq=False, plotTag='', simpleG=False):

    
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
        self.runTag = runData.runTag
        self.objCalls = 0
        self.displayFreq = displayFreq
        self.plotTag = plotTag
        self.simpleG = simpleG
        # Initialize values
        self.nApprox = int(self.width / self.resolution + 1)  # number of approximation points
        self.alphas = np.copy(runData.alphas)  # copy over objective function weights

        # generate approximation points, including endpoints
        self.approxPoints = np.arange(0, self.width + self.resolution / 2, self.resolution)


        # check if there is a seed target fluence, if not, default to sin
        if trueFluenceVector is not None and len(self.approxPoints) == len(trueFluenceVector):
            self.f = trueFluenceVector
        else:
            self.f = np.array(np.sin(self.sinScalar * self.approxPoints) + self.sinGap)

        # initialize y vector (intensities)
        self.y = np.zeros(self.K)
        self.nY = 0
        # self.y = np.array([])

        # initiailze D sparse matrix of dimension (0,num points)
        self.D = np.zeros((self.nApprox, self.K))


        # initialize solution D vector
        self.g = np.zeros(self.nApprox)
        self.lVec = np.array([])
        self.rVec = np.array([])
        self.Dtolerance = 0.0001
                # start ongoing graph if necessary
        if (self.realTimePlotting):
            plt.figure(2)
            self.figCounter = 0
            plt.ion()
            plt.show()

    
    def calcObjGrad(self, y):
        """Returs obj, gradient

        :return a,b: returns objective function, gradient vector
        """
        self.g = self.D.dot(y)
        diff = self.f - self.g
        return self.evalObj(y, diff=diff), self.D.transpose().dot(self.evalGrad(y, diff=diff))

    
    def evalGrad(self, y, diff=None):
        """Evaluates gradient in y-space

        :return grad: returns gradient
        """
        # if diff==None, then calc diff, otherwise return gradient
        if diff is None:
            self.g = self.D.dot(y)
            diff = self.f - self.g
        return -2. * self.alphas * diff

    
    def evalObj(self, y, diff=None):
        """Evaluates objective

        :return obj: returns objective function value
        """
        # if diff==None, then calc diff, otherwise return objective
        if diff is None:
            self.g = self.D.dot(y)
            diff = self.f - self.g
        return np.sum(self.alphas * (diff ** 2))

    
    def finalObjEval(self):
        """Evaluates objective with erf leaves rather than unit step leaves

        :return obj: returns objective function value
        """
        # generate new D
        Ds = np.zeros((self.nApprox, len(self.y)))
        for y in range(len(self.y)):
            Ds[:, y] = 1. / 2. * (sps.erf((self.approxPoints - self.lVec[y]) / self.sigma) + sps.erfc(
                (self.approxPoints - self.rVec[y]) / self.sigma) - 1)
        g = Ds.dot(self.y)
        diff = self.f - g
        return np.sum(self.alphas * (diff ** 2))

    
    def solvePP(self, y):
        """Solve the leaf pricing problem given some fluence y

        :return aper: returns self.addAper returns
        """
        # calc gradient
        grad = self.evalGrad(y)
        lBest, rBest = -1, -1
        # do search
        # helper variables
        maxSoFar, maxEndingHere, lE, rE = 0, 0, 0, 0
        for i in range(self.nApprox):
            maxEndingHere += grad[i]
            if maxEndingHere >= 0:
                maxEndingHere, lE, rE = 0, i + 1, i + 1
            if maxSoFar > maxEndingHere:
                maxSoFar, rE = maxEndingHere, i + 1
                lBest, rBest = lE, rE
        # add aperture to model
        if lBest == -1 and rBest == -1:
            return self.addAper(0, 0)
        else:
            return self.addAper(lBest, rBest)  # rE is non-inclusive

    
    def solve(self, aperMax):
        """Run full solution methodology for CG - iterate between PP and RMP"""
        self.aperMax = aperMax
        while self.nY < self.aperMax:
            self.solvePP(self.y)
            self.solveRMP()
            #print 'Aperures {} of {} added, lPos: {}, rPos: {}, Obj: {}'.format(str(np.size(self.y)), str(self.aperMax), lPos, rPos,str(self.obj))

    
    def addAper(self, l, r):
        """Generates next line of D-matrix given left and right leaf positions

        :return a,b: returns aperture endpoints
        """        # calculate col of D, add to sparse matrix using error function...make midpoint then write out function...truncate at very small
        # calc Dcol along entire axis, then truncate
        Dcol = np.zeros(self.nApprox)
        if self.simpleG:

            Dcol[l:r] = 1

        else:
            Dcol = 1. / 2. * (sps.erf((self.approxPoints - self.approxPoints[l]) / self.sigma) + sps.erfc(
                (self.approxPoints - self.approxPoints[(r - 1)]) / self.sigma) - 1)
            Dcol[Dcol < self.Dtolerance] = 0


        self.D[:, self.nY] = Dcol



        # save l and r positions

        self.lVec = np.append(self.lVec, self.approxPoints[l])
        self.rVec = np.append(self.rVec, self.approxPoints[r-1])
        self.nY +=1
        return self.approxPoints[l], self.approxPoints[r - 1]

    
    def solveRMP(self):
        """Solves restricted master problem"""
        # todo build in upper bounds on fluence
        self.res = spo.minimize(self.calcObjGrad, x0=self.y.copy(), method='L-BFGS-B', jac=True,
                                bounds=np.array([(0, None) for i in range(np.size(self.y))]),
                                options={'ftol': 1e-6, 'disp': self.displayFreq})
        self.y = self.res['x']
        self.obj = self.res['fun']

    
    def getErfInput(self):
        """Builds input vector for explicit model

        :return erfInputVector: returns seed vector for Explicit model
        """
        # initialize return vector
        K = np.size(self.y)
        erfInputVector = np.zeros(3*K)
        # populate intensities
        erfInputVector[0:K] = np.copy(self.y)
        # populate centers
        erfInputVector[K:2*K] = self.lVec + 1./2.*(self.rVec - self.lVec)
        # populate widths
        erfInputVector[2*K:3*K] = 1./2.*(self.rVec - self.lVec)

        return erfInputVector

    
    def output(self, filename):
        """Saves to a matlab file"""
        io.savemat(self.directory + '/'+self.runTag + '_' + filename, {'y': self.y, 'l': self.lVec,
                              'r': self.rVec, 'obj': self.obj,
                              'sinGap': self.sinGap, 'K': self.K, 'width': self.width,
                              'numApprox': self.nApprox, 'sinScalar': self.sinScalar, 'sigma': self.sigma,
                              'alphas': self.alphas})

    def printSolution(self, ongoingfig=False, intermediateY=None, finalShow = False):
        """Plots target and generated fluence as well as aperture fluences"""

        # plot main function

        if ongoingfig:
            plt.figure(2)
            plt.clf()
            yVec = np.copy(intermediateY)
        else:
            plt.figure(2)
            plt.close()
            plt.figure(3)
            plt.ioff()
            yVec = np.copy(self.y)

        # Objective function evaluation and plotting

        plt.plot(self.approxPoints, self.f, 'r')

        # plots total sequenced fluence
        g = self.D.dot(yVec)
        plt.plot(self.approxPoints, g, 'g')

        # Sets coherent limits to plot
        plt.ylim(0, 1.2 * max(np.max(g), self.sinGap + 1))
        plt.xlim(0, self.width)



        # plots each individual aperture
        for k in range(np.size(yVec)):
            # plot left error function up to center
            plt.plot(self.approxPoints, yVec[k] * self.D[:, k], 'b')
            # updates figure
        if ongoingfig:
            plt.draw()
            if self.realTimePlotSaving:
                plt.savefig(self.directory + '/' + 'CGiterPlotOut_' + str(self.figCounter) + '_.png')
                self.figCounter += 1
        else:
            plt.title('Method: CG, obj: ' + str(round(self.finalObjEval(), 5)) + ', nAper: ' + str(np.size(self.y)))
            plt.xlabel('Position along MLC opening')
            plt.ylabel('Fluence')
            plt.savefig(self.directory + '/' + self.runTag + '_' + self.plotTag + '.png')
            if finalShow:
                plt.show()
            else:
                plt.show(block = False)

    def closePlots(self):
        """Closes all open plots from matplotlib"""
        plt.close('all')