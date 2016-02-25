"""
Created on 29 October 2015
@author: Troy Long
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import scipy.cluster as spc
import scipy.special as sps
from scipy import io

warnings.filterwarnings('ignore','.*different initialization.*')

class stratGreedy(object):
    """Greedy clinically-based leaf sequencing solver"""

    def __init__(self, f,L, runData, K=None, plotTag = 'clinGreed'):
        self.f, self.L, self.K = f, L,None
        self.Linit = self.L
        self.width, self.sigma = 1.0* runData.width, 1.0* runData.sigma
        self.M = self.f.shape[0]
        self.directory = runData.directory
        self.runTag = runData.runTag
        self.plotTag = plotTag
        self.eps = 0.001
        self.approxPoints = np.arange(0,self.width+(1.0*self.width)/(self.M-1),1.0*self.width/(self.M-1))
        if bool(K):
            self.K = K
        self.alphas = np.copy(runData.alphas)       
        

    def runClinicalReturnErfSeed(self):
        '''
        This is the main run function that runs the full algorithm and returns the seed and objective. This calls **self.runStratGreedy**. 

        If a K is given, the algorithm will try to match that K. It builds up L then does a binary search along L to get close to K until K is reached or the binary search step is 1.

        :return a,b,c: resulting vector seed, objective, resulting number of apertures
        '''
        self.Kout = 0
        if self.K is None:
            self.Kout = self.runStratGreedy()
            return self.getErfInput(), self.getObj(), self.Kout
        
        #initialize starting L
        self.Kout = self.runStratGreedy(self.L)
        
        while self.Kout<self.K:            
            Kstart = self.Kout
            self.L = self.L * 2
            self.Kout = self.runStratGreedy(self.L)
            # print 'Starting L: L: ',self.L,', K: ',self.Kout, ' K target: ',self.K
            if self.Kout == Kstart:
                break


        #binary search
        Lstep = int(self.L/2)
        self.L = self.L - Lstep
        while Lstep>0:
            self.Kout = self.runStratGreedy(self.L)
            # print 'Binary Search: L: ',self.L,', K: ',self.Kout
            if self.Kout == self.K:
                break
            else:
                Lstep = int(Lstep/2)
                if self.Kout> self.K:                    
                    self.L = self.L - Lstep
                else:
                    self.L = self.L + Lstep

        return self.getErfInput(),self.getObj(), self.Kout




    def runStratGreedy(self,L=None):
        """ 
        Runs stratification and greedy aperture generation.

        :param levels: stratification levels.
        :param ind: level indices of each classified point along fluence curve.
        :param self.f_strat: stratified fluence.
        :return numK: return the resulting number of apertures used

        """
        if not bool(L):
            L = self.Linit
        # stratify levels
        levels,ind = spc.vq.kmeans2(self.f, L)
        f_strat = levels[ind]
        self.f_strat_latest = np.copy(f_strat)
        y,m,a = [],[],[]
        l,r = [],[]
        gradMask = np.zeros(f_strat.shape)
        # run iterative approach
        while f_strat.max()>self.eps:
            # find widest opening using CG pricing problem
            # find "gradient" by finding non-neg indices
            gradMask[:] = f_strat.shape[0]+1
            gradMask[f_strat>self.eps] = -1

            # run pricing problem
            lBest,rBest = -1, -1
            maxSoFar,maxEndingHere, lE, rE = 0,0,0,0
            for i in range(f_strat.shape[0]):
                maxEndingHere+=gradMask[i]
                if maxEndingHere>=0:
                    maxEndingHere,lE,rE = 0,i+1,i+1
                if maxSoFar>maxEndingHere:
                    maxSoFar,rE = maxEndingHere,i+1
                    lBest,rBest = lE,rE
            # add best block
            if lBest ==-1 and rBest ==-1:
                print 'error in the runStratGreedy algorithm'
            else:
                y.append(f_strat[lBest:rBest].min())
                l.append(lBest)
                r.append(rBest)
                f_strat[lBest:rBest]-= f_strat[lBest:rBest].min()

        #generate m,a
        numK = len(y)
        m = [0.5*(r[i] + l[i])*self.width/self.M for i in range(numK)]
        a = [0.5*(r[i] - l[i])*self.width/self.M for i in range(numK)]
        self.y, self.m, self.a = np.array(y).copy(), np.array(m).copy(), np.array(a).copy()

        return numK

    def getObj(self,gH=None):
        """Calculates objective function

        :return self.obj: Objective function when calculated using errof functions
        """
        if gH is None:
            g = np.zeros(self.M)
            for k in range(len(self.y)):                            
                g+= self.y[k] * 0.5 * (sps.erf((self.approxPoints - (self.m[k] - self.a[k]))/(self.sigma)) - sps.erf((self.approxPoints - (self.m[k] + self.a[k]))/(self.sigma)))
        else:
            g = gH.copy()
        self.obj = np.sum(self.alphas * ((self.f - g) ** 2))
        return self.obj

    def getErfInput(self):
        """
        Outputs a vector for the explicit model

        :return erfInputVector: concatinated [intensities,centers,half-widths], length 3*numApers
        """
        K = self.y.shape[0]
        erfInputVector = np.zeros(3*K)
        # populate intensities
        erfInputVector[0:K] = np.copy(self.y)
        # populate centers
        erfInputVector[K:2*K] = np.copy(self.m)
        # populate widths
        erfInputVector[2*K:3*K] = np.copy(self.a)
        return erfInputVector

    def output(self,filename):
        """Saves a MATLAB file with model outputs"""
        io.savemat(self.directory + '/' + self.runTag + '_' + filename, {'y': self.y, 'm': self.m,'a': self.a, 'obj': self.obj, 'K': self.obj, 'width': self.width, 'numApprox': self.M, 'alphas': self.alphas, 'sigma':self.sigma, 'K':self.y.shape[0]})

    def plotStrat(self, fontsize=20):
        """Plots the initial Fluence, stratified fluence, and aperture fluences"""

        plt.plot(self.approxPoints, self.f, 'r', linestyle='dotted', zorder=2, linewidth=2)
        # plt.plot(self.approxPoints,self.f_strat_latest,'c', linestyle = '')
        
        plt.ylim(0, 1.2 * max(np.max(self.y), np.max(self.f)))
        plt.xlim(0, self.width)
        # calculate g total, plot g holder
        gHolder = np.zeros(self.approxPoints.shape)
        g = np.zeros(self.approxPoints.shape)        

        for k in range(len(self.y)):
            gHolder = self.y[k] * 0.5 * (sps.erf((self.approxPoints - (self.m[k] - self.a[k]))/(self.sigma)) - sps.erf((self.approxPoints - (self.m[k] + self.a[k]))/(self.sigma)))
            plt.plot(self.approxPoints, gHolder,'b')
            g+= gHolder
        plt.plot(self.approxPoints, g, 'g', linestyle='solid')
        plt.title(
            'Method: Conv, obj: ' + str(round(self.getObj(g), 5)) + ', K: ' + str(self.y.shape[0]) + ', L: ' + str(
                self.L), fontsize=fontsize)
        plt.xlabel('Position along MLC opening', fontsize=fontsize)
        plt.ylabel('Fluence', fontsize=fontsize)
        plt.savefig(self.directory + '/' + self.runTag + '_' + self.plotTag + '.png')

        plt.show()
        
    def closePlots(self):
        """Closes all open plots from matplotlib"""
        plt.close('all')