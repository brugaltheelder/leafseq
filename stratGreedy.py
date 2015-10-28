import numpy as np
import scipy.cluster as spc
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore','.*different initialization.*')

class stratGreedy(object):
	"""Greedy clinically-based leaf sequencing solver"""
	def __init__(self, f,L,width,K=None):
		self.f, self.L, self.K = f, L,None
		self.width = width
		self.M = self.f.shape[0]
		self.eps = 0.001
		if bool(K):
			self.K = K
		self.y, self.m, self.a = [], [],[]

	def runStratGreedy(self,L=None):
		if not bool(L):
			L = self.L
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

		return y,m,a,numK

	def plotStrat(self,y,m,a):
		approxPoints = np.arange(0,self.width+(1.0*self.width)/(self.M-1),1.0*self.width/(self.M-1))
		plt.plot(approxPoints,self.f,'r')
		plt.plot(approxPoints,self.f_strat_latest,'b')

		print self.f, approxPoints
		plt.show()