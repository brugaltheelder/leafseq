__author__ = 's162195'

import numpy as np
import scipy.special as sps
import matplotlib.pyplot as plt



class functionGetter:
    def __init__(self, UpperRange, Resolution):
        self.fig = plt.figure()
        self.width, self.resolution = UpperRange, Resolution
        self.nApprox = int(self.width / self.resolution + 1)
        self.approxPoints = np.arange(0, self.width + self.resolution / 2, self.resolution)

    def erfSum(self, y, m, a, sigma, truncate = 0):
        g = np.zeros(self.nApprox)
        # for each approximation point, calculate the sequenced fluence
        for i in xrange(self.nApprox):
            # case with approx[i] <= m_k (center to right of approx point)
            idx = m >= self.approxPoints[i]
            g[i] += np.sum(y[idx] * (1 + sps.erf((self.approxPoints[i] - (m[idx] - a[idx])) / sigma)))
            # case with approx[i] > m_k (center to left of approx point)
            idx = m < self.approxPoints[i]
            g[i] += np.sum(y[idx] * (sps.erfc((self.approxPoints[i] - (m[idx] + a[idx])) / sigma)))
        g[g<truncate] = 0
        print np.min(g[g>0])
        return g

    def sinFunction(self, a, b):
        return np.sin(b * self.approxPoints) + a

    def functionPlotter(self, f, dimr, dimc, pos, blockVar= True):
        ax = self.fig.add_subplot(int(str(dimr) + str(dimc) + str(pos)))
        ax.set_ylim(0, 1.2 * np.max(f))
        ax.plot(self.approxPoints, f)
        plt.show(block = blockVar)
