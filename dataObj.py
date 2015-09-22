__author__ = 's162195'

import numpy as np


class dataObj():
    def __init__(self, objParams, res, K, sigma, width, alphas, aperParams):
        self.objParams = objParams  # list of objective parameters [ci, bi for sin]
        self.resolution = res  # resolution of funciton approximation
        self.numAper = K  # number of apertures
        self.sigma = sigma  # Scalar to make erf steeper
        self.width = width
        self.alphas = np.copy(alphas)
        self.minAperWidth, self.maxAperWidth, self.aperCenterOffset = aperParams