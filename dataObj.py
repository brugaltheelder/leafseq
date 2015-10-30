__author__ = 's162195'

import numpy as np



class dataObj():
    """object holds core data for the explicit model

    :param self.objParams: objective function parameters (not really used anymore)
    :param self.res: resolution of function approximation
    :param self.numAper: Number of apertures to generate
    :param self.sigma: scalar for erf function (to make steeper)
    :param self.width: width of aperture opening
    :param self.alphas: weighting vector for fluence differences
    :param self.runTag: string tag for a particular function
    :param self.directory: output directory for the plots and matlab .mat output archives
    :param self.kReal: number of apertures to use after clinical greedy method is run (assigned later)
    """

    def __init__(self, objParams, res, K, sigma, width, alphas, aperParams, runTag, directory):
        self.objParams = objParams  # list of objective parameters [ci, bi for sin]
        self.resolution = res  # resolution of funciton approximation
        self.numAper = K  # number of apertures
        self.sigma = sigma  # Scalar to make erf steeper
        self.width = width
        self.alphas = np.copy(alphas)
        self.minAperWidth, self.maxAperWidth, self.aperCenterOffset = aperParams
        self.runTag = runTag
        self.directory = directory
        self.kReal = None

    def setKreal(self,k):
        """Sets ''self.kReal'' """
        self.kReal = k