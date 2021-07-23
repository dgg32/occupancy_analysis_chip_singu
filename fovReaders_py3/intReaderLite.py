from .fovReaderLite import FovReaderLite
from . import utilities as util
import numpy as np

class IntReaderLite(FovReaderLite):

    def __init__(self, filePath=None):
        super(IntReaderLite, self).__init__(filePath)
        self.filePath = filePath
        return

    def readInt(self, cycles=None):
        '''
            Read Int file:
            Parameters:
                cycles [None, int, list]:Number or range of cycles to read (default is all cycles)
            Return:
                data (dnb, channel, cycles): Intensity values
        '''
        data = self._readData(cycles, dtype=np.float16).transpose((2, 1, 0))
        return data

    def calculateRho(self, cycles=None):
        '''
            Calculate Rho (Average Raw CBI for each channel)
        '''
        #if normalization values not present
        #raise ValueError("Normalization values not present in {}".format(self.filePath))
        #TODO: Needs to be a file that has a normalization value
        return

