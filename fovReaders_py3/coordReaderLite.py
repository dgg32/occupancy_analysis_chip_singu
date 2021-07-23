from .fovReaderLite import FovReaderLite
from .utilities import _bundle
import numpy as np

class CoordReaderLite(FovReaderLite):

    def __init__(self, filePath=None):
        super(CoordReaderLite, self).__init__(filePath)
        return

    def readCoord(self, cycles=None):
        '''
            Read Coordinate File
            NOTE:Requires that coordinates are saved as float32
        '''
        data = self._readData(cycles, dtype=np.uint64)
        data.dtype = np.float32
        cycles, ch, _ = data.shape
        data =data.reshape(cycles, ch, -1, 2)
        return data
