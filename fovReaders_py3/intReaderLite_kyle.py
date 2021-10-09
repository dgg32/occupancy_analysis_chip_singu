from fovReaderLite import FovReaderLite
import utilities as util
import numpy as np
import ipdb

class IntReaderLite(FovReaderLite):

    def __init__(self, filePath=None):
        super(IntReaderLite, self).__init__(filePath)
        self.filePath = filePath
        int_type = filePath.split(".")[-1]
        self.dtype = {"RInt" : np.float16, "FInt" : np.float32}.get(int_type, np.float16)
        return

    def readInt(self, cycles=None):
        '''
            Read Int file:
            Parameters:
                cycles [None, int, list]:Number or range of cycles to read (default is all cycles)
            Return:
                data (dnb, channel, cycles): Intensity values
        '''
        data = self._readData(cycles, dtype=self.dtype).transpose((2, 1, 0))
        return data

    def renormalizeInts(self, cycles=None, ints=None):
        '''
            Calculate Rho (Average Raw CBI for each channel)
        '''
        #if normalization values not present
        #raise ValueError("Normalization values not present in {}".format(self.filePath))
        #TODO: Needs to be a file that has a normalization value

        if ints is None:
            ints = self.readInt(cycles)
        start_cycle, end_cycle = self._parse_cycle_parameter(cycles)
        dnbs, channels, cycle_count = ints.shape
        norm_array = np.zeros((2, channels, cycle_count))
        #Generate norm array (subtract one to shift to 0 indexing)
        for cy in range(start_cycle - 1, end_cycle - 1):
            for ch in range(channels):
                chunk = self.header.chunkEntries[cy * channels + ch]
                norm_array[0, ch, cy] = chunk.NorLow
                norm_array[1, ch, cy] = chunk.NorUp

        #Use norm array to denormalize final intensities
        orig_ints = ints * (norm_array[1] - norm_array[0]) + norm_array[0]
        return orig_ints

    def calculateRho(self, cycles=None):
        ints = self.readInt(cycles)
        orig_ints = self.renormalizeInts(cycles, ints=ints)
        channels = self.header.fileTag["ChannelNum"]
        ch_i_mask = ints.argmax(axis=1)

        '''
        TODO: The criteria for which dnbs are filtered based on the final intensities needs some work
            Currently filtering DNBs as DNBs where the final intensity is neither Nan or the max float
        value, but there are still DNBs with abnormally high final intensity values (>100?) these values
        will majorly skew the average rho calculations. Need to determine a better way to filter out the
        bad DNBs to make this method useful.
        '''
        good_dnbs = np.all((~(np.isnan(ints) | (ints == np.finfo(self.dtype).max))), axis=1)


        rho = np.zeros((channels, ints.shape[2]))
        for cy_i in range(ch_i_mask.shape[1]):
            for ch_i in range(channels):
                ch_mask = (ch_i_mask[:, cy_i] == ch_i) & good_dnbs[:, cy_i]
                ch_ints = orig_ints[ch_mask, ch_i, cy_i].astype(np.float64)
                rho[ch_i, cy_i] = ch_ints.mean(axis=0)
        return rho
