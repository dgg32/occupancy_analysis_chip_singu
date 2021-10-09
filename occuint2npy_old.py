import sys
import os
import numpy as np
from intReader import IntensityBin

import logging.config
logger = logging.getLogger(__name__)
from sap_funcs import setup_logging
from sap_funcs import output_table

import datetime

class Int2npy(object):

    def __init__(self, data_dp, fov, start_cycle, cycle_range=10, read_len=20, output_dp='', int_type='finInts',basecaller='v2',
                 log_dp='', log_overrides={}):
        self.data_dp = data_dp
        self.fov = fov
        self.start_cycle = int(start_cycle)
        self.cycle_range = cycle_range
        self.read_len = int(read_len) if int(read_len)>=cycle_range else int(read_len)+10#max cycle length (read_len+start_cycle) to search for good cycles
        self.good_cycles = []
        self.output_dp = output_dp if output_dp else data_dp
        self.int_type = int_type
        self.basecaller = basecaller
        self.max_val = 65504 #maximum value for a float16

        self.posinfo_fp = os.path.join(self.output_dp, '%s.posiIndex.txt' % self.fov)
        self.int_fp = os.path.join(self.output_dp, '%s_%s.npy' % (self.fov, int_type))
        self.background_fp = os.path.join(self.output_dp, self.fov + '_background.npy')
        self.norm_paras_fp = os.path.join(self.output_dp, self.fov + '_normParas.npy')

        sub_log_fn = os.path.join(log_dp, '%s.log' % fov)
        sub_error_log_fn = os.path.join(log_dp, '%s_errors.log' % fov)
        override_dict = {'sub.log': sub_log_fn, 'sub_errors.log': sub_error_log_fn}
        override_dict.update(log_overrides)
        setup_logging(overrides=override_dict)

        logger.info('%s - Int2npy initialized.' % self.fov)
        logger.info('Numpy version: %s' % np.version.version)
        logger.debug(self.__dict__)
        return

    def get_coords_zebracall(self, intensity_bin_array):
        logger.info('%s - Extracting positional information from intensity bin array...' % self.fov)
        data = np.zeros((intensity_bin_array.metrics.SpotNumber, (intensity_bin_array.metrics.ChannelNumber * 2) + 3), dtype=np.float)

        # extract subpixel coords
        metList = ["BlockId", "DnbRow", "DnbCol", "Coords"]
        blockList = intensity_bin_array.metrics.ExtractOrder
        ct = 0
        for b in blockList:
            blk = intensity_bin_array.metrics.Blocks[b]
            blkData = intensity_bin_array.getMetricsByBlock(blk, metList)
            data[ct:ct + blkData.shape[0], ] = blkData
            ct += blkData.shape[0]

        logger.info('%s - Positional information extracted.' % self.fov)
        data = data[:,:5].T.tolist()
        data = list(zip(range(intensity_bin_array.metrics.SpotNumber), *data))
        logger.debug(str(data[0]))
        return data

    def get_coords_Lite(self,cycle):
        from fovReaders_py3 import coordReaderLite,utilities
        coord_fp = os.path.join(self.data_dp, 'Coord', '%s.Coord' % self.fov)
        coordReader = coordReaderLite.CoordReaderLite(coord_fp)
        spot_num = coordReader.header.fileTag.SpotNum
        coords = coordReader.readCoord([cycle,cycle]) # which cycle? #cycle x ch x dnbs x pos
        blocks = utilities.sequencerVersion(coordReader.header.version)
        logger.info('%s - Positional information extracted.' % self.fov)
        data = np.zeros( (spot_num,5),dtype=np.float )
        data[:,0] = blocks.block_ids
        data[:,1] = blocks.row_ids
        data[:,2] = blocks.col_ids
        data[:,3:] = coords[0,0,:,:] #only need relative locations; same for all chs so just pick first
        data = list(zip(range(coords.shape[2]),*(data.T.tolist())))
        logger.debug(str(data[0]))
        return data
        

    def _get_good_cycles(self):
        cycles = range(self.start_cycle, self.start_cycle + self.read_len)
        if self.basecaller=='v2':
            for i, cycle in enumerate(cycles):
                int_bin_fp = os.path.join(self.data_dp, self.int_type, 'S%03d' % cycle, '%s.int' % self.fov)
                if os.path.exists(int_bin_fp):
                    self.good_cycles.append(cycle-1)
        elif self.basecaller=='Lite':
            from fovReaders_py3 import intReaderLite
            int_fp = os.path.join(self.data_dp, 'FInt', '%s.FInt' % self.fov)
            intReader = intReaderLite.IntReaderLite(int_fp)
            for i,cycle in enumerate(cycles):
                cycle_ints = intReader.readInt([cycle,cycle]) #read one cycle at a time
                if (~np.all(cycle_ints==self.max_val)) and (~np.all(cycle_ints==0)): #assume drop cycles will be set to all max value or all 0s
                    self.good_cycles.append(cycle-1) # convert to 0 indexing
        self.good_cycles = np.array(self.good_cycles)
        if len(self.good_cycles)<self.cycle_range: 
            logger.warning('Only {0:d} good cycles < {1:d} required cycles'.format(len(self.good_cycles),self.cycle_range))

            
    def _cycles_summary(self):
        cycles_range = np.arange(self.start_cycle-1, self.good_cycles.max()+1)
        mask = np.isin(cycles_range,self.good_cycles)
        self.cycles_summary = [ 
            ['Failed Cycles',list(cycles_range[~mask]+1)], #convert back to 1 indexing for summary table
            ['Used Cycles',list(cycles_range[mask]+1)]
        ]




    def run(self):
        ##goal
        ##np.save(self.int_fp, intensities)
        ##np.save(self.norm_paras_fp, norm_paras)
        ##np.save(self.background_fp, background)


        start_time = datetime.datetime.now()
        logger.info('%s - Consolidating intensities in npy format...' % self.fov)
        #intensities = np.empty((num_dnbs, 4, len(cycles)), dtype=np.float)

        
        cycles = range(self.start_cycle, self.start_cycle + self.read_len)
        logger.debug(f'Sixing; {self.start_cycle}, {self.read_len}, {cycles}')

        intensities = []
        norm_paras = np.zeros((16, len(cycles)))
        background = np.zeros((100, 4, len(cycles)))
        if self.basecaller=='v2':
            logger.debug('Reading Zebracall intensities')
            for i, cycle in enumerate(cycles):
                int_bin_fp = os.path.join(self.data_dp, self.int_type, 'S%03d' % cycle, '%s.int' % self.fov)
                if os.path.exists(int_bin_fp):
                    self.good_cycles.append(cycle-1) # convert to 0 indexing
                    ib = IntensityBin(int_bin_fp)
                    cycle_ints = ib.metrics.IntsData.reshape(ib.metrics.ChannelNumber, -1).T

                    logger.debug('cycle_ints.shape: %s' % str(cycle_ints.shape))
                    intensities.append(cycle_ints[:,:,None]) #add third axis for cycles
                    try:
                        norm_paras[:, i] = ib.metrics.NormalizationValue
                    except:
                        pass
                    try:
                        background[:, :, i] = ib.dump_block_backgrounds()
                    except:
                        pass
                    # if cycle == self.start_cycle:  # output posiIndex file
                    if not os.path.exists(self.posinfo_fp):
                        # if 'V0.2' in self.data_dp:
                        #     num_colors = 2
                        # else:
                        #     num_colors = 4
                        pos_list = self.get_coords_zebracall(ib)
                        output_table(self.posinfo_fp, pos_list, delimiter='\t')
                        logger.debug('posinfo_fp created.')
                else:
                    logger.debug('Missing Data Cycle {0:03d} (1 indexing)'.format(cycle))
                    continue
                if len(self.good_cycles)==self.cycle_range:
                    break

        elif self.basecaller=='Lite':
            # ignore background and norm_paras for Lite since there is not corresponding metaData?
            logger.debug('Reading Lite intensities')
            from fovReaders_py3 import intReaderLite
            int_fp = os.path.join(self.data_dp, 'FInt', '%s.FInt' % self.fov)
            intReader = intReaderLite.IntReaderLite(int_fp)

            logger.debug(f'Sixing; {cycles}, range: {self.cycle_range}')
            for i,cycle in enumerate(cycles):
                cycle_ints = intReader.readInt([cycle,cycle]) #read one cycle at a time
                #assume drop cycles will be set to all max value or all 0s
                if (~np.all(cycle_ints==self.max_val)) and (~np.all(cycle_ints==0)): 
                    logger.debug('cycle_ints.shape: %s' % str(cycle_ints.shape))
                    intensities.append(cycle_ints)
                    self.good_cycles.append(cycle-1) # convert to 0 indexing
                else:
                    logger.debug('Missing Data Cycle {0:03d} (1 indexing)'.format(cycle))
                    continue
                if len(self.good_cycles)==self.cycle_range:
                    break
            #create posiIndex file

            ####Sixing; the amount of good_cycles depends on cycle_range 
            logger.debug(f'Sixing; self.good_cycles: {self.good_cycles}')

            pos_list = self.get_coords_Lite(self.good_cycles[0]+1)
            output_table(self.posinfo_fp, pos_list, delimiter='\t')
            logger.debug('posinfo_fp created.')
            
        self.good_cycles = np.array(self.good_cycles)
        self._cycles_summary()
        ##TODO Throw Error here? and break out of occupancy for FOV?
        if len(self.good_cycles)<self.cycle_range: 
            logger.warning('FOV:{0}, Only {1:d} good cycles < {2:d} required cycles'.\
                                format(self.fov,len(self.good_cycles),self.cycle_range))

        intensities = np.concatenate(intensities,axis=2).astype(np.float32)
        # # swap axes so that order is: num_dnbs, num_channels, num_cycles ##not needed with concatenate
        # intensities = np.swapaxes(intensities, 0, 1)
        # intensities = np.swapaxes(intensities, 1, 2)
        logger.debug('%s - intensities.shape: %s' % (self.fov, str(intensities.shape)))

        # convert max values?
        """
        dnb, channel, cycle = np.where(int_data == int_data.max())
        intensities[dnb, :, cycle] = 0
        """
        print ("intensities", intensities)
        print ("norm_paras", norm_paras)
        np.save(self.int_fp, intensities)
        np.save(self.norm_paras_fp, norm_paras)
        np.save(self.background_fp, background)
        time_diff = datetime.datetime.now() - start_time
        logger.info('%s - Complete (%s)' % (self.fov, time_diff))
        return self.int_fp, self.posinfo_fp, self.norm_paras_fp, self.background_fp

    def complete_bypass(self):
        if os.path.exists(self.int_fp) and os.path.exists(self.posinfo_fp) and os.path.exists(self.norm_paras_fp) \
                and os.path.exists(self.background_fp):
            self._get_good_cycles()
            self._cycles_summary()
            logger.info('%s - Bypass successful.' % self.fov)
            return self.int_fp, self.posinfo_fp, self.norm_paras_fp, self.background_fp
        logger.warning('%s - Could not bypass bin2npy!' % self.fov)
        return self.run()

def main(data_dp, fov, occupancy_cycle, output_dp):
    i2n = Int2npy(data_dp, fov, occupancy_cycle, output_dp=output_dp)
    int_fp = i2n.run()
    return int_fp

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])

