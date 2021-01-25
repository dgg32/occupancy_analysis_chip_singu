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

    def __init__(self, data_dp, fov, start_cycle, cycle_range=10, read_len=20, output_dp='', int_type='finInts',
                 log_dp='', log_overrides={}):
        self.data_dp = data_dp
        self.fov = fov
        self.start_cycle = int(start_cycle)
        self.cycle_range = cycle_range
        self.read_len = int(read_len) #max cycle length (read_len+start_cycle) to search for good cycles
        self.good_cycles = []
        self.output_dp = output_dp if output_dp else data_dp
        self.int_type = int_type

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

    def get_coords(self, intensity_bin_array, num_colors=4):
        logger.info('%s - Extracting positional information from intensity bin array...' % self.fov)
        data = np.zeros((intensity_bin_array.metrics.SpotNumber, (num_colors * 2) + 3), dtype=np.float)

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
        data = zip(range(intensity_bin_array.metrics.SpotNumber), *data)
        logger.debug(str(data[0]))
        return data

    def _get_good_cycles(self):
        cycles = range(self.start_cycle, self.start_cycle + self.read_len)
        for i, cycle in enumerate(cycles):
            int_bin_fp = os.path.join(self.data_dp, self.int_type, 'S%03d' % cycle, '%s.int' % self.fov)
            if os.path.exists(int_bin_fp):
                self.good_cycles.append(cycle-1)
        self.good_cycles = np.array(self.good_cycles)
        if len(self.good_cycles)<self.cycle_range: 
            logger.warning('Only {0:d} good cycles < {1:d} required cycles'.format(len(self.good_cycles),self.cycle_range))
            
    def _cycles_summary(self):
        cycles_range = np.arange(self.start_cycle-1, self.good_cycles.max()+1)
        mask = np.isin(cycles_range,self.good_cycles)
        self.cycles_summary = [ 
            ['Failed Cycles',cycles_range[~mask]+1], #convert back to 1 indexing for summary table
            ['Used Cycles',cycles_range[mask]+1] 
        ]

    def run(self):
        start_time = datetime.datetime.now()
        logger.info('%s - Consolidating intensities in npy format...' % self.fov)
        #intensities = np.empty((num_dnbs, 4, len(cycles)), dtype=np.float)

        cycles = range(self.start_cycle, self.start_cycle + self.read_len)

        intensities = []
        norm_paras = np.zeros((16, len(cycles)))
        background = np.zeros((100, 4, len(cycles)))
        for i, cycle in enumerate(cycles):
            int_bin_fp = os.path.join(self.data_dp, self.int_type, 'S%03d' % cycle, '%s.int' % self.fov)
            if os.path.exists(int_bin_fp):
                self.good_cycles.append(cycle-1) # convert to 0 indexing
                ib = IntensityBin(int_bin_fp)
                cycle_ints = ib.metrics.IntsData.reshape(ib.metrics.ChannelNumber, -1).T

                logger.debug('cycle_ints.shape: %s' % str(cycle_ints.shape))
                intensities.append(cycle_ints)
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
                    if 'V0.2' in self.data_dp:
                        num_colors = 2
                    else:
                        num_colors = 4
                    pos_list = self.get_coords(ib, num_colors=num_colors)
                    output_table(self.posinfo_fp, pos_list, delimiter='\t')
                    logger.debug('posinfo_fp created.')
            else:
                logger.debug('Missing Data Cycle {0:03d} (1 indexing)'.format(cycle))
                continue
            if len(self.good_cycles)==self.cycle_range:
                break
        self.good_cycles = np.array(self.good_cycles)
        self._cycles_summary()
        ##TODO Throw Error here? and break out of occupancy for FOV?
        if len(self.good_cycles)<self.cycle_range: 
            logger.warning('FOV:{0}, Only {1:d} good cycles < {2:d} required cycles'.\
                                format(self.fov,len(self.good_cycles),self.cycle_range))

        intensities = np.asarray(intensities, dtype=np.float32)
        # swap axes so that order is: num_dnbs, num_channels, num_cycles
        intensities = np.swapaxes(intensities, 0, 1)
        intensities = np.swapaxes(intensities, 1, 2)
        logger.debug('%s - intensities.shape: %s' % (self.fov, str(intensities.shape)))

        # convert max values?
        """
        dnb, channel, cycle = np.where(int_data == int_data.max())
        intensities[dnb, :, cycle] = 0
        """
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

