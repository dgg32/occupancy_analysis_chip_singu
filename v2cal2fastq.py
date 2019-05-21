import sys
import os
import numpy as np

import logging.config
logger = logging.getLogger(__name__)
from sap_funcs import setup_logging

import datetime


class V2Cal2Fastq(object):

    def __init__(self, data_dp, fov, occupancy_cycle, occupancy_range,
                 blocks_fp, output_dp='', log_dp='', log_overrides={}):
        self.data_dp = data_dp
        self.fov = fov
        self.output_dp = output_dp if output_dp else data_dp
        self.occupancy_cycle = occupancy_cycle
        self.occu_range = occupancy_range
        self.cal_fp = os.path.join(data_dp, 'calFile', '%s.cal' % fov)
        self.fastq_fp = os.path.join(self.output_dp, '%s.fq.gz' % fov)
        if blocks_fp:
            self.blocks_vect = np.load(blocks_fp)
        else:
            self.blocks_vect = False

        sub_log_fn = os.path.join(log_dp, '%s.log' % fov)
        sub_error_log_fn = os.path.join(log_dp, '%s_errors.log' % fov)
        override_dict = {'sub.log': sub_log_fn, 'sub_errors.log': sub_error_log_fn}
        override_dict.update(log_overrides)
        setup_logging(overrides=override_dict)

        logger.info('%s - V2Cal2Fastq initialized.' % self.fov)
        logger.info('Numpy version: %s' % np.version.version)
        logger.debug(self.__dict__)
        return

    def run(self):
        from calReader import Cal

        start_time = datetime.datetime.now()
        logger.info('%s - Generating fastq file from cal file...' % self.fov)
        if not os.path.exists(self.cal_fp):
            return

        cal_obj = Cal()
        cal_obj.load(self.cal_fp, center_bool=self.blocks_vect)
        cal_obj.writefq(self.fastq_fp, idPrefix=self.fov, cycles=[self.occupancy_cycle,
                                                                  self.occupancy_cycle+self.occu_range])

        time_diff = datetime.datetime.now() - start_time
        logger.info('%s - Complete (%s)' % (self.fov, time_diff))
        return self.fastq_fp

    def complete_bypass(self):
        if os.path.exists(self.fastq_fp):
            logger.info('%s - Bypass successful.' % self.fov)
            return self.fastq_fp
        logger.warning('%s - Could not bypass bin2npy!' % self.fov)
        return self.run()


def main(data_dp, fov, occupancy_cycle, output_dp):
    c2f = V2Cal2Fastq(data_dp, fov, occupancy_cycle, output_dp=output_dp)
    fastq_fp = c2f.run()
    return fastq_fp

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])

