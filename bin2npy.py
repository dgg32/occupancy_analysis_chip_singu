import os
import numpy as np
import extract_intensities_np

from zap_funcs import setup_logging
import logging
import sys

class Bin2npy(object):

    def __init__(self, int_path, fov, start_cycle, cycle_range=10, output_dp='', int_type='finInts',
                 log_dp='', log_overrides={}):
        self.int_path = int_path
        self.fov = fov
        self.start_cycle = int(start_cycle)
        self.cycle_range = cycle_range
        self.output_dp = output_dp if output_dp else int_path
        self.int_type = int_type
        self.npy_fp = os.path.join(self.output_dp, '%s.npy' % self.fov)

        sub_log_fn = os.path.join(log_dp, '%s.log' % fov)
        sub_error_log_fn = os.path.join(log_dp, '%s_errors.log' % fov)
        override_dict = {'sub.log': sub_log_fn, 'sub_errors.log': sub_error_log_fn}
        override_dict.update(log_overrides)
        setup_logging(overrides=override_dict)
        self.logger = logging.getLogger(__name__)
        self.logger.info('Consolidating %s intensities in npy format...' % fov)
        return

    def run(self):
        #intensities = np.empty((num_dnbs, 4, len(cycles)), dtype=np.float)
        intensities = []
        for cycle in range(self.start_cycle, self.start_cycle + self.cycle_range):
            bin_fp = os.path.join(self.int_path, self.int_type, 'S%03d' % cycle, '%s.bin' % self.fov)
            intensities.append(extract_intensities_np.main(bin_fp))

        intensities = np.asarray(intensities, dtype=np.float)
        # swap axes so that order is: num_dnbs, num_channels, num_cycles
        intensities = np.swapaxes(intensities, 0, 1)
        intensities = np.swapaxes(intensities, 1, 2)

        np.save(self.npy_fp, intensities)
        self.logger.info('Complete')
        return self.npy_fp

def main(intensities_dp, fov, occupancy_cycle, output_dp):
    b2n = Bin2npy(intensities_dp, fov, occupancy_cycle, output_dp=output_dp)
    npy_fp = b2n.run()


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])

