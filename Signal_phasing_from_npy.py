# import extract_intensities_np
# import subset_color_matrix_correction
import os
import csv
import sys
import numpy as np
import datetime
from zap_funcs import setup_logging
import logging

def average_ints(int_list):
    int_count = len(int_list)
    if int_count:
        return sum(int_list) / float(int_count)
    else:
        return 'NA'

class Zebra_Phase_Calculation(object):
    bases = ['A', 'C', 'G', 'T']

    def __init__(self, slide, fov, subset, int_vals, output_dp='', basecall_percentile=75,
                 log_dp='', log_overrides={}):
        self.slide = slide
        self.ctm = None
        self.lane = 'L0a'
        self.fov = fov + '_' + subset
        self.int_vals = int_vals
        self.cycle_range = [0, int(int_vals.shape[2])]
        self.basecall_percentile = basecall_percentile
        self.output_dp = output_dp
        self.phase_dict = {'lag': {'A': {'peak': [], 'base': [], 'delt': [], 'perc': []},
                                   'C': {'peak': [], 'base': [], 'delt': [], 'perc': []},
                                   'G': {'peak': [], 'base': [], 'delt': [], 'perc': []},
                                   'T': {'peak': [], 'base': [], 'delt': [], 'perc': []}},
                           'runon': {'A': {'peak': [], 'base': [], 'delt': [], 'perc': []},
                                     'C': {'peak': [], 'base': [], 'delt': [], 'perc': []},
                                     'G': {'peak': [], 'base': [], 'delt': [], 'perc': []},
                                     'T': {'peak': [], 'base': [], 'delt': [], 'perc': []}},
                           'read': {'A': {'peak': [], 'base': [], 'delt': [], 'perc': []},
                                    'C': {'peak': [], 'base': [], 'delt': [], 'perc': []},
                                    'G': {'peak': [], 'base': [], 'delt': [], 'perc': []},
                                    'T': {'peak': [], 'base': [], 'delt': [], 'perc': []}}}

        sub_log_fn = os.path.join(log_dp, '%s.log' % fov)
        sub_error_log_fn = os.path.join(log_dp, '%s_errors.log' % fov)
        override_dict = {'sub.log': sub_log_fn, 'sub_errors.log': sub_error_log_fn}
        override_dict.update(log_overrides)
        setup_logging(overrides=override_dict)
        self.logger = logging.getLogger(__name__)
        self.logger.info('Initiating %s signal phasing calculation...' % fov)
        self.logger.info('Numpy version: %s' % np.version.version)
        return



    def populate_data(self, cycle):
        read_ints = self.intensities[1]
        runon_ints = self.intensities[2]
        lag_ints = self.intensities[0]

        read_calls = self.basecalls[1]
        runon_calls = self.basecalls[2]
        lag_calls = self.basecalls[0]

        valid_read = self.valid_dnbs[1]
        valid_runon = self.valid_dnbs[2]
        valid_lag = self.valid_dnbs[0]

        valid_calls = np.logical_and(np.logical_and(valid_read, valid_runon), valid_lag)

        self.read_calls = read_calls[valid_calls] if valid_calls.any() and read_calls.any() else read_calls
        self.runon_calls = runon_calls[valid_calls] if valid_calls.any() and runon_calls.any() else runon_calls
        self.lag_calls = lag_calls[valid_calls] if valid_calls.any() and lag_calls.any() else lag_calls

        self.read_ints = read_ints[valid_calls]
        self.runon_ints = runon_ints[valid_calls] if runon_ints.any() else runon_ints
        self.lag_ints = lag_ints[valid_calls] if lag_ints.any() else lag_ints
        return

    def max_basecall(self, intensities):
        if not intensities.any(): return np.array([]), True
        max_dnbs = (np.apply_along_axis(lambda r: r == np.max(r),1,intensities)).astype(int)
        valid_dnbs = max_dnbs.sum(axis=1) == 1
        return max_dnbs, valid_dnbs

    def percentile_basecall(self, intensities):
        if not intensities.any(): return np.array([]), True
        percs = np.nanpercentile(intensities, self.basecall_percentile, axis=0)
        top_dnbs = (intensities > percs).astype(int)
        valid_dnbs = top_dnbs.sum(axis=1) == 1
        return top_dnbs, valid_dnbs

    def load_ints(self, cycle_number):
        return self.int_vals[:, :, cycle_number]

    def load_cc_ints(self, cycle_number):
        intensities = self.load_ints(cycle_number)
        # TODO Turn back on once debugging is futher along
        # if cycle_number > 3:
        #     return self.crosstalk_correct_ints(cycle_number, intensities, cob1=True) if intensities.any() else \
        #         intensities
        # else:
        #     if intensities.any():
        #         intensities, self.ctm = self.crosstalk_correct_ints(cycle_number, intensities, cob1=False)
        #     return intensities
        return intensities.copy()

    # def crosstalk_correct_ints(self, cycle_number, intensities, cob1):
    #     return subset_color_matrix_correction.main(self.slide, self.lane, self.fov, cycle_number, intensities,
    #                                                self.output_dp, cob=cob1, crosstalk_matrix=self.ctm)


    def calculate_read(self):
        self.phase_values['read'] = {}
        for i, base in enumerate(self.bases):
            self.phase_values['read'][base] = {}
            if self.read_calls.any() and self.lag_calls.any() and self.runon_calls.any():
                # read ints where lag or run on max_int does not match current int of interest (i)
                lag_filter = self.lag_calls.argmax(1).flatten() != i if self.lag_calls.any() else True
                ro_filter = self.runon_calls.argmax(1).flatten() != i if self.runon_calls.any() else True
                filtered_read_ints = self.read_ints[:,i][np.logical_and(lag_filter,ro_filter)]
                filtered_read_calls = self.read_calls[:,i][np.logical_and(lag_filter,ro_filter)].astype(bool)
                self.logger.debug('len(filtered_read_ints): %s' % len(filtered_read_ints))
                #self.logger.debug('len(filtered_read_calls): %s' % len(filtered_read_calls))

                peak_ints = filtered_read_ints[filtered_read_calls]
                base_ints =  filtered_read_ints[~filtered_read_calls]

                self.logger.debug('len(peak_ints): %s' % len(peak_ints))
                self.logger.debug('len(base_ints): %s' % len(base_ints))

                peak_value =  np.nanmean(peak_ints)
                base_value =  np.nanmean(base_ints)
                delt_value = peak_value - base_value
            else:
                peak_value = base_value = delt_value = 'NA'
            self.logger.debug('read, %s, %s' % (base, delt_value))
            self.phase_values['read'][base]['peak'] = peak_value
            self.phase_values['read'][base]['base'] = base_value
            self.phase_values['read'][base]['delt'] = delt_value
        return

    def calculate_phase(self, phase):
        if phase == 'lag':
            phase_calls = self.lag_calls
        else:
            phase_calls = self.runon_calls
        self.phase_values[phase] = {}
        for i, base in enumerate(self.bases):
            self.phase_values[phase][base] = {}
            if self.read_calls.any() and phase_calls.any():
                # read ints where read max_int does not match current int of interest
                read_filter = self.read_calls.argmax(1).flatten() != i if self.read_calls.any() else True
                filtered_read_ints = self.read_ints[:,i][read_filter]
                filtered_phase_calls = phase_calls[:, i][read_filter].astype(bool)

                self.logger.debug('len(filtered_read_ints): %s' % len(filtered_read_ints))
                self.logger.debug('len(filtered_phase_calls): %s' % len(filtered_phase_calls))

                peak_ints = filtered_read_ints[filtered_phase_calls]
                base_ints = filtered_read_ints[~filtered_phase_calls]

                self.logger.debug('len(peak_ints): %s' % len(peak_ints))
                self.logger.debug('len(base_ints): %s' % len(base_ints))

                peak_value =  np.mean(peak_ints)
                base_value =  np.mean(base_ints)
                delt_value = peak_value - base_value
            else:
                peak_value = base_value = delt_value = 'NA'
            self.logger.debug('%s, %s, %s' % (phase, base, delt_value))
            self.phase_values[phase][base]['peak'] = peak_value
            self.phase_values[phase][base]['base'] = base_value
            self.phase_values[phase][base]['delt'] = delt_value
        return

    def basecall(self, cycle_intensities):
        if self.basecall_percentile == float('inf'):
            basecalls, valid_dnbs = self.max_basecall(cycle_intensities)
        else:
            basecalls, valid_dnbs = self.percentile_basecall(cycle_intensities)
        return basecalls, valid_dnbs

    def increment_arrays(self, cycle_number):
        cycle_intensities = self.load_cc_ints(cycle_number)
        self.intensities.append(cycle_intensities)
        basecalls, valid_dnbs = self.basecall(cycle_intensities)
        self.basecalls.append(basecalls)
        self.valid_dnbs.append(valid_dnbs)
        return

    def calculate_percs(self):
        for base in self.bases:
            delt_values = {}
            NA_check = False
            for phase in self.phases:
                delt_values[phase] = self.phase_values[phase][base]['delt']
                if delt_values[phase] == 'NA':
                    NA_check = True
            if NA_check:
                for phase in self.phases:
                    self.phase_values[phase][base]['perc'] = 'NA'
            else:
                phase_sum = sum(delt_values.values())
                for phase in self.phases:
                    phase_perc = delt_values[phase] / float(phase_sum)
                    self.phase_values[phase][base]['perc'] = phase_perc

    def output_cycle_results(self, cycle_number):
        # output_fn = '%s_%s_%s_S%03d_Phase_Results.csv' % (self.slide, self.lane, self.fov, cycle_number)
        # output_fp = os.path.join(self.output_dp, output_fn)
        # self.logger.debug('output_fp: %s' % output_fp)
        value_types = ['peak', 'base', 'delt']
        phases = ['read', 'runon']
        if cycle_number > 1: phases.append('lag')
        # base_row = [self.slide, self.lane, self.fov, cycle_number]
        # with open(output_fp, 'wb') as output_f:
        #     output_csv = csv.writer(output_f)
        #     header = ['slide', 'lane', 'fov', 'cycle', 'int_type', 'base', 'phase', 'comp', 'value']
        #     output_csv.writerow(header)
        for base in self.bases:
            # calculate percentages
            delt_values = []
            for phase in phases:
                for vt in value_types:
                    self.phase_dict[phase][base][vt].append(self.phase_values[phase][base][vt])
                delt_values.append(self.phase_values[phase][base]['delt'])

            phase_sum = sum(delt_values) if 'NA' not in delt_values else 0
            for pi, phase in enumerate(phases):
                phase_perc = delt_values[pi] / float(phase_sum) if phase_sum else 0
                if phase_perc == np.nan:
                    phase_perc = 0
                self.phase_dict[phase][base]['perc'].append(phase_perc)
        return

    def run(self):
        self.intensities = [np.array([])]
        self.basecalls = [np.array([])]
        self.valid_dnbs = [True]

        for cycle_number in range(self.cycle_range[1]):
            self.logger.debug('cycle_number: %s' % cycle_number)
            self.increment_arrays(cycle_number)
            if not cycle_number: continue
            if len(self.intensities) == 4:
                self.intensities.pop(0)
                self.basecalls.pop(0)
                self.valid_dnbs.pop(0)
            self.populate_data(cycle_number)

            self.phase_values = {}

            self.calculate_read()
            self.calculate_phase('runon')
            # if cycle_number > 1:
            self.calculate_phase('lag')

            self.output_cycle_results(cycle_number)
        return self.phase_dict

def main(args):
    start = datetime.datetime.now()
    if len(args) > 1 or True:
        slide = 'CL100046011'
        lane = 'L01'
        fov = 'C002R023'
        cycle_range = [1, 201]
        int_path = r'\\zebra-stor\zebra-01\Zebra_Images\CL100046011\2018-04-18_13_20_07_1001160061_R0143_B_CL100046011_qy_SE400_result\CL100046011\L01\zebracall_online_v1.1.0.17505\Intensities\rawInts'
    else:
        i = 1
        slide = args[i]; i += 1
        lane = args[i]; i += 1
        fov = args[i]; i += 1
        cycle_range = args[i]; i += 1
        int_path = args[i]; i += 1

    zpc = Zebra_Phase_Calculation(slide, lane, fov, cycle_range, int_path, log_overrides={'INFO': 'DEBUG'})
    zpc.run()

    stop = datetime.datetime.now()
    print (stop-start)
    return 1

if __name__ == '__main__':
    main(sys.argv)