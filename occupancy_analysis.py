import sys

import logging.config
logger = logging.getLogger(__name__)

import os
from sap_funcs import setup_logging
from sap_funcs import traceback_msg
from sap_funcs import prepare_json_dict

import datetime
import glob

###### Version and Date
occupancy_version = 'v4.0_0B'
prog_date = '2018-09-17'

###### Usage
usage = '''

     Version %s by Christian Villarosa  %s

     Usage: python %s <JSON parameters>

''' % (occupancy_version, prog_date, os.path.basename(sys.argv[0]))

class OccupancyAnalysis(object):
    toggles = {
        'bin2npy': True
    }
    def __init__(self, platform='', data_dp='Data', slide='FLOWSLIDE', lane='L0X', fov='', blocks=[], cycle_start=1,
                 cycle_range=10, fastq_fp=None, temp_dp='Temp', output_dp='Output', parameter_overrides={},
                 log_dp='', log_overrides={}):
        from sap_funcs import make_dir

        self.start_time = datetime.datetime.now()
        parameter_overrides = prepare_json_dict(parameter_overrides)

        self.platform = parameter_overrides.pop('platform', platform)
        self.data_dp = parameter_overrides.pop('data_dp', data_dp)
        if not os.path.exists(self.data_dp):
            logger.error('Data directory path / data_dp (%s) does not exist!' % self.data_dp)
            sys.exit()
        self.slide = parameter_overrides.pop('slide', slide)
        self.lane = parameter_overrides.pop('lane', lane)
        self.fov = parameter_overrides.pop('fov', fov)
        self.blocks = parameter_overrides.pop('blocks', blocks)
        print(self.blocks)
        if bool(self.blocks):
            self.blocks = eval(self.blocks)
        self.start_cycle = parameter_overrides.pop('cycle_start', int(cycle_start))
        self.start_cycle = int(self.start_cycle)
        self.occupancy_range = parameter_overrides.pop('cycle_range', int(cycle_range))
        self.occupancy_range = int(self.occupancy_range)
        self.fastq_fp = parameter_overrides.pop('fastq_fp', fastq_fp)
        self.temp_dp = parameter_overrides.pop('temp_dp', temp_dp)
        output_dp = parameter_overrides.pop('output_dp', output_dp)
        if os.path.basename(output_dp) == self.fov:
            self.fov_dp = output_dp
            self.output_dp = os.path.dirname(output_dp)
        else:
            self.output_dp = output_dp
            self.fov_dp = os.path.join(output_dp, self.fov)

        self.log_overrides = parameter_overrides.pop('log_overrides', log_overrides)

        # report name can be specified in parameter_overrides
        self.report_name = parameter_overrides.pop('report_name', '%s_%s_%s_Occupancy_Analysis_C%02d-C%02d' %
                                                   (self.slide, self.lane, self.fov, self.start_cycle,
                                                    self.start_cycle + self.occupancy_range - 1))

        self.bypass = parameter_overrides.pop('bypass', {})
        if self.platform == 'v1':
            self.bypass['bin2npy'] = self.bypass.pop('bin2npy', False)
        elif self.platform == 'blackbird':
            self.bypass['nanocall_conversion'] = self.bypass.pop('nanocall_conversion', False)
        else:
            self.bypass['int2npy'] = self.bypass.pop('int2npy', False)
        self.bypass['intensity_analysis'] = self.bypass.pop('intensity_analysis', False)
        self.bypass['neighbor_analysis'] = self.bypass.pop('neighbor_analysis', False)
        self.bypass['label_analysis'] = self.bypass.pop('label_analysis', False)

        self.intensity_analysis_bypass = {}
        self.intensity_analysis_bypass['calculate_thresholds'] = self.bypass.pop('calculate_thresholds', False)

        self.neighbor_analysis_bypass = {}
        self.neighbor_analysis_bypass['get_possible_split_groups'] =  \
            self.bypass.pop('get_possible_split_groups', False)
        self.neighbor_analysis_bypass['calculate_split_percentage'] = \
            self.bypass.pop('calculate_split_percentage', False)
        self.neighbor_analysis_bypass['plot_multicalls'] = \
            self.bypass.pop('plot_multicalls', False)
        self.neighbor_analysis_bypass['plot_normCBI'] = \
            self.bypass.pop('plot_normCBI', False)
        self.neighbor_analysis_bypass['plot_nonCBI'] = \
            self.bypass.pop('plot_nonCBI', False)
        self.neighbor_analysis_bypass['plot_chastity'] = \
            self.bypass.pop('plot_chastity', False)
        self.neighbor_analysis_bypass['plot_SHI'] = \
            self.bypass.pop('plot_SHI', False)
        self.neighbor_analysis_bypass['plot_cbi_rank'] = \
            self.bypass.pop('plot_cbi_rank', False)
        self.neighbor_analysis_bypass['plot_cbi_thresholds'] = \
            self.bypass.pop('plot_cbi_thresholds', False)
        self.neighbor_analysis_bypass['plot_splits'] = \
            self.bypass.pop('plot_splits', False)

        self.label_analysis_bypass = {}
        # disable KDE plots by default
        self.label_analysis_bypass['plot_cbi_KDEs'] = self.bypass.pop('plot_cbi_KDEs', True)
        self.label_analysis_bypass['plot_cbi_hist'] = self.bypass.pop('plot_cbi_hist', False)

        # store files in fov subfolder
        if os.path.basename(self.temp_dp) != self.fov:
            self.temp_dp = os.path.join(self.temp_dp, self.fov)
        make_dir(self.temp_dp)

        make_dir(self.output_dp)
        make_dir(self.fov_dp)
        log_dp = os.path.join(self.output_dp, 'Logs')
        self.log_dp = parameter_overrides.pop('log_dp', log_dp)
        make_dir(self.log_dp)

        sub_log_fn = os.path.join(self.log_dp, '%s.log' % self.fov)
        sub_error_log_fn = os.path.join(self.log_dp, '%s_errors.log' % self.fov)
        override_dict = {'sub.log': sub_log_fn, 'sub_errors.log': sub_error_log_fn}
        override_dict.update(self.log_overrides)
        setup_logging(config_path='log_occupancy.yaml', overrides=override_dict)
        logger.info('Initiating Occupancy Analysis for %s...' % self.fov)

        for k, v in parameter_overrides.items():
            logger.warning('Extraneous parameter found - %s: %s' % (k, v))

        logger.debug(self.__dict__)
        return

    def process_data(self, platform):
        posinfo_fp, norm_paras_fp, background_fp = '', '', ''
        if platform == 'v1':
            int_dp = os.path.join(self.data_dp, 'Intensities')
            int_fp = self.run_bin2npy(int_dp, self.fov, self.start_cycle, self.occupancy_range, self.temp_dp)
            coords_fp, neighbors_fp, blocks_fp = self.run_pos2neighbors(int_dp, self.temp_dp, self.fov, self.blocks)
            fastq_fp = self.run_cal2fastq(self.data_dp, self.fov)
        elif platform == 'blackbird':
            coords_fp = 'cb_coords.npy' if self.fov == 'C006R013' else 'sp_coords.npy'
            neighbors_fp = 'cb_neighbors.npy' if self.fov == 'C006R013' else 'sp_neighbors.npy'
            blocks_fp = None
            int_fp, fastq_fp = self.run_nanocall_conversion(self.slide, self.lane, self.fov,
                                                            self.start_cycle, self.occupancy_range, self.temp_dp)
        else:
            int_fp, posinfo_fp, norm_paras_fp, background_fp = self.run_int2npy(
                self.data_dp, self.fov, self.start_cycle, self.occupancy_range, self.temp_dp)
            coords_fp, neighbors_fp, blocks_fp = self.run_pos2neighbors(self.data_dp, self.temp_dp, self.fov,
                                                                        self.blocks, v1=False)
            fastq_fp = self.run_v2cal2fastq(self.temp_dp, self.fov, blocks_fp)
        return int_fp, posinfo_fp, coords_fp, neighbors_fp, blocks_fp, fastq_fp, norm_paras_fp, background_fp

    def run_bin2npy(self, data_dp, fov, start_cycle, occupancy_range, temp_dp):
        from bin2npy import Bin2npy

        b2n = Bin2npy(data_dp, fov, start_cycle, occupancy_range, temp_dp)
        if self.bypass['bin2npy']:
            int_fp = b2n.complete_bypass()
        else:
            int_fp = b2n.run()
        return int_fp

    def run_pos2neighbors(self, data_dp, temp_dp, fov, blocks, v1=True):
        import pos2neighbor

        # use/store from temp
        coords_fp = os.path.join(temp_dp, '%s_coords.npy' % fov)
        neighbors_fp = os.path.join(temp_dp, '%s_neighbors.npy' % fov)
        blocks_fp = os.path.join(temp_dp, '%s_blocks.npy' % fov) if blocks else None
        if not os.path.exists(coords_fp) or not os.path.exists(neighbors_fp) or \
                (blocks_fp is not None and not os.path.exists(blocks_fp)):
            if os.path.exists( coords_fp) and os.path.exists(neighbors_fp) and \
                    os.path.exists(blocks_fp):
                coords_fp =  coords_fp
                neighbors_fp =  neighbors_fp
                blocks_fp =  blocks_fp
            else:
                # CREATE NEIGHBORS.npy IF IT DOES NOT ALREADY EXIST
                posinfo_fp = os.path.join(temp_dp, '%s.posiIndex.txt' % fov)
                pos2neighbor.main(posinfo_fp, coords_fp, neighbors_fp, blocks_fp, 1, blocks, v1)
        return coords_fp, neighbors_fp, blocks_fp

    def run_cal2fastq(self, data_dp, fov):
        try:
            fastq_fp = glob.glob(os.path.join(data_dp, '%s*.fq.gz' % fov))[0]
        except:
            # implement cal2fastq later
            fastq_fp = self.fastq_fp
        return fastq_fp

    def run_nanocall_conversion(self, slide, lane, fov, start_cycle, occupancy_range, temp_dp):
        from nanocall_conversion import NanocallConverter
        ncc = NanocallConverter(slide, lane, fov, start_cycle, occupancy_range, temp_dp)
        if self.bypass['nanocall_conversion']:
            int_fp, fastq_fp = ncc.complete_bypass()
        else:
            int_fp, fastq_fp = ncc.run()
        return int_fp, fastq_fp

    def run_int2npy(self, data_dp, fov, start_cycle, occupancy_range, temp_dp):
        from occuint2npy import Int2npy
        i2n = Int2npy(data_dp, fov, start_cycle, occupancy_range, temp_dp)
        if self.bypass['int2npy']:
            int_fp, posinfo_fp, norm_paras_fp, background_fp = i2n.complete_bypass()
        else:
            int_fp, posinfo_fp, norm_paras_fp, background_fp = i2n.run()
        return int_fp, posinfo_fp, norm_paras_fp, background_fp

    def run_v2cal2fastq(self, data_dp, fov, blocks_fp):
        try:
            fastq_fp = glob.glob(os.path.join(data_dp, '%s*.fq.gz' % fov))[0]
        except:
            from v2cal2fastq import V2Cal2Fastq
            c2f = V2Cal2Fastq(self.data_dp, fov, self.start_cycle, self.occupancy_range, blocks_fp,
                              output_dp=self.temp_dp)
            fastq_fp = c2f.run()
        return fastq_fp

    def run_intensity_analysis(self, slide, lane, fov, start_cycle, occupancy_range,
                               int_fp, norm_paras_fp, background_fp, blocks_fp, temp_dp,
                               bypass):
        from intensity_analysis import IntensityAnalysis
        cal_fp = os.path.join(self.data_dp, 'calFile', '%s.cal' % fov)
        self.int_analysis = IntensityAnalysis(slide, lane, fov, start_cycle, occupancy_range,
                                              cal_fp, int_fp, norm_paras_fp, background_fp, blocks_fp,
                                              temp_dp, bypass, self.log_dp)
        if self.bypass['intensity_analysis']:
            self.rho_results, self.snr_results, self.thresholds_summary, self.cbi_bypassed = self.int_analysis.complete_bypass()
        else:
            self.rho_results, self.snr_results, self.thresholds_summary, self.cbi_bypassed = self.int_analysis.run()
        self.nonCBI_quartiles = self.int_analysis.nonCBI_quartiles
        return

    def run_neighbor_clustering(self, int_analysis, neighbors_fp, report_name):
        from neighbors_clustering import NeighborClustering
        block_bool = int_analysis.load_block_bool()
        nc = NeighborClustering(int_analysis, neighbors_fp, block_bool, report_name)
        nc.run(subsets='mixed')
        return

    def run_neighbor_analysis(self, int_analysis, coords_fp, neighbors_fp, blocks_fp, fastq_fp, bypass):
        from neighbor_analysis import NeighborAnalysis

        nbr_analysis = NeighborAnalysis(int_analysis, coords_fp, neighbors_fp, blocks_fp, fastq_fp, bypass=bypass,
                                        log_dp=self.log_dp)
        # neighbor_analysis can only be bypassed if int_analysis was since the latter recreates the label array
        if self.bypass['neighbor_analysis'] and self.cbi_bypassed:
            self.neighbors_summary, self.neighbors_results = nbr_analysis.complete_bypass()
        else:
            self.neighbors_summary, self.neighbors_results = nbr_analysis.run()
        return

    def run_label_analysis(self, int_analysis, bypass):
        from label_analysis import LabelAnalysis

        lbl_analysis = LabelAnalysis(int_analysis, bypass,
                                     log_dp=self.log_dp)

        if self.bypass['label_analysis']:
            self.size_summary, self.size_results, \
            self.multicall_summary, self.multicall_results, \
            self.chastity_summary, self.chastity_results, \
            self.SHI_results, \
            self.mixed_summary, \
            self.empty_splits_results, self.mixed_splits_results, \
            self.familial_results, \
            self.singular_summary, \
            self.splits_summary, self.splits_results, \
            self.cbi_quartile_results, self.snr1_quartile_results, self.snr2_quartile_results = \
                lbl_analysis.complete_bypass()
        else:
            self.size_summary, self.size_results, \
            self.multicall_summary, self.multicall_results, \
            self.chastity_summary, self.chastity_results, \
            self.SHI_results, \
            self.mixed_summary, \
            self.empty_splits_results, self.mixed_splits_results, \
            self.familial_results, \
            self.singular_summary, \
            self.splits_summary, self.splits_results, \
            self.cbi_quartile_results, self.snr1_quartile_results, self.snr2_quartile_results = lbl_analysis.run()
            if not bool(self.blocks):
                lbl_analysis_center = LabelAnalysis(int_analysis, bypass, center=True)
                self.size_summary_center, self.size_results_center, \
                self.multicall_summary_center, self.multicall_results_center, \
                self.chastity_summary_center, self.chastity_results_center, \
                self.SHI_results_center, \
                self.mixed_summary_center, \
                self.empty_splits_results_center, self.mixed_splits_results_center, \
                self.familial_results_center, \
                self.singular_summary_center, \
                self.splits_summary_center, self.splits_results_center, \
                self.cbi_quartile_results_center, self.snr1_quartile_results_center, self.snr2_quartile_results_center = \
                    lbl_analysis_center.run()
        return

    def output_reports(self):
        from sap_funcs import output_table
        from sap_funcs import output_df

        logger.debug('Outputting reports...')
        summary_fp = os.path.join(self.fov_dp, '%s_Summary.csv' % self.report_name)
        summary_data = self.singular_summary + self.size_summary + self.mixed_summary + self.multicall_summary + \
                       self.chastity_summary + self.neighbors_summary + self.splits_summary + self.thresholds_summary + self.rho_results
        output_table(summary_fp, summary_data, ['', self.fov])
        if not bool(self.blocks):
            center_summary_fp = os.path.join(self.fov_dp, '%s_Center2x2_Summary.csv' % self.report_name)
            center_summary_data = self.singular_summary_center + self.size_summary_center +\
                                  self.mixed_summary_center + self.multicall_summary_center + \
                                  self.chastity_summary_center + self.splits_summary_center
            output_table(center_summary_fp, center_summary_data, ['', self.fov])

            size_results_fp = os.path.join(self.fov_dp, '%s_Size_Results.csv' % self.report_name)
            results_data = self.size_results + self.snr_results
            output_table(size_results_fp, results_data, ['', self.fov])

            mixed_results_fp = os.path.join(self.fov_dp, '%s_Mixed_Results.csv' % self.report_name)
            results_data = self.multicall_results + self.SHI_results + self.chastity_results
            output_table(mixed_results_fp, results_data, ['', self.fov])

            split_results_fp = os.path.join(self.fov_dp, '%s_Split_Results.csv' % self.report_name)
            results_data = self.splits_results + self.neighbors_results + self.empty_splits_results + \
                           self.mixed_splits_results + self.familial_results
            output_table(split_results_fp, results_data, ['', self.fov])

            quartiles_header = ['', 'Q1', 'Q2', 'Q3', 'Q4']
            cbi_quartiles_fp = os.path.join(self.fov_dp, '%s_CBI_Quartiles.csv' % self.report_name)
            output_table(cbi_quartiles_fp, self.cbi_quartile_results, quartiles_header)

            snr1_quartiles_fp = os.path.join(self.fov_dp, '%s_SNR1_Quartiles.csv' % self.report_name)
            output_table(snr1_quartiles_fp, self.snr1_quartile_results, quartiles_header)

            snr2_quartiles_fp = os.path.join(self.fov_dp, '%s_SNR2_Quartiles.csv' % self.report_name)
            output_table(snr2_quartiles_fp, self.snr2_quartile_results, quartiles_header)

            nonCBI_quartiles_fp = os.path.join(self.fov_dp, '%s_nonCBI_Quartiles.csv' % self.report_name)
            index_list = ['%s nonCBI %s %dth' % (base, non_base, q) for base in 'ACGT' for q in [25, 50, 75] for non_base in 'ACGT']
            output_df(nonCBI_quartiles_fp, self.nonCBI_quartiles, index_list)
            logger.debug('Output completed.')
            return summary_fp, size_results_fp, mixed_results_fp, split_results_fp, \
                   cbi_quartiles_fp, snr1_quartiles_fp, snr2_quartiles_fp, nonCBI_quartiles_fp, center_summary_fp
        else:
            size_results_fp = os.path.join(self.fov_dp, '%s_Size_Results.csv' % self.report_name)
            results_data = self.size_results + self.snr_results
            output_table(size_results_fp, results_data, ['', self.fov])

            mixed_results_fp = os.path.join(self.fov_dp, '%s_Mixed_Results.csv' % self.report_name)
            results_data = self.multicall_results + self.SHI_results + self.chastity_results
            output_table(mixed_results_fp, results_data, ['', self.fov])

            split_results_fp = os.path.join(self.fov_dp, '%s_Split_Results.csv' % self.report_name)
            results_data = self.splits_results + self.neighbors_results + self.empty_splits_results + \
                           self.mixed_splits_results + self.familial_results
            output_table(split_results_fp, results_data, ['', self.fov])

            quartiles_header = ['', 'Q1', 'Q2', 'Q3', 'Q4']
            cbi_quartiles_fp = os.path.join(self.fov_dp, '%s_CBI_Quartiles.csv' % self.report_name)
            output_table(cbi_quartiles_fp, self.cbi_quartile_results, quartiles_header)

            snr1_quartiles_fp = os.path.join(self.fov_dp, '%s_SNR1_Quartiles.csv' % self.report_name)
            output_table(snr1_quartiles_fp, self.snr1_quartile_results, quartiles_header)

            snr2_quartiles_fp = os.path.join(self.fov_dp, '%s_SNR2_Quartiles.csv' % self.report_name)
            output_table(snr2_quartiles_fp, self.snr2_quartile_results, quartiles_header)

            nonCBI_quartiles_fp = os.path.join(self.fov_dp, '%s_nonCBI_Quartiles.csv' % self.report_name)
            index_list = ['%s nonCBI %s %dth' % (base, non_base, q) for base in 'ACGT' for q in [25, 50, 75] for non_base in 'ACGT']
            output_df(nonCBI_quartiles_fp, self.nonCBI_quartiles, index_list)
            logger.debug('Output completed.')
            return summary_fp, size_results_fp, mixed_results_fp, split_results_fp, \
                   cbi_quartiles_fp, snr1_quartiles_fp, snr2_quartiles_fp, nonCBI_quartiles_fp

    def copy_temps(self):
        path_parameters = {
            'fov_dp': self.fov_dp,
            'temp_dp': self.temp_dp
        }
        if os.name == 'posix':
            os.system('rsync -zarv --include="*/" --include="*.png" --include="*.csv" '
                      '--include="*_Labels.npy" --include="*_SNR_Values.npy" '
                      '--exclude="*" %(temp_dp)s/ %(fov_dp)s >> %(fov_dp)s/Copy_Log.txt' % path_parameters)
        else:
            try:
                os.system('robocopy %(temp_dp)s %(fov_dp)s *.csv *.png *_Labels.npy *_SNR_Values.npy *_avg_CBI.npy '
                          '/s /R:0 /W:0 >> %(fov_dp)s/Copy_Log.txt' % path_parameters)
            except:
                os.system('robocopy %(temp_dp)s %(fov_dp)s *.csv *.png *_Labels.npy *_SNR_Values.npy '
                          '/s /R:0 /W:0 >> %(fov_dp)s/Copy_Log.txt' % path_parameters)
        return

    def run(self):
        int_fp, posinfo_fp, coords_fp, neighbors_fp, blocks_fp, fastq_fp, norm_paras_fp, background_fp = self.process_data(
            self.platform)
        self.run_intensity_analysis(self.slide, self.lane, self.fov,
                                    self.start_cycle, self.occupancy_range, int_fp,
                                    norm_paras_fp, background_fp, blocks_fp, self.temp_dp,
                                    self.intensity_analysis_bypass)
        self.run_neighbor_clustering(self.int_analysis, neighbors_fp, self.report_name)
        self.run_neighbor_analysis(self.int_analysis, coords_fp, neighbors_fp, blocks_fp, fastq_fp, self.neighbor_analysis_bypass)
        self.run_label_analysis(self.int_analysis, self.label_analysis_bypass)

        if not bool(self.blocks):
            summary_fp, size_results_fp, mixed_results_fp, split_results_fp, \
            cbi_quartiles_fp, snr1_quartiles_fp, snr2_quartiles_fp, nonCBI_quartiles_fp, center_summary_fp = self.output_reports()
            self.copy_temps()
            return summary_fp, size_results_fp, mixed_results_fp, split_results_fp, cbi_quartiles_fp, snr1_quartiles_fp, \
                   snr2_quartiles_fp, nonCBI_quartiles_fp, center_summary_fp
        else:
            summary_fp, size_results_fp, mixed_results_fp, split_results_fp, \
            cbi_quartiles_fp, snr1_quartiles_fp, snr2_quartiles_fp, nonCBI_quartiles_fp = self.output_reports()
            self.copy_temps()
            return summary_fp, size_results_fp, mixed_results_fp, split_results_fp, cbi_quartiles_fp, snr1_quartiles_fp, \
                   snr2_quartiles_fp, nonCBI_quartiles_fp



@traceback_msg
def main(arguments):
    start_time = datetime.datetime.now()

    # single argument indicates json file path
    if len(arguments) == 1 and os.path.isfile(arguments[0]):
        parameter_overrides_fp = arguments[0]
        occupancy_parameters = prepare_json_dict(parameter_overrides_fp)

        oa = OccupancyAnalysis(parameter_overrides=occupancy_parameters)
        oa.run()

        time_diff = datetime.datetime.now() - start_time
        logger.info('Occupancy analysis completed. (%s)' % time_diff)
    else:
        print usage

    return

if __name__ == '__main__':
    logging.info("Starting logger...")
    main(sys.argv[1:])