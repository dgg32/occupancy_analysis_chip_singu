import sys

import logging.config
logger = logging.getLogger(__name__)

import os
from sap_funcs import setup_logging
from sap_funcs import traceback_msg
from sap_funcs import prepare_json_dict
from sap_funcs import make_dir
from sap_funcs import output_table

import datetime
import glob
import multiprocessing as mp
CPU_COUNT = mp.cpu_count()
import time

import numpy as np

###### Version and Date
occupancy_version = 'v4.0_0B'
prog_date = '2018-09-17'

###### Usage
usage = '''

     Version %s by Christian Villarosa  %s

     Usage: python %s <JSON parameters>

''' % (occupancy_version, prog_date, os.path.basename(sys.argv[0]))

def parse_arguments(arguments):
    import argparse
    ArgParser = argparse.ArgumentParser(usage=usage, version=occupancy_version)
    ArgParser.add_argument("-p", "--platform", action="store", dest="platform", default="v2",
                           help="Data platform. Options: v1, v2, BB")
    ArgParser.add_argument("-s", "--slide", action="store", dest="slide", default="FLOWSLIDE",
                           help="Flowslide/Flowcell string ID. Example: CL100090543")
    ArgParser.add_argument("-l", "--lane", action="store", dest="lane", default="L01",
                           help="Lane string ID. Example: L01")
    ArgParser.add_argument("-f", "--fov_list", nargs="+", dest="fov_list", default=[],
                           help="List of FOVs to process. Empty list will process all.")
    ArgParser.add_argument('-b', '--blocks', action="store", dest='blocks', default=[],
                           help="Block numbers to include, empty list will include all")
    ArgParser.add_argument("-c", "--cycle_start", action="store", dest="cycle_start", default=1,
                           help="First cycle to be used in analysis. Used in conjunction with cycle range.")
    ArgParser.add_argument("-r", "--cycle_range", action="store", dest="cycle_range", default=10,
                           help="Number of cycles to analyze. Used in conjunction with cycle start.")
    ArgParser.add_argument("-a", "--cycle_list", nargs="+", dest="cycle_list", default=[],
                           help="Manually specify cycle numbers to analyze. Used in lieu of cycle start/range.")
    ArgParser.add_argument("-e", "--occupancy_version", action="store", dest="occupancy_version", default="",
                           help="Specify older version to run. Default: current version")
    ArgParser.add_argument("-t", "--temp_dp", action="store", dest="temp_dp", default="",
                           help="temp directory path. Default: current temp directory")
    ArgParser.add_argument("-o", "--output_dp", action="store", dest="output_dp", default="",
                           help="Output directory path. Default: current working directory")
    ArgParser.add_argument("-E", "--emails", nargs="+", dest="emails", default=[],
                           help="Emails to notify upon report completion.")
    ArgParser.add_argument("-d", "--data", action="store", dest="data_dp", default='',
                           help="Emails to notify upon report completion.")

    para, args = ArgParser.parse_known_args()

    # if len(args) != 1:
    #     ArgParser.print_help()
    #     print >>sys.stderr, "\nERROR: The parameters number is not correct!"
    #     sys.exit(1)

    occupancy_parameters = vars(para)
    # occupancy_parameters['data_dp'] = args[0]
    return vars(para)


def populate_default_parameters(occupancy_parameters):
    import re

    if 'platform' not in occupancy_parameters or not bool(occupancy_parameters['platform']):
        occupancy_parameters['platform'] = 'v1'

    ### PLATFORM DEPENDENCY ###
    if 'slide' not in occupancy_parameters or not bool(occupancy_parameters['slide']):
        try:
            if occupancy_parameters['platform'] == 'v1':
                slide = re.search(r'CL[0-9]+', os.path.basename(occupancy_parameters['data_dp'])).group()
            elif occupancy_parameters['platform'] == 'bb':
                slide = re.search(r'GS[0-9]+-FS3', os.path.basename(occupancy_parameters['data_dp'])).group()
            else:
                raise Exception
        except:
            slide = 'FLOWSLIDE'
        occupancy_parameters['slide'] = slide

    if 'lane' not in occupancy_parameters or not bool(occupancy_parameters['slide']):
        try:
            lane = re.search(r'L[0-9]{2}', os.path.basename(occupancy_parameters['data_dp'])).group()
        except:
            lane = 'L0X'
        occupancy_parameters['lane'] = lane

    if 'temp_dp' not in occupancy_parameters or not bool(occupancy_parameters['temp_dp']):
        occupancy_parameters['temp_dp'] = 'Temp'

    if 'output_dp' not in occupancy_parameters or not bool(occupancy_parameters['output_dp']):
        occupancy_parameters['output_dp'] = 'Output'
    if 'blocks' not in occupancy_parameters or not bool(occupancy_parameters['output_dp']):
        occupancy_parameters['blocks'] = []
    return occupancy_parameters

###### To-Do List
# - completely wrap v1 process
# - make wrapper cross-platform compatible
# - finish populating defaults
# - error showing up in info logs
# - mp error logs

def ints2fov_list(data_dp, platform):
    """
    Scour intensity files to determine list of FOVs.
    :param data_dp:
    :return:
    """
    from sap_funcs import int_extensions

    ### PLATFORM DEPENDENCY ###
    int_files = glob.glob(os.path.join(data_dp, 'finInts/S002/*.%s' % int_extensions[platform]))
    return [os.path.basename(f)[:-4] for f in int_files]

@traceback_msg
def fov_occupancy(fov, occupancy_parameters):
    from occupancy_analysis import OccupancyAnalysis
    logger.debug('fov_occupancy called')
    occupancy_parameters['fov'] = fov
    oa = OccupancyAnalysis(parameter_overrides=occupancy_parameters)
    return oa.run()

def fov_occupancy_star(arguments):
    #setup_logging(config_path='log_occupancy.yaml')
    logger.debug('fov_occupancy_star called')
    return fov_occupancy(*arguments)

def get_identification_strings(slide, lane='L0X', *spillover):
    slide = slide if slide else 'FLOWCELL'
    return slide, lane

def generate_final_paths(grouped_reports):
    final_report_fps = []
    for report_group in grouped_reports:
        priming_report = report_group[0]
        fov_dp, report_fn = os.path.split(priming_report)
        output_dp = os.path.dirname(fov_dp)
        comp_strings = report_fn.split('_Occupancy_Analysis_')
        slide, lane = get_identification_strings(*comp_strings[0].split('_'))
        new_fn = '%s_%s_Occupancy_Analysis_%s' % (slide, lane, comp_strings[-1])
        final_report_fps.append(os.path.join(output_dp, new_fn))
    return final_report_fps

def calculate_averages(metrics, data):
    data = zip(*data) # transpose
    metric_count = len(data)
    ignored_metrics = ['Most Frequent 10-mer']
    ignored_metrics = [metric for metric in ignored_metrics if metric in metrics]
    ignored_indices = [metrics.index(i) for i in ignored_metrics]
    data = [d for id, d in enumerate(data) if id not in ignored_indices]
    data = np.asarray(data, dtype=np.float32)
    avg_data = np.mean(data, 1).tolist()
    avg_list = []
    offset = 0
    for r in range(metric_count):
        if r in ignored_indices:
            avg_list.append('N/A')
            offset += 1
        else:
            avg_list.append(avg_data[r - offset])
    return avg_list

def calculate_quartiles_averages(data):
    data = np.asarray(data, dtype=np.float32)
    return zip(*np.mean(data, 0).tolist()) # convert to list and transpose

def consolidate_reports(report_lists):
    grouped_reports = zip(*report_lists)
    final_report_fps = generate_final_paths(grouped_reports)
    for g, report_group in enumerate(grouped_reports):
        final_report_fp = final_report_fps[g]

        fovs = []
        data = []
        for r, report_fp in enumerate(report_group):
            with open(report_fp, 'r') as report_f:
                report_table = [line.split(',') for line in report_f.read().split('\r\n') if line]
            if r == 0:
                metrics = [row[0] for row in report_table[1:]]
            fov = report_table[0][1]
            fovs.append(fov)

            values = [row[1] if len(row) == 2 else row[1:] for row in report_table[1:]]
            data.append(values)
        data = [y for x,y in sorted(zip(fovs, data))]
        if final_report_fp.endswith('Quartiles.csv'):
            avg_data = calculate_quartiles_averages(data)
            data_table = zip(metrics, *avg_data)
            output_table(final_report_fp, data_table, header=['', 'Q1', 'Q2', 'Q3', 'Q4'])
        else:
            avg_data = calculate_averages(metrics, data)
            data = zip(metrics, avg_data, *data)
            fovs = sorted(fovs)
            output_table(final_report_fp, data, header=['', 'AVG'] + fovs)
    return


def main(arguments):
    print('starting occupancy')
    start_time = datetime.datetime.now()

    # single argument indicates json file path
    if len(arguments) == 1 and os.path.isfile(arguments[0]):
        occupancy_json_fp = arguments[0]
        occupancy_parameters = prepare_json_dict(occupancy_json_fp)
    else:
        occupancy_parameters = parse_arguments(arguments)

    occupancy_parameters = populate_default_parameters(occupancy_parameters)

    bypass = dict(occupancy_parameters['bypass']) if ('bypass' in occupancy_parameters) else {}
    bypass['temp_deletion'] = bypass.pop('temp_deletion', True)

    make_dir(occupancy_parameters['temp_dp'])
    make_dir(occupancy_parameters['output_dp'])
    log_dp = os.path.join(occupancy_parameters['output_dp'], 'Logs')
    make_dir(log_dp)

    local_log_fn = 'dev_info.log'
    local_error_log_fn = 'dev_errors.log'
    remote_log_fn = os.path.join(log_dp, 'info.log')
    remote_error_log_fn = os.path.join(log_dp, 'errors.log')
    override_dict = {
        'remote.log': remote_log_fn, 'remote_errors.log': remote_error_log_fn,
        'instance_info.log': local_log_fn, 'instance_errors.log': local_error_log_fn}
    if 'log_overrides' in occupancy_parameters:
        occupancy_parameters['log_overrides'].update(override_dict)
    else:
        occupancy_parameters['log_overrides'] = override_dict
    setup_logging(config_path='log_occupancy.yaml', overrides=occupancy_parameters['log_overrides'])

    logger.info('Python Version: %s' % sys.version)

    fov_list = occupancy_parameters.pop('fov_list', [])

    if not fov_list:
        fov_list = ints2fov_list(occupancy_parameters['data_dp'], occupancy_parameters['platform'])

    pool = mp.Pool(processes=len(fov_list), maxtasksperchild=1)
    logger.info('Launching fov occupancy subprocess pool...')
    parameters_list = [(fov, occupancy_parameters) for fov in fov_list]

    occupancy_outputs = pool.imap_unordered(fov_occupancy_star, parameters_list)
    logger.info('start cycle ' + str([occupancy_parameters['cycle_start']]))
    logger.info('cycle range ' + str([occupancy_parameters['cycle_range']]))

    logger.debug('Pool launched...')
    time.sleep(10)
    logger.debug('Closing pool...')
    pool.close()
    logger.debug('Pool closed.')
    logger.debug('Joining pool...')
    pool.join()
    logger.debug('Pool joined.')

    occupancy_outputs = list(occupancy_outputs)
    occupancy_exceptions = [oo for oo in occupancy_outputs if type(oo) != tuple]
    for exception in occupancy_exceptions:
        logger.error('%s' % exception)
    occupancy_results = [oo for oo in occupancy_outputs if type(oo) == tuple]

    consolidate_reports(occupancy_results)

    if os.name == 'posix':
        os.system('rsync -zarv --include="*/" --include="*.png" --include="*.csv" '
                  '--include="*_Labels.npy" --include="*_SNR_Values.npy" '
                  '--exclude="*" %(temp_dp)s/ %(output_dp)s >> %(output_dp)s/Copy_Log.txt' % occupancy_parameters)
    else:
        os.system('robocopy %(temp_dp)s %(output_dp)s *.csv *.png *_Labels.npy *_SNR_Values.npy *_avg_CBI.npy '
                  '/s /R:0 /W:0 >> %(output_dp)s/Copy_Log.txt' % occupancy_parameters)

    if bypass['temp_deletion']:
        logger.debug('Retaining temporary files.')
    else:
        os.rmdir('%(temp_dp)s' % occupancy_parameters)

    """
    occupancy_outputs = list(occupancy_outputs)
    occupancy_exceptions = [output for output in occupancy_outputs if output != None]
    # email exceptions?
    
    """
    end_time = datetime.datetime.now()
    ela_time = end_time - start_time
    logger.info('Occupancy analysis complete! (%s)' % ela_time)
    return

if __name__ == '__main__':
    main(sys.argv[1:])