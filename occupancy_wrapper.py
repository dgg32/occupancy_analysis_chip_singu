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
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

###### Version and Date
occupancy_version = 'v4.1_0A'
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
    ArgParser.add_argument("-L", "--consolidate_lanes", action="store_false", dest="consolidate_lanes", default=True,
                           help="Flag to not compile all lanes to single workbook. Default true.")
    para, args = ArgParser.parse_known_args()

    # if len(args) != 1:
    #     ArgParser.print_help()
    #     print >>sys.stderr, "\nERROR: The parameters number is not correct!"
    #     sys.exit(1)

    occupancy_parameters = vars(para)
    # occupancy_parameters['data_dp'] = args[0]
    return occupancy_parameters


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

    consolidate_lanes = occupancy_parameters['consolidate_lanes']
    occupancy_parameters.pop('consolidate_lanes')
    return occupancy_parameters, consolidate_lanes

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
    grouped_reports = zip(*report_lists)[:-5]
    final_report_fps = generate_final_paths(grouped_reports)
    for g, report_group in enumerate(grouped_reports):
        final_report_fp = final_report_fps[g]

        fovs = []
        metrics, data = [], []
        for r, report_fp in enumerate(report_group):
            with open(report_fp, 'r') as report_f:
                sp = '\n' if final_report_fp.endswith('Cluster_Mixed_Summary.csv') else '\r\n'
                report_table = [line.split(',') for line in report_f.read().split(sp) if line]

            if r == 0:
                metrics = [row[0] for row in report_table[1:]]
            fov = report_table[0][-1] if final_report_fp.endswith('Cluster_Mixed_Summary.csv') else report_table[0][1]
            fovs.append(fov)

            values = [row[1] if len(row) == 2 else row[1:] for row in report_table[1:]]
            if final_report_fp.endswith('Cluster_Mixed_Summary.csv'):
                values = [row[-1] for row in report_table[1:]]
            data.append(values)
        data = [y for x, y in sorted(zip(fovs, data))]
        if final_report_fp.endswith('Quartiles.csv'):
            avg_data = calculate_quartiles_averages(data)
            data_table = zip(metrics, *avg_data)
            output_table(final_report_fp, data_table, header=['', 'Q1', 'Q2', 'Q3', 'Q4'])
        else:
            avg_data = calculate_averages(metrics, data)
            data = zip(metrics, avg_data, *data)
            fovs = sorted(fovs)
            output_table(final_report_fp, data, header=['', 'AVG'] + fovs)
    return final_report_fps


def consolidate_split_base_comp_reports(report_lists):
    #splits_order = ['All Split', 'Horizontal', 'Vertical', 'Diagonal', 'Multi']
    fov_reports = zip(*report_lists)[-1]
    fov_dp, report_fn = os.path.split(fov_reports[0])
    output_dp = os.path.dirname(fov_dp)
    comp_strings = report_fn.split('_Occupancy_Analysis_')
    slide, lane = get_identification_strings(*comp_strings[0].split('_'))
    new_fn = '%s_%s_Occupancy_Analysis_%s' % (slide, lane, comp_strings[-1])
    final_report_fp = os.path.join(output_dp, new_fn)
    dfs = []
    for fov_report in fov_reports:
        df = pd.read_csv(fov_report, index_col=[0, 1, 2])
        dfs.append(df)
    df = pd.concat(dfs).T
    curr_lane = df.columns.levels[0][0]
    df = df.stack().assign(AVG=df.mean(level=2, axis='columns').stack()).unstack()
    df = df.sort_index(axis='columns').rename(columns={'AVG': curr_lane, '': 'AVG'})
    df.T.to_csv(final_report_fp)
    return


def consolidate_fov_plots(report_lists, output_dp):
    make_dir(os.path.join(output_dp, 'npy'))

    grouped_reports = zip(*report_lists)[-5:-1]
    fovs = []
    parent_child_reports = []
    parent_child_fp = ''

    # reports order: split_cbi_ratio_dist_npy, parent_cbi_dist_npy, children_cbi_dist_npy, avgCBI_hist_npy
    for g, report_group in enumerate(grouped_reports):
        report_group = np.sort(np.array(report_group))
        fovs = [os.path.split(f)[1].split('_')[2] for f in report_group]
        if len(fovs[0]) == 3:
            fovs = [os.path.split(f)[1].split('_')[3] for f in report_group]

        fov_dp, report_fn = os.path.split(report_group[0])
        fn = report_fn.split('_')
        if len(fn[1]) == 3:
            new_fn = '%s_%s_Occupancy_Analysis_%s_%s' % (fn[0], fn[1], fn[3], '_'.join(fn[4:]))
            cycles = fn[3]
        else:
            new_fn = '%s_%s_Occupancy_Analysis_%s_%s' % (fn[0], fn[2], fn[4], '_'.join(fn[5:]))
            cycles = fn[4]
        final_report_fp = os.path.join(output_dp, new_fn).replace('.npy', '.png')

        if g == 0:
            plot_split_cbi_ratio_dist(report_group, fovs, final_report_fp)
        elif g == 3:
            threshold_fps = [
                os.path.join(os.path.split(fov_dp)[0], fov, '%s_finInts_%s_Final_Thresholds.npy' % (fov, cycles)) for
                fov in fovs]
            plot_cbi(report_group, fovs, final_report_fp, threshold_fps)
        else:
            parent_child_reports.append(report_group)
            parent_child_fp = final_report_fp

    parent_child_fp = parent_child_fp.replace('_Children', '')
    plot_split_cbi_dist(parent_child_reports[0], parent_child_reports[1], fovs, parent_child_fp)
    return


def consolidate_lane_plots(slide_dp, occupancy_fn, prefix):
    fovs = [None, None, None, None]
    slide = prefix.split('_')[0]
    cycles = prefix.split('_')[-1]
    lanes = ['L01', 'L02', 'L03', 'L04']
    # populate group reports
    groups = ['Split_CBI-Ratio_Distributions', 'avgCBI_Hist', 'CBI_Distributions']
    report_groups = dict((group, []) for group in groups)
    final_report_fps = []
    for i, report in enumerate(groups):
        threshold_fps = []
        for lane in lanes:
            lane_dp = os.path.join(os.path.split(slide_dp)[0], lane, occupancy_fn)
            f = os.path.join(lane_dp, 'npy', '%s_%s_Occupancy_Analysis_%s_%s.npy' % (slide, lane, cycles, report))
            if os.path.exists(f):
                report_groups[report].append(f)
                threshold_fps.append(os.path.join(lane_dp, 'npy', '%s_%s_Occupancy_Analysis_%s_Final_Thresholds.npy' % (
                    slide, lane, cycles)))
        final_report_fps.append(os.path.join(slide_dp, '%s_%s.png' % (prefix, report)))

    # plot
    plot_split_cbi_ratio_dist(report_groups['Split_CBI-Ratio_Distributions'], fovs, final_report_fps[0], lanes=True)
    plot_cbi(report_groups['avgCBI_Hist'], fovs, final_report_fps[1], threshold_fps, lanes=True)
    plot_split_cbi_dist([None, None, None, None], report_groups['CBI_Distributions'], fovs, final_report_fps[2],
                        lanes=True)
    return


def plot_split_cbi_ratio_dist(report_group, fovs, final_report_fp, lanes=False):
    center_fovs = []
    fig1, axes1 = plt.subplots(3, 2, figsize=(12, 8), sharex=True, sharey=True)
    if lanes:
        fig1, axes1 = plt.subplots(2, 2, figsize=(12, 6), sharex=True, sharey=True)

    for ax1, report_fp, fov in zip(axes1.flatten(), report_group, fovs):
        if lanes:
            fov = os.path.basename(report_fp).split('_')[1]

        data = np.load(report_fp)
        if fov not in ['C001R006', 'C006R060']:
            center_fovs.append(data)

        ax1.plot(data[0], data[1], label='Horizontal', alpha=0.6)
        ax1.plot(data[4], data[5], label='Vertical', alpha=0.6)
        ax1.plot(data[8], data[9], label='Diagonal', alpha=0.6)
        ax1.set_xlabel('Small/Large CBI Ratio')
        ax1.set_ylabel('Density')
        ax1.tick_params(labelsize=12)
        ax1.set_title(fov, fontsize=14)
    axes1[0, 1].legend(bbox_to_anchor=(0.96, 0.9), prop={'size': 12})
    title = '%s' % os.path.basename(final_report_fp).replace('.png', '')
    if lanes:
        title += ': avg of center mFOVs'
    fig1.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig1.savefig(final_report_fp.replace('_Distributions.png', '_Small-Large.png'))
    plt.gcf().clear()
    plt.close()

    fig2, axes2 = plt.subplots(3, 2, figsize=(12, 8), sharex=True, sharey=True)
    if lanes:
        fig2, axes2 = plt.subplots(2, 2, figsize=(12, 6), sharex=True, sharey=True)
    for ax2, report_fp, fov in zip(axes2.flatten(), report_group, fovs):
        if lanes:
            fov = os.path.basename(report_fp).split('_')[1]

        data = np.load(report_fp)
        ax2.plot(data[2], data[3], label='Horizontal: Left/Right', alpha=0.6)
        ax2.plot(data[6], data[7], label='Vertical: Up/Down', alpha=0.6)
        ax2.plot(data[10], data[11], label='Diagonal:\nSum(Small)/Large CBI', alpha=0.6)
        ax2.set_xlabel('CBI Ratio')
        ax2.set_ylabel('Density')
        ax2.tick_params(labelsize=12)
        ax2.set_title(fov, fontsize=14)
    axes2[0, 1].legend(bbox_to_anchor=(0.96, 0.9), prop={'size': 12})
    title = '%s' % os.path.basename(final_report_fp).replace('.png', '')
    if lanes:
        title += ': avg of center mFOVs'
    fig2.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig2.savefig(final_report_fp.replace('_Distributions.png', '_Directional.png'))
    plt.gcf().clear()
    plt.close()

    if not lanes:
        center_fovs = np.array(center_fovs)
        center_fovs = np.mean(center_fovs, axis=0)
        f = os.path.join(os.path.dirname(final_report_fp), 'npy',
                         os.path.basename(final_report_fp).replace('.png', '.npy'))
        np.save(f, center_fovs)
    return center_fovs


def plot_cbi(report_group, fovs, final_report_fp, threshold_fps, lanes=False):
    center_fovs, center_fovs_thres = [], []
    cbi_labels = ['All DNBs', 'Singular', 'Non-Children', 'Non-Parent', 'Non-Split',
                  'Low SHI', 'High Chastity', 'Single-call', 'Non-Mixed', 'Non-Mixed/Split']
    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True, sharey=True)
    fig2, axes2 = plt.subplots(3, 2, figsize=(14, 10), sharex=True, sharey=True)
    if lanes:
        fig, axes = plt.subplots(2, 2, figsize=(14, 6), sharex=True, sharey=True)
        fig2, axes2 = plt.subplots(2, 2, figsize=(14, 6), sharex=True, sharey=True)
    for ax, ax2, report_fp, thresholds_fp, fov in zip(axes.flatten(), axes2.flatten(), report_group, threshold_fps, fovs):
        if lanes:
            fov = os.path.basename(report_fp).split('_')[1]

        data = np.load(report_fp)
        if fov not in ['C001R006', 'C006R060']:
            center_fovs.append(data)

        xs = np.linspace(0, 3.5, 1000)[:-1]
        for i, label in enumerate(cbi_labels):
            if label in ['All DNBs', 'Non-Children', 'Non-Parent', 'Non-Split', 'Non-Mixed/Split']:
                ax.plot(xs, data[i], label=label, alpha=0.6)
            if label in ['All DNBs', 'Low SHI', 'High Chastity', 'Single-call', 'Non-Mixed', 'Non-Mixed/Split']:
                ax2.plot(xs, data[i], label=label, alpha=0.6)
        if os.path.exists(thresholds_fp):
            thresholds = np.load(thresholds_fp)
            if fov not in ['C001R006', 'C006R060']:
                center_fovs_thres.append(thresholds)
            empty_fth, small_fth, large_fth, outlier_fth = thresholds
            ax.axvline(x=empty_fth, color='red', linestyle='--', alpha=0.6)
            ax.axvline(x=small_fth, color='blue', linestyle='--', alpha=0.6)
            ax.axvline(x=large_fth, color='green', linestyle='--', alpha=0.6)
            ax.axvline(x=outlier_fth, color='red', linestyle='--', alpha=0.6)

            ax2.axvline(x=empty_fth, color='red', linestyle='--', alpha=0.6)
            ax2.axvline(x=small_fth, color='blue', linestyle='--', alpha=0.6)
            ax2.axvline(x=large_fth, color='green', linestyle='--', alpha=0.6)
            ax2.axvline(x=outlier_fth, color='red', linestyle='--', alpha=0.6)
        ax.set_xlabel('CBI')
        ax.set_ylabel('Density')
        ax.tick_params(labelsize=12)
        ax.set_title(fov, fontsize=14)

        ax2.set_xlabel('CBI')
        ax2.set_ylabel('Density')
        ax2.tick_params(labelsize=12)
        ax2.set_title(fov, fontsize=14)
    axes[0, 1].legend(bbox_to_anchor=(0.96, 0.9), prop={'size': 12})
    axes2[0, 1].legend(bbox_to_anchor=(0.96, 0.9), prop={'size': 12})
    title = '%s' % os.path.basename(final_report_fp).replace('.png', '')
    if lanes:
        title += ': avg of center mFOVs'
    fig.suptitle(title + ' (Non-Split)', fontsize=16)
    fig2.suptitle(title + ' (Non-Mixed)', fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig2.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(final_report_fp.replace('.png', '_NonSplit.png'))
    fig2.savefig(final_report_fp.replace('.png', '_NonMixed.png'))
    plt.gcf().clear()
    plt.close()

    if not lanes:
        center_fovs = np.array(center_fovs)
        center_fovs = np.mean(center_fovs, axis=0)
        f = os.path.join(os.path.dirname(final_report_fp), 'npy',
                         os.path.basename(final_report_fp).replace('.png', '.npy'))
        np.save(f, center_fovs)

        center_fovs_thres = np.array(center_fovs_thres)
        center_fovs_thres = np.mean(center_fovs_thres, axis=0)
        np.save(f.replace('_avgCBI_Hist.npy', '_Final_Thresholds.npy'), center_fovs_thres)
    return


def plot_split_cbi_dist(parent_reports, children_reports, fovs, final_report_fp, lanes=False):
    center_fovs = []
    directions = ['Horizontal', 'Vertical', 'Diagonal', 'Multiple']
    fig, axes = plt.subplots(3, 2, figsize=(12, 8), sharex=True, sharey=True)
    if lanes:
        fig, axes = plt.subplots(2, 2, figsize=(12, 6), sharex=True, sharey=True)
    for ax, child_report_fp, parent_report_fp, fov in zip(axes.flatten(), children_reports, parent_reports, fovs):
        if lanes:
            fov = os.path.basename(child_report_fp).split('_')[1]

        data1 = np.load(child_report_fp)
        if parent_report_fp is not None:
            data2 = np.load(parent_report_fp)
        else:
            data2 = data1[4:]
            data1 = data1[:4]

        if fov not in ['C001R006', 'C006R060']:
            center_fovs.append(np.vstack([data1, data2]))

        xs = np.linspace(0, 4, 500)
        for i in range(data1.shape[0]):
            ax.plot(xs, data1[i], label='%s (Children)' % directions[i], linestyle='dashed', alpha=0.6)
            ax.plot(xs, data2[i], label='%s (Parents)' % directions[i], alpha=0.6)
        ax.set_xlabel('CBI')
        ax.set_ylabel('Density')
        ax.tick_params(labelsize=12)
        ax.set_title(fov, fontsize=14)
    axes[0, 1].legend(bbox_to_anchor=(0.96, 0.9), prop={'size': 12})
    title = '%s' % os.path.basename(final_report_fp).replace('.png', '')
    if lanes:
        title += ': avg of center mFOVs'
    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(final_report_fp)
    plt.gcf().clear()
    plt.close()

    if not lanes:
        center_fovs = np.array(center_fovs)
        center_fovs = np.mean(center_fovs, axis=0)
        f = os.path.join(os.path.dirname(final_report_fp), 'npy',
                         os.path.basename(final_report_fp).replace('.png', '.npy'))
        np.save(f, center_fovs)
    return


def get_parent_report_names(f):
    lane_dp, occupancy_fn = os.path.split(os.path.dirname(f))
    slide_dp, lane = os.path.split(lane_dp)
    slide, lane, occ, analysis, cycles = os.path.basename(f).split('_')[:5]
    prefix = '%s_%s_%s_%s' % (slide, occ, analysis, cycles)

    slide_dp = os.path.join(slide_dp, occupancy_fn)
    if not os.path.isdir(slide_dp):
        os.makedirs(slide_dp)

    return slide_dp, occupancy_fn, prefix


def consolidate_lane_reports(slide_dp, occupancy_fn, prefix):
    report_names = ['ACGT_splits', 'CBI_Quartiles', 'Center2x2_Summary', 'Cluster_Mixed_Summary', 'Mixed_Results',
                    'Size_Results', 'SNR1_Quartiles', 'SNR2_Quartiles', 'Split_Results', 'Summary']
    slide = prefix.split('_')[0]
    cycles = prefix.split('_')[-1]
    for report in report_names:
        metrics = []
        lanes, dfs, avgs = [], [], []
        for lane in ['L01', 'L02', 'L03', 'L04']:
            lane_dp = os.path.join(os.path.split(slide_dp)[0], lane, occupancy_fn)
            f = os.path.join(lane_dp, '%s_%s_Occupancy_Analysis_%s_%s.csv' % (slide, lane, cycles, report))
            if os.path.exists(f):
                idx = [0, 1, 2] if report == 'ACGT_splits' else 0
                df = pd.read_csv(f, index_col=idx)
                metrics = df.index
                if 'AVG' in df.columns.tolist():
                    avgs.append(df['AVG'].values)
                    # assuming AVG column is always first?
                    df = df[df.columns.tolist()[1:]]
                dfs.append(df)
                lanes.append(lane)
        if len(lanes) == 0:
            continue

        if report == 'ACGT_splits':
            df = pd.concat(dfs)
            df = df.sort_index(level=[1, 0])
            df.to_csv(os.path.join(slide_dp, '%s_ACGT_splits.csv' % prefix))
        else:
            writer = pd.ExcelWriter(os.path.join(slide_dp, '%s_%s.xlsx' % (prefix, report)), engine='xlsxwriter')
            workbook = writer.book
            lane_border = workbook.add_format()
            lane_border.set_right()

            c = 0
            index = True
            if len(avgs) > 0:
                avgs = np.array(avgs).T
                df_avgs = pd.DataFrame(avgs, index=metrics, columns=['AVG'] * len(lanes))
                df_avgs.to_excel(writer, sheet_name=report, startcol=c, startrow=1, index=index)
                c += len(lanes) + 1
                worksheet = writer.sheets[report]
                worksheet.set_column(c - 1, c - 1, None, lane_border)
                for i in range(len(lanes)):
                    worksheet.write(0, i + 1, lanes[i])
                index = False

            for i, lane in enumerate(lanes):
                df = dfs[i]
                df.to_excel(writer, sheet_name=report, startcol=c, startrow=1, index=index)
                index = False
                c += df.shape[1]
                if i == 0 and len(avgs) == 0:
                    c += 1
                worksheet = writer.sheets[report]
                worksheet.set_column(c - 1, c - 1, None, lane_border)
                for col_num in range(df.shape[1]):
                    cc = c - df.shape[1]
                    worksheet.write(0, col_num + cc, lane)
            worksheet = writer.sheets[report]
            worksheet.freeze_panes(0, 1)
            worksheet.set_column(0, 0, 30)
            if report == 'Summary' or report == 'Center2x2_Summary':
                worksheet.conditional_format(2, 1, 2, c, {'type': '3_color_scale'})
                for col in [6, 11, 15, 19]:
                    worksheet.conditional_format(col, 1, col, c, {'type': '3_color_scale',
                                                                  'min_color': '#63BE7B',
                                                                  'mid_color': '#FFEB84',
                                                                  'max_color': '#F8696B'})
                if 'HighSHI  (%ofTotal)' in df.index.tolist():
                    worksheet.conditional_format(21, 1, col, 21, {'type': '3_color_scale',
                                                                  'min_color': '#63BE7B',
                                                                  'mid_color': '#FFEB84',
                                                                  'max_color': '#F8696B'})

            writer.save()
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

    occupancy_parameters, consolidate_lanes = populate_default_parameters(occupancy_parameters)

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

    final_report_fps = consolidate_reports(occupancy_results)
    consolidate_split_base_comp_reports(occupancy_results)
    consolidate_fov_plots(occupancy_results, occupancy_parameters['output_dp'])

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
    if consolidate_lanes:
        time.sleep(10)
        f = final_report_fps[0]
        completed_lanes = [os.path.exists(f.replace(occupancy_parameters['lane'], 'L01')),
                           os.path.exists(f.replace(occupancy_parameters['lane'], 'L02')),
                           os.path.exists(f.replace(occupancy_parameters['lane'], 'L03')),
                           os.path.exists(f.replace(occupancy_parameters['lane'], 'L04'))]
        if np.sum(completed_lanes) > 1:
            slide_output_dp, occupancy_fn, prefix = get_parent_report_names(f)
            consolidate_lane_reports(slide_output_dp, occupancy_fn, prefix)
            consolidate_lane_plots(slide_output_dp, occupancy_fn, prefix)

    end_time = datetime.datetime.now()
    ela_time = end_time - start_time
    logger.info('Occupancy analysis complete! (%s)' % ela_time)
    return


if __name__ == '__main__':
    main(sys.argv[1:])