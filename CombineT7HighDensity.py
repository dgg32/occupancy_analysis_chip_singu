import sys
import os
import glob
import subprocess


def run_directory(dir):
    for slide in os.listdir(dir):
        dp = os.path.join(dir, slide, 'L01')
        op = os.path.join('//prod/pv-10/home/ajorjorian/HD_Occupancy', slide)
        tp = os.path.join('//prod/pv-10/home/ajorjorian/HD_Temp', slide)
        to_run_list = ['python', 'occupancy_wrapper.py', '-d', dp, '-s',
                       slide, '-l', 'L01', '-t', tp, '-o',
                       op, '-c', '1', '-r', '1' ]
        a = subprocess.Popen(to_run_list)
    a.wait()


def combine_results(dir1, dir2):
    to_run_list = ['python', 'Combine_Reports_Single_Cycle.py']
    for slide in ['E100021815', 'E100021821', 'E100021822', ]:
        to_run_list.append(os.path.join(dir1, slide, 'L01', 'Occupancy_Reports_C01-C10_v4.4.0A'))
    to_run_list.append('T7_5-13-21')
    # for slide in os.listdir(dir2):
    #     to_run_list.append(os.path.join(dir2, slide))
    subprocess.Popen(to_run_list)


if __name__ == '__main__':
    # run_directory('//prod/pv-10/home/ajorjorian/HighDensityIntensities/DryloadEDTA')
    # run_directory('//prod/pv-10/home/ajorjorian/HighDensityIntensities/DryloadPotassium')
    # run_directory('//prod/pv-10/home/ajorjorian/HighDensityIntensities/DryLoadLGBMg')
    combine_results('//prod/hustor-01/zebra/TDI_V40.1/dev_cdbe366_pr_1/Results/', '//prod/hustor-01/zebra/ZebraV2.1/V2_eb22bea_CPU_cap_integ-0.6-0.8-0.7-1.2/Results/')
    # combine_results('//prod/hustor-01/zebra/ZebraV2.1/V2_eb22bea_CPU_cap_integ-0.6-0.8-0.7-1.2/Results/')