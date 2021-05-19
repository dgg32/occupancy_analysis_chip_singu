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


def combine_results(dir1):
    to_run_list = ['python', 'Combine_Reports_Single_Cycle.py']
    for slide in os.listdir(dir1):
        if 'JH' in slide:
            to_run_list.append(os.path.join(dir1, slide))
    to_run_list.append('DryLoad_5-18-21')
    # for slide in os.listdir(dir2):
    #     to_run_list.append(os.path.join(dir2, slide))
    subprocess.Popen(to_run_list)


if __name__ == '__main__':
    # run_directory('//prod/pv-10/home/ajorjorian/HighDensityIntensities/DryloadEDTA')
    # run_directory('//prod/pv-10/home/ajorjorian/HighDensityIntensities/DryloadPotassium')
    # run_directory('//prod/pv-10/home/ajorjorian/HighDensityIntensities/JHHD05152021')
    combine_results('//prod/pv-10/home/ajorjorian/HD_Occupancy')