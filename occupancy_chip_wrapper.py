## whole chip calculation

import sys, os
import pathlib
import argparse

usage = '''
    Usage: specify -p, -d, -s and -o  
'''

ArgParser = argparse.ArgumentParser(usage=usage)

ArgParser.add_argument("-p", "--platform", action="store", dest="platform", default="Lite",
                        help="Data platform. Options: v1, v2, BB, Lite")
ArgParser.add_argument("-d", "--data", action="store", dest="data_dp", 
                        help="Raw data path")
ArgParser.add_argument("-l", "--lane", action="store", dest="lane", 
                        help="Name of the lane")
ArgParser.add_argument("-s", "--slide", action="store", dest="slide", 
                        help="Name of the slide")
ArgParser.add_argument("-o", "--output", action="store", dest="output_dp", 
                        help="Output path")
ArgParser.add_argument("-c", "--start", action="store", dest="cycle_start", 
                        help="Start cycle")
ArgParser.add_argument("-r", "--range", action="store", dest="cycle_range", 
                        help="Number of cycles to analyze. Used in conjunction with cycle start")

args = ArgParser.parse_args()

lane_path = os.path.join(args.data_dp, args.lane)
output_path = os.path.join(args.output_dp, args.slide, args.lane, args.platform)
temp_path = os.path.join(args.output_dp, args.slide, args.lane, args.platform, "tmp")
command = f"python /hwfssz8/MGI_BCC/USER/huangsixing/occupancy_analysis/occupancy_wrapper.py -p {args.platform} -s {args.slide} -l {args.lane} -c {args.cycle_start} -r {args.cycle_range} -o {output_path} -t {temp_path} -d {lane_path}"

#### test

p = pathlib.Path(output_path)
p.mkdir(parents=True, exist_ok=True)


output_file = os.path.join(output_path, "test.txt")

o = open(output_file, 'w+')
o.write(command)
o.close()

os.system(command)