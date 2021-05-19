import os
import glob
import sys
import pandas as pd
import numpy as np

def run(dir_list, out_name):
    summary_out = pd.DataFrame()
    for i, d in enumerate(dir_list):
        print(d)
        try:
            split_res = glob.glob(os.path.join(d, '*' + 'Single_Cycle_Split_Results.csv'))[0]
            print(split_res)
            # col_name = os.path.basename(os.path.dirname(split_res))
            col_name = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(split_res))))
            print(col_name)
            a = pd.read_csv(split_res, index_col=0)
            if not i:
                single_cycle_out = pd.DataFrame(index=a.index)
                single_cycle_out['Valid'] = np.zeros(len(single_cycle_out))
                single_cycle_out['MixedMatch'] = np.zeros(len(single_cycle_out))
                single_cycle_out['Channel'] = np.zeros(len(single_cycle_out))
                single_cycle_out['Mean'] = np.zeros(len(single_cycle_out))
                single_cycle_out['SplitDirect'] = np.zeros(len(single_cycle_out))
                for i in single_cycle_out.index.values:
                    if 'Including' in i:
                        single_cycle_out.loc[i, 'MixedMatch'] = True
                    else:
                        single_cycle_out.loc[i, 'MixedMatch'] = False
                    if 'Mean' in i:
                        single_cycle_out.loc[i, 'Mean'] = True
                    else:
                        single_cycle_out.loc[i, 'Mean'] = False
                    if 'Concordant' in i:
                        single_cycle_out.loc[i, 'Channel'] = i.split(' ')[4]
                    else:
                        single_cycle_out.loc[i, 'Channel'] = False
                    if '% Called' in i:
                        single_cycle_out.loc[i, 'Channel'] = i.split(' ')[2]
                    if 'Valid' in i:
                        single_cycle_out.loc[i, 'Valid'] = True
                    else:
                        single_cycle_out.loc[i, 'Valid'] = False
                    if 'vert' in i:
                        single_cycle_out.loc[i, 'SplitDirect'] = 'Vert'
                    elif 'horiz' in i:
                        single_cycle_out.loc[i, 'SplitDirect'] = 'Horiz'
                    else:
                        single_cycle_out.loc[i, 'SplitDirect'] = False


            single_cycle_out[col_name] = a['AVG']
        except:
            pass
    single_cycle_out.drop_duplicates(inplace=True)
    single_cycle_out.to_csv('//prod/pv-10/home/ajorjorian/' + out_name + '.csv')
    for i, d in enumerate(dir_list):
        try:
            split_res = glob.glob(os.path.join(d, '*C10_Summary.csv'))[0]
            # col_name = os.path.basename(os.path.dirname(split_res))
            col_name = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(split_res))))
            a = pd.read_csv(split_res, index_col=0)
            if not i:
                summary_out = pd.DataFrame(index=a.index)

            summary_out[col_name] = a['AVG']
        except:
            pass
    summary_out.to_csv('//prod/pv-10/home/ajorjorian/' + out_name + 'Summaries.csv')

if __name__ == '__main__':
    run(sys.argv[1:-1], sys.argv[-1])