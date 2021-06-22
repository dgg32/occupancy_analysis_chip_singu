import calReader
import os
import sys
import occuint2npy
import pos2neighbor
import glob
import numpy as np
import pandas as pd
split_rate_dict = {}
direction_dict = {"horiz": [3, 4], 'vert': [1, 6], '': [0, 1, 2, 3, 4, 5, 6, 7]}


def get_fovs(data_folder):
    fovs = []
    int_files = glob.glob(os.path.join(data_folder, 'finInts', 'S001', '*.int'))
    for int_file in int_files:
        fovs.append(os.path.basename(int_file).split('.')[0])
    return fovs


def get_positional_index(data_folder, temp_folder, fov):
    pos_index = os.path.join(temp_folder, '%s.posiIndex.txt' % fov)
    if not os.path.isfile(pos_index):
        a = occuint2npy.Int2npy(data_folder, fov, 1, output_dp=temp_folder)
        a.run()
    return pos_index


def get_neighbor_arrs(pos_index, temp_folder, fov):
    if not os.path.isfile(os.path.join(temp_folder, '%s_neighbors.npy' % fov)):
        pos2neighbor.main(pos_index, os.path.join(temp_folder, '%s_coords.npy' % fov),
                          os.path.join(temp_folder, '%s_neighbors.npy' % fov), False, True, False, False)
    neighbor_arr = np.load(os.path.join(temp_folder, '%s_neighbors.npy' % fov))
    return neighbor_arr


def get_call_arr(data_folder, fov):
    cal_file = os.path.join(data_folder, 'calFile', '%s.cal' % fov)
    cal_obj = calReader.Cal()
    cal_obj.load(cal_file,V40=True)
    call_dict = cal_obj.basesDigit
    qual_dict = cal_obj.qual
    call_arr = np.zeros([len(call_dict[1]), 10])
    qual_arr = np.zeros([len(qual_dict[1]), 10])
    for i in range(1,11):
        call_arr[:, i-1] = call_dict[i]
        qual_arr[:, i-1] = qual_dict[i]
    return call_arr, qual_arr

def neighbor_to_single_counting(neighbor_arr):
    sampling = np.random.choice([False, True], neighbor_arr.shape[0], p=[0.8, 0.2])
    return neighbor_arr[sampling]

def get_split_rates(input_list):
    j=0
    out_frame = pd.DataFrame(columns=['slide', 'Fov', 'PolyG', 'Split_Rate', 'Horiz', 'Vert'])
    for a in input_list:
        data_folder = a[0]
        temp_folder = a[1]
        # if not os.path.isdir(temp_folder):
        os.makedirs(temp_folder,exist_ok=True)
        slide = '_'.join(os.path.basename(os.path.dirname(data_folder)).split('_')[1:3])
        fovs = get_fovs(data_folder)
        for fov in fovs:
            pos_index = get_positional_index(data_folder, temp_folder, fov)
            neighbor_arr = get_neighbor_arrs(pos_index, temp_folder, fov)
            # neighbor_arr = neighbor_to_single_counting(neighbor_arr)
            call_arr, qual_arr = get_call_arr(data_folder, fov)
            poly_g = (((call_arr == 71).astype(int).sum(axis=1)) >= 8)[neighbor_arr[:, 0]]
            not_poly_g = np.logical_not(poly_g)
            print(call_arr)
            print(call_arr.shape)
            print(neighbor_arr)
            print(neighbor_arr.shape)
            print(call_arr[neighbor_arr].shape)
            equivalancy = np.equal(call_arr[neighbor_arr][:, 0, :].reshape(call_arr[neighbor_arr][:, 0, :].shape[0], 1,
                                                                           call_arr[neighbor_arr][:, 0, :].shape[1]),
                                   call_arr[neighbor_arr][:, 1:, :])
            equivalancy = equivalancy.sum(axis=2) > 7
            print(equivalancy.shape)
            equiv_bool = equivalancy.sum(axis=1) > 0
            split_equivalancy_q_scores = qual_arr[equiv_bool].mean()
            non_split_equivalancy_q_scores = qual_arr[np.logical_not(equiv_bool)].mean()
            equivalancy_poly_g = equivalancy[poly_g]
            split_equivalancy_q_scores_poly_g = qual_arr[poly_g][equiv_bool[poly_g]].mean()
            non_split_equivalancy_q_scores_poly_g = qual_arr[poly_g][np.logical_not(equiv_bool[poly_g])].mean()
            equivalancy_not_poly_g = equivalancy[not_poly_g]
            split_equivalancy_q_scores_not_poly_g = qual_arr[not_poly_g][equiv_bool[not_poly_g]].mean()
            non_split_equivalancy_q_scores_not_poly_g = qual_arr[not_poly_g][np.logical_not(equiv_bool[not_poly_g])].mean()
            print(equivalancy)
            out_frame.loc[j, 'slide'] = slide
            out_frame.loc[j, 'Fov'] = fov
            out_frame.loc[j, 'PolyG'] = 'Any'
            out_frame.loc[j, 'Split_Rate'] = float((equivalancy.sum(axis=1) > 0).sum())/equivalancy.shape[0]

            out_frame.loc[j, 'Horiz'] = float(((equivalancy[:, 3] + equivalancy[:, 4]) > 0).sum()) / equivalancy.shape[0]
            out_frame.loc[j, 'Vertz'] = float(((equivalancy[:, 1] + equivalancy[:, 6]) > 0).sum()) / equivalancy.shape[0]
            out_frame.loc[j, 'SplitQual'] = split_equivalancy_q_scores
            out_frame.loc[j, 'NonSplitQual'] = non_split_equivalancy_q_scores
            j += 1
            out_frame.loc[j, 'slide'] = slide
            out_frame.loc[j, 'Fov'] = fov
            out_frame.loc[j, 'PolyG'] = 'None'
            out_frame.loc[j, 'Split_Rate'] = float((equivalancy_not_poly_g.sum(axis=1) > 0).sum()) / equivalancy_not_poly_g.shape[0]

            out_frame.loc[j, 'Horiz'] = float(((equivalancy_not_poly_g[:, 3] + equivalancy_not_poly_g[:, 4]) > 0).sum()) / equivalancy_not_poly_g.shape[
                0]
            out_frame.loc[j, 'Vertz'] = float(((equivalancy_not_poly_g[:, 1] + equivalancy_not_poly_g[:, 6]) > 0).sum()) / equivalancy_not_poly_g.shape[
                0]
            out_frame.loc[j, 'SplitQual'] = split_equivalancy_q_scores_poly_g
            out_frame.loc[j, 'NonSplitQual'] = non_split_equivalancy_q_scores_poly_g
            j += 1
            out_frame.loc[j, 'slide'] = slide
            out_frame.loc[j, 'Fov'] = fov
            out_frame.loc[j, 'PolyG'] = 'Only'
            out_frame.loc[j, 'Split_Rate'] = float((equivalancy_poly_g.sum(axis=1) > 0).sum()) / equivalancy_poly_g.shape[0]

            out_frame.loc[j, 'Horiz'] = float(((equivalancy_poly_g[:, 3] + equivalancy_poly_g[:, 4]) > 0).sum()) / equivalancy_poly_g.shape[
                0]
            out_frame.loc[j, 'Vertz'] = float(((equivalancy_poly_g[:, 1] + equivalancy_poly_g[:, 6]) > 0).sum()) / equivalancy_poly_g.shape[
                0]
            out_frame.loc[j, 'SplitQual'] = split_equivalancy_q_scores_not_poly_g
            out_frame.loc[j, 'NonSplitQual'] = non_split_equivalancy_q_scores_not_poly_g
            j += 1

    out_frame.to_excel('//prod/pv-10/home/ajorjorian/T10_Occupancy.xlsx')



if __name__ == '__main__':
    get_split_rates([['//prod/pv-10/home/ajorjorian/V0.2_FP2100000084_0304/L01',
                    '//prod/pv-10/home/ajorjorian/V0.2_FP2100000084_0304_temp'],
                    ['//prod/pv-10/home/ajorjorian/V0.2_FP2100000038_0908/L01',
                     '//prod/pv-10/home/ajorjorian/V0.2_FP2100000038_0908_temp']])





