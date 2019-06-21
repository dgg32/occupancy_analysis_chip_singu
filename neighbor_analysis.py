from __future__ import division
import glob
import numpy as np
import os
import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt

from matplotlib import gridspec
from scipy.stats import gaussian_kde
import scipy.integrate
import gzip

import logging.config
logger = logging.getLogger(__name__)
from sap_funcs import setup_logging

from sap_funcs import make_dir

import sys
from intensity_analysis import IntensityAnalysis
from intensity_analysis import label_dict

import cPickle as pickle
import traceback

import datetime

def merge_dnb_lists(lists):
    """
    Input format: list of lists of DNBs (class DNB)

    Merges all lists that share at least one common element
    ex. [[a,b,c],[b,e,f],[x,y,z]] -> [[a,b,c,e,f],[x,y,z]]

    Output format: list of lists of DNBs (class DNB)
    """
    res = []
    if lists:
        res.append(lists[0])
    merge_events = 0
    if len(lists) > 1:
        for l in lists[1:]:
            listset = set(l)
            merged = False
            for index in range(len(res)):
                rset = set(res[index])
                if len(listset & rset) != 0:
                    res[index] = list(listset | rset)
                    merged = True
                    merge_events += 1
                    break
            if not merged:
                res.append(l)
    if merge_events == 0:
        return res
    else:
        return merge_index_lists(res)

def merge_index_lists(lists):
    """
    Input format: list of lists of DNBs (class DNB)

    Merges all lists that share at least one common element
    ex. [[a,b,c],[b,e,f],[x,y,z]] -> [[a,b,c,e,f],[x,y,z]]

    Output format: list of lists of DNBs (class DNB)
    """
    res = []
    if lists:
        res.append(lists[0])
    merge_events = 0
    if len(lists) > 1:
        for l in lists[1:]:
            listset = set(l)
            merged = False
            for index in range(len(res)):
                rset = set(res[index])
                if len(listset & rset) != 0:
                    res[index] = list(listset | rset)
                    merged = True
                    merge_events += 1
                    break
            if not merged:
                res.append(l)
    if merge_events == 0:
        return res
    else:
        return merge_index_lists(res)

def area_under_curve(x, y, th=1):
    """
    Input format: x = array-like
                  y = array-like
                  th = numerical

    Calculates the proportions of the area under the xy curve from 0 to th and from th to the end

    Output format: 2 float values between 0 and 100, representing relative percentages
    """
    indices = [i for i in range(len(y)) if x[i] <= th]
    idx = np.int(indices[-1])
    less_than = 100 * scipy.integrate.simps(y[:int(idx + 1)], x[:int(idx + 1)]) / scipy.integrate.simps(y, x)
    greater_than = 100 - less_than
    return less_than, greater_than


def unpack(l):
    return [item for sublist in l for item in sublist]


def calculate_significance(p1, n1, p2, n2):
    if not n1 or not n2:
        return ''
    p_hat = (n1 * p1 + n2 * p2) / (n1 + n2)
    #print("p_hat: " + str(p_hat))
    sigma = np.sqrt(p_hat * (1 - p_hat) * (1 / n1 + 1 / n2))
    #print("sigma: " + str(sigma))
    zscore = (p1 - p2) / sigma
    #print("zscore: " + str(zscore))
    # significant level of 0.05
    if abs(zscore) > 3.291:
        return ' (!!!)'
    elif abs(zscore) > 2.576:
        return ' (!!)'
    elif abs(zscore) > 1.96:
        return ' (!)'
    else:
        return ''


class DNB:
    def __init__(self, idx):
        self.idx = idx
        self.left = self.right = self.up = self.down = self.ur = self.ul = self.dr = self.dl = None

    def get_idx(self):
        return self.idx

    def mat(self):
        return [self, self.ul, self.up, self.ur, self.left, self.right, self.dl, self.down, self.dr]

    def num_neighbors(self):
        return sum(val is not None for val in self.mat()[1:])


class NeighborAnalysis(object):
    def __init__(self, int_analysis, coords_fp='', neighbors_fp='', blocks_fp=None, fastq_fp='', output_dp='', bypass={},
                 log_dp='', log_overrides={}):
        self.start_time = datetime.datetime.now()

        self.prefix = int_analysis.prefix
        self.lane = int_analysis.lane
        self.fov = int_analysis.fov
        self.start_cycle = int_analysis.start_cycle
        self.cycle_range = int_analysis.cycle_range
        self.empty_fth = int_analysis.empty_fth
        empty_calls = np.zeros(40).astype(np.int8)
        empty_calls.shape = (1,4,10)
        self.called_signals = np.append(empty_calls, int_analysis.called_signals, axis=0)
        self.naCBI_data = int_analysis.naCBI_data
        self.label_arr = int_analysis.label_arr
        self.coords_fp = coords_fp
        self.neighbors_fp = neighbors_fp
        self.blocks_fp = blocks_fp
        self.fastq_fp = fastq_fp

        self.output_dp = output_dp if output_dp else os.path.dirname(int_analysis.int_fp)
        make_dir(self.output_dp)

        self.na_summary_fp = os.path.join(self.output_dp, '%s_Neighbor_Summary.p' % self.prefix)
        self.na_results_fp = os.path.join(self.output_dp, '%s_Neighbor_Results.p' % self.prefix)

        self.split_rates_fp = os.path.join(self.output_dp, '%s_Split_Rates.npy' % self.prefix)

        self.horiz_splits_fp = os.path.join(self.output_dp, '%s_Horizontal_Splits.p' % self.prefix)
        self.vert_splits_fp = os.path.join(self.output_dp, '%s_Vertical_Splits.p' % self.prefix)
        self.diag_splits_fp = os.path.join(self.output_dp, '%s_Diagonal_Splits.p' % self.prefix)
        self.multi_splits_fp = os.path.join(self.output_dp, '%s_Multiple_Splits.p' % self.prefix)

        self.possible_split_groups_fp = os.path.join(self.output_dp, '%s_Possible_Split_Groups.p' % self.prefix)
        self.ACGT_dist_fp = os.path.join(self.output_dp, '%s_ACGT_dist.p' % self.prefix)
        self.sequence_strings_fp = os.path.join(self.output_dp, '%s_Sequence_Strings.npy' % self.prefix)

        self.ACGT_dist_csv = os.path.join(self.output_dp, '%s_%s_%s_Occupancy_Analysis_%s_ACGT_splits.csv' % tuple(self.prefix.split('_')))
        self.split_cbi_ratio_dist_npy = os.path.join(self.output_dp, '%s_Split_CBI-Ratio_Distributions.npy' % self.prefix)
        self.parent_cbi_dist_npy = os.path.join(self.output_dp, '%s_Parents_CBI_Distributions.npy' % self.prefix)
        self.children_cbi_dist_npy = os.path.join(self.output_dp, '%s_Children_CBI_Distributions.npy' % self.prefix)
        self.bypass = bypass
        """
        self.bypass['get_possible_split_groups'] = self.bypass.pop('get_possible_split_groups', False)
        self.bypass['calculate_split_percentage'] = self.bypass.pop('calculate_split_percentage', False)
        self.bypass['plot_splits'] = self.bypass.pop('plot_splits', False)
        self.bypass['plot_multicalls'] = self.bypass.pop('plot_multicalls', False)
        self.bypass['plot_nonCBI'] = self.bypass.pop('plot_nonCBI', False)
        self.bypass['plot_chastity'] = self.bypass.pop('plot_chastity', False)
        self.bypass['plot_cbi_rank'] = self.bypass.pop('plot_cbi_rank', False)
        self.bypass['plot_cbi_thresholds'] = self.bypass.pop('plot_cbi_thresholds', False)
        """

        self.log_dp = log_dp
        self.log_overrides = log_overrides
        sub_log_fn = os.path.join(log_dp, '%s.log' % self.fov)
        sub_error_log_fn = os.path.join(log_dp, '%s_errors.log' % self.fov)
        override_dict = {'sub.log': sub_log_fn, 'sub_errors.log': sub_error_log_fn}
        override_dict.update(log_overrides)
        setup_logging(overrides=override_dict)
        logger.info('%s - Numpy version: %s' % (self.fov, np.version.version))

        logger.debug(self.__dict__)
        return

    def load_block_bool(self):
        block_bool = np.load(self.blocks_fp) if self.blocks_fp is not None else None
        logger.debug('%s - block bool DNB count: %s' % (self.fov, np.sum(block_bool)))
        return block_bool

    def load_neighbors(self):

        """
        Creates a neighbors array numpy file if it does not exist already
        neighbors_arr = np.array of lists in the format
                            [[center, upper left, up, upper right, left, right, lower left, down, lower right]
                            for each index]
        :return:
        """

        neighbors_arr = np.load(self.neighbors_fp) + 1
        num_spots = neighbors_arr.shape[0]
        return neighbors_arr, num_spots

    def match_binary_arrays(self, primary_dnb, neighbor_dnb):
        # false positive rate for 3+ multicalls is too high
        if self.label_arr[label_dict['Multicall']][neighbor_dnb-1] > 2: return False
        primary_mc = self.called_signals[primary_dnb]
        neighbor_mc = self.called_signals[neighbor_dnb]
        match_array = primary_mc & neighbor_mc
        matched_calls = np.sum(match_array, 0)
        return np.all(matched_calls)

    def expand_nonsplit_sequences(self, sequence_strings, parents_only=False):
        if parents_only:
            nonsplit = self.label_arr[label_dict['Parents']] == 0
        else:
            nonsplit = np.logical_and(self.label_arr[label_dict['Children']] == 0,
                                      self.label_arr[label_dict['Parents']] == 0)
        nonsplit_sequences = sequence_strings[np.where(nonsplit)[0]]
        return np.random.choice(nonsplit_sequences, len(sequence_strings))

    def expand_adapter_sequences(self, sequence_strings, adapter_count):
        unique_sequences = np.unique(sequence_strings)
        adapter_sample = np.asarray([unique_sequences[0]] * adapter_count)
        unique_sample = np.random.choice(unique_sequences[1:], len(sequence_strings) - adapter_count)
        sequences = np.concatenate((adapter_sample, unique_sample))
        np.random.shuffle(sequences)
        return sequences

    def shuffle_sequences(self, sequence_strings):
        sequences = sequence_strings.copy()
        np.random.shuffle(sequences)
        return sequences

    def expand_lowdiversity_sequences(self, sequence_strings, average_count):
        unique_sequences = np.unique(sequence_strings)
        unique_cap = int((len(sequence_strings) / float(average_count)) + 0.5)
        return np.random.choice(unique_sequences[:unique_cap], len(sequence_strings))

    def get_expanded_split_groups(self, sequence_strings):
        sequences = {}
        for i, decamer in enumerate(sequence_strings):
            if decamer not in sequences:
                sequences[decamer] = []
            sequences[decamer].append(i)
        #return (sequences[key] for key in sequences.keys() if len(sequences[key]) > 1)
        possible_split_groups = [sequences[key] for key in sequences.keys() if len(sequences[key]) > 1]
        logger.info('%s - Sequence count: %s' % (self.fov, len(possible_split_groups)))
        return possible_split_groups

    def get_possible_split_groups(self, mixed_indices):
        """
        Input format: positional_txt_file = path of the posIndex text file
                      folder_path = directory to save the neighbors to (FOV folder)
                      fastq_file_path = path of the read.fq file
                      start_cycle = integer 0 or 11

        Creates a neighbors array numpy file if it does not exist already
        Creates a hash table with key = unique 10mer, val = array of indices where the 10mer can be found

        Output format: neighbors_arr = np.array of lists in the format
                            [[center, upper left, up, upper right, left, right, lower left, down, lower right]
                            for each index]
                       possible_split_groups = list of lists containing indices that share the same 10mer
                       num_sequences = integer value representing the total number of DNBs
                                       that don't contain an 'N' in their 10mers
        """
        # CREATE HASH TABLE {10mer:[[idx, split=False]]}
        sequences = {}
        idx = 0

        ACGT_dist = {'A': 0, 'C': 0, 'G': 0, 'T': 0}

        sequence_strings = []
        with gzip.open(self.fastq_fp, 'r') as fq:
            for i, line in enumerate(fq):
                if (i + 1) % 4 == 1:
                    idx = int(line.split('_')[-1].split('/')[0])
                if (i + 1) % 4 == 2:
                    decamer = line.rstrip()[0:10]
                    sequence_strings.append(decamer)
                    if 'N' not in decamer:
                        for base in decamer:
                            ACGT_dist[base] += 1

                        if decamer in sequences:
                            sequences[decamer].append(idx)
                        else:
                            sequences[decamer] = [idx]
        logger.info('%s - Non-N DNB Count: %s' % (self.fov, len(sequences.keys())))

        # temp?
        sequence_list = sequences.keys()
        count_list = [len(sequences[v]) for v in sequence_list]
        sequence_list = [seq for count, seq in sorted(zip(count_list, sequence_list), reverse=True)]
        count_list = sorted(count_list, reverse=True)
        logger.debug('%s - Most frequent sequences:' % self.fov)
        for i in range(10):
            logger.debug('%s - %s - %s' % (self.fov, sequence_list[i], count_list[i]))

        sequence_strings = np.asarray(sequence_strings)
        # if self.block_bool is not None: sequence_strings[self.block_bool]

        # filter neighbors to only mixed primary DNBs
        mixed_neighbors = self.neighbors_arr[mixed_indices]

        start = datetime.datetime.now()
        # generator for neighbors that match binary arrays with primary
        mixed_adjacent = ([val for val in gn if val and (val == gn[0] or self.match_binary_arrays(gn[0], val))] for
                    gn in mixed_neighbors)
        logger.info('%s - Mixed Adjacent Time: %s' % (self.fov, (datetime.datetime.now() - start)))

        start = datetime.datetime.now()
        for adj in mixed_adjacent:
            self.label_arr[label_dict['MixedSplit']][adj[0] - 1] = len(adj[1:])
            # add primary index to sequences index list if neighbor sequence is already being tracked (doesn't have N)
            # and primary index is not already associated with it
            add_count = len([sequences[sequence_strings[n-1]].append(adj[0] - 1) for n in adj[1:] if
                             sequence_strings[n-1] in sequences and adj[0] - 1 not
                             in sequences[sequence_strings[n-1]]])
            self.label_arr[label_dict['HiddenSplit']][adj[0] - 1] = add_count
        logger.info('%s - Mixed Split Time: %s' % (self.fov, (datetime.datetime.now() - start)))

        # A list of lists, each containing indices that share the same 10mers
        #possible_split_groups = (sequences[key] for key in sequences.keys() if len(sequences[key]) > 1)
        possible_split_groups = [sequences[key] for key in sequences.keys() if len(sequences[key]) > 1]
        logger.info('%s - Sequence count: %s' % (self.fov, len(possible_split_groups)))

        # start = datetime.datetime.now()
        # sequences_chararray = np.chararray(len(sequence_strings), itemsize=10)
        # sequences_chararray[:] = np.array(sequence_strings)
        # neighbor_sequences = np.chararray(self.neighbors_arr.shape, itemsize=10)
        # temp_neighbors = np.apply_along_axis(lambda r: [i if i else r[0] for i in r], 1, self.neighbors_arr) - 1
        # neighbor_sequences[:] = sequences_chararray[temp_neighbors]
        # logger.info('Pre-Plot Convoluted Base Count Time: %s' % (datetime.datetime.now() - start))
        # for base in 'ACGT':
        #     self.plot_convoluted_base_counts(base, neighbor_sequences)
        # self.plot_convoluted_gc_counts(neighbor_sequences)
        # s2 = datetime.datetime.now()
        # for base in 'ACGT':
        #     self.plot_base_counts(base, sequences_chararray)
        # self.plot_GC_counts(sequences_chararray)
        # logger.info('Non-Convoluted Plot Time: %s' % (datetime.datetime.now() - s2))
        # logger.info('Convoluted Base Count Time: %s' % (datetime.datetime.now() - start))
        return possible_split_groups, ACGT_dist, sequence_strings

    def get_coords(self, coords_fp):
        self.coords = np.load(coords_fp)
        return

    def plot_convoluted_gc_counts(self, neighbor_sequences):
        plt.style.use('dark_background')
        base_counts = np.sum(neighbor_sequences.count('C'), 1) + np.sum(neighbor_sequences.count('G'), 1)
        logger.info('%s - GC Counts:\n%s\n%s' % (self.fov, np.unique(base_counts, return_counts=True)))

        fig = plt.figure(figsize=(20, 20))
        plt.scatter(self.coords[0], self.coords[1], s=1, c=base_counts)
        x1, x2, y1, y2 = plt.axis()
        xymin = min(x1, y1)
        xymax = max(x2, y2)
        plt.axis((xymin, xymax, xymin, xymax))

        plt.gca().invert_yaxis()
        plt.legend(markerscale=30, prop={'size': 60})

        plt.tight_layout()
        plt.colorbar()
        png_path = os.path.join(self.output_dp, '%s_Convoluted_GC_Counts.png' % (self.prefix))
        try:
            fig.savefig(png_path)
        except IOError:
            fig.savefig('\\\\?\\' + png_path)
        plt.gcf().clear()
        plt.close()
        return

    def plot_convoluted_base_counts(self, base, neighbor_sequences):
        plt.style.use('dark_background')
        base_counts = np.sum(neighbor_sequences.count(base), 1)
        logger.info('%s - %s Counts:\n%s\n%s' % (self.fov, base, np.unique(base_counts, return_counts=True)))

        fig = plt.figure(figsize=(20, 20))
        plt.scatter(self.coords[0], self.coords[1], s=1, c=base_counts)
        x1, x2, y1, y2 = plt.axis()
        xymin = min(x1, y1)
        xymax = max(x2, y2)
        plt.axis((xymin, xymax, xymin, xymax))
        plt.gca().invert_yaxis()
        plt.legend(markerscale=30, prop={'size': 60})

        plt.tight_layout()
        plt.colorbar()
        png_path = os.path.join(self.output_dp, '%s_Convoluted_%s_Counts.png' % (self.prefix, base))
        try:
            fig.savefig(png_path)
        except IOError:
            fig.savefig('\\\\?\\' + png_path)
        plt.gcf().clear()
        plt.close()
        return

    def plot_GC_counts(self, sequences):
        plt.style.use('dark_background')
        base_counts = sequences.count('G') + sequences.count('C')
        logger.info('%s - GC Counts:\n%s\n%s' % (self.fov, np.unique(base_counts, return_counts=True)))

        fig = plt.figure(figsize=(20, 20))
        plt.scatter(self.coords[0], self.coords[1], s=1, c=base_counts)
        x1, x2, y1, y2 = plt.axis()
        xymin = min(x1, y1)
        xymax = max(x2, y2)
        plt.axis((xymin, xymax, xymin, xymax))
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.colorbar()
        png_path = os.path.join(self.output_dp, '%s_%s_Counts.png' % (self.prefix, 'GC'))
        try:
            fig.savefig(png_path)
        except IOError:
            fig.savefig('\\\\?\\' + png_path)
        plt.gcf().clear()
        plt.close()
        return

    def plot_base_counts(self, base, sequences):
        plt.style.use('dark_background')
        base_counts = sequences.count(base)
        logger.info('%s - %s Counts:\n%s\n%s' % (self.fov, base, np.unique(base_counts, return_counts=True)))

        fig = plt.figure(figsize=(20, 20))
        plt.scatter(self.coords[0], self.coords[1], s=1, c=base_counts)
        x1, x2, y1, y2 = plt.axis()
        xymin = min(x1, y1)
        xymax = max(x2, y2)
        plt.axis((xymin, xymax, xymin, xymax))
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.colorbar()
        png_path = os.path.join(self.output_dp, '%s_%s_Counts.png' % (self.prefix, base))
        try:
            fig.savefig(png_path)
        except IOError:
            fig.savefig('\\\\?\\' + png_path)
        plt.gcf().clear()
        plt.close()
        return

    def save_labels(self):
        labels_fp = os.path.join(self.output_dp, '%s_Labels' % self.prefix)
        np.save(labels_fp, self.label_arr)
        return

    def calculate_split_percentage(self, possible_split_groups, sequence_strings, label=True):
        """
        Input format: positional_txt_file = path of the posIndex text file
                      folder_path = directory to save the neighbors to (FOV folder)
                      fastq_file_path = path of the read.fq file
                      start_cycle = integer 0 or 11

        Uses the neighbors_arr to determine groups/members from possible_split_groups that are actually split DNBs
        Within each group of split DNBs, categorizes them by single split or multiple split
        Further categorizes the single splits into horizontal, vertical, or diagonal

        Output format: FOV = string like C002R008, etc
                       6x floats representing the percentages of total number of splits, mutli splits, horiz splits,
                            vertical splits, diagonal splits, and unconnected splits
                       4x (multi, horiz, vert, diag) arrays containing arrays/lists of groups of positional indices
                            that share the same 10mer
        """
        from intensity_analysis import get_max_multicalls

        def dnb_lookup(dnb_index, array):
            return array[np.where(array[:,0] == dnb_index)[0][0]]

        def count_neighbor_types(dnb_array, label, index_mod=0, debug=False):
            if not index_mod:
                dnb_array = np.asarray(map(DNB.get_idx, dnb_array))
            else:
                dnb_array = np.asarray(dnb_array) - 1
            # create array of CBI values from dnb_array
            cbi_values = self.naCBI_data[dnb_array]
            # count children
            children = np.sum(cbi_values < cbi_values[0])
            if label: self.label_arr[label_dict['Children']][dnb_array[0]] += children
            # parents are remaining DNbs
            if label: self.label_arr[label_dict['Parents']][dnb_array[0]] += len(dnb_array) - children - 1
            return

        horiz_count = 0
        vert_count = 0
        diag_count = 0
        multi_count = 0
        num_spatdups = 0
        num_split_dnbs = 0

        graph = {}

        horizontal_splits = []
        vertical_splits = []
        diagonal_splits = []
        multiple_splits = []

        group_time = datetime.timedelta(0)
        adj_time = datetime.timedelta(0)
        # For each group of same 10mers, create an array of indices that are direct neighbors
        for group in possible_split_groups:
            group_len = len(group)
            if group_len < 1000:
                start = datetime.datetime.now()
                adjacent = []
                for j in group:
                    graph[j] = DNB(j)
                done = 0
                for i in group:
                    done += 1
                    curr = graph[i]
                    if i >= 0 and i < len(self.neighbors_arr):
                        n = self.neighbors_arr[i] - 1

                        if n[1] in group:
                            curr.ul = graph[n[1]]
                            # graph[n[1]].dr = curr
                            curr.diag = True
                            # graph[n[1]].diag = True
                        if n[2] in group:
                            curr.up = graph[n[2]]
                            # graph[n[2]].down = curr
                            curr.vert = True
                            # graph[n[2]].vert = True
                        if n[3] in group:
                            curr.ur = graph[n[3]]
                            # graph[n[3]].dl = curr
                            curr.diag = True
                            # graph[n[3]].diag = True
                        if n[4] in group:
                            curr.left = graph[n[4]]
                            # graph[n[4]].right = curr
                            curr.horiz = True
                            # graph[n[4]].horiz = True
                        if n[5] in group:
                            curr.right = graph[n[5]]
                            # graph[n[5]].left = curr
                            curr.horiz = True
                            # graph[n[5]].horiz = True
                        if n[6] in group:
                            curr.dl = graph[n[6]]
                            # graph[n[6]].ur = curr
                            curr.diag = True
                            # graph[n[6]].diag = True
                        if n[7] in group:
                            curr.down = graph[n[7]]
                            # graph[n[7]].up = curr
                            curr.vert = True
                            # graph[n[7]].vert = True
                        if n[8] in group:
                            curr.dr = graph[n[8]]
                            # graph[n[8]].ul = curr
                            curr.diag = True
                            # graph[n[8]].diag = True

                        if curr.num_neighbors() > 0:
                            nb = curr.mat()
                            nb = [val for val in nb if val is not None]
                            adjacent.append(nb)
                # Currently adjacent is a list of lists with double counting of the DNBs
                map(count_neighbor_types, adjacent, [label]*len(adjacent))
                adjacent = merge_dnb_lists(adjacent)
                stop = datetime.datetime.now()
                time_diff = stop - start
                group_time += time_diff

                # Within each group of split DNBs, one is declared the "seed" and the rest are the extraneous mirages
                # To properly declare the seed, check CBIs of each index. The index with the highest CBI should be the seed.
                start = datetime.datetime.now()
                for adjgroup in adjacent:
                    if len(adjgroup) > 0:
                        length = len(adjgroup)
                        assert length > 1, "Group size less than 2"
                        num_split_dnbs += length - 1
                        try:
                            seed = max(adjgroup, key=lambda x: self.naCBI_data[x.idx])
                        except:
                            logger.warning(traceback.format_exc())
                            seed = adjgroup[0]
                        group_sequences = sequence_strings[[i.idx for i in adjgroup]]
                        sequence_counts = np.unique(group_sequences, return_counts=True)[1]
                        num_spatdups += sum(sequence_counts - 1)
                        """
                        if len(sequence_counts) > 1 and max(sequence_counts) > 2:
                            print 'Multiple Sequences!'
                            for c, i in enumerate(adjgroup):
                                print 'DNB %02d [%s] (%s call) ' % (c, i.idx, self.label_arr[label_dict['Multicall']][i.idx])
                                print self.neighbors_arr[i.idx]-1
                                print sequence_strings[i.idx]
                                print self.called_signals[i.idx+1]
                        mc_counts = np.asarray([self.label_arr[label_dict['Multicall']][i.idx] for i in adjgroup])
                        call_counts, cc_counts = np.unique(mc_counts, return_counts=True)
                        count_dict = dict(zip(call_counts, cc_counts))
                        if sum([v for k,v in count_dict.items() if k > 1]) / sum(count_dict.values()) > 0.5:
                            print count_dict
                            if count_dict == {2: 2}:
                                for c, i in enumerate(adjgroup):
                                    print 'DNB %02d [%s] (%s call) ' % (c, i.idx, self.label_arr[label_dict['Multicall']][i.idx])
                                    print self.neighbors_arr[i.idx] - 1
                                    print sequence_strings[i.idx]
                                    print self.called_signals[i.idx + 1]
                                mc2_combined = self.called_signals[adjgroup[0].idx + 1] & self.called_signals[adjgroup[1].idx + 1]
                                print get_max_multicalls(np.sum(mc2_combined, 0), 6)
                                print mc2_combined
                        """
                        if length > 2:
                            multi_count += length - 1
                            # seed = max(adjgroup, key=lambda x: x.num_neighbors())

                            if seed.left:
                                horiz_count += 1
                            if seed.right:
                                horiz_count += 1
                            if seed.up:
                                vert_count += 1
                            if seed.down:
                                vert_count += 1
                            if seed.ul:
                                diag_count += 1
                            if seed.ur:
                                diag_count += 1
                            if seed.dl:
                                diag_count += 1
                            if seed.dr:
                                diag_count += 1

                            temp = [val.idx for val in adjgroup]
                            multiple_splits.append(temp)

                        elif length == 2:
                            """
                            call_counts = [self.label_arr[label_dict['Multicall']][i.idx] for i in adjgroup]
                            if any([cc > 1 for cc in call_counts]):
                                
                                print 'Mixed Split!'
                                for c, i in enumerate(adjgroup):
                                    print 'DNB %02d [%s] (%s call) ' % (c, i.idx, self.label_arr[label_dict['Multicall']][i.idx])
                                    print self.neighbors_arr[i.idx] - 1
                                    print sequence_strings[i.idx]
                                    print self.called_signals[i.idx + 1]
                            """
                            if seed.left or seed.right:
                                horiz_count += 1
                                if seed.left:
                                    horizontal_splits.append([seed.left.idx, seed.idx])
                                else:
                                    horizontal_splits.append([seed.idx, seed.right.idx])
                            if seed.up or seed.down:
                                vert_count += 1
                                if seed.up:
                                    vertical_splits.append([seed.up.idx, seed.idx])
                                else:
                                    vertical_splits.append([seed.idx, seed.down.idx])
                            if seed.ur or seed.ul or seed.dl or seed.dr:
                                diag_count += 1
                                temp = [seed.ur, seed.ul, seed.dl, seed.dr]
                                temp = [val for val in temp if val is not None]
                                if len(temp) == 1:
                                    diagonal_splits.append([seed.idx, temp[0].idx])

                        # labeling de facto parent according to number of children in family as a whole
                        if label:
                            self.label_arr[label_dict['FamilySize']][seed.idx] += length - 1
                stop = datetime.datetime.now()
                time_diff = stop - start
                adj_time += time_diff
            else:
                logger.warning('%s - Large group found: %s' % (self.fov, group_len))
                logger.info('%s - Processing group with NumPy...' % self.fov)
                start = datetime.datetime.now()
                group = np.asarray(group) + 1
                group_mask = np.isin(self.neighbors_arr[:, 0], group)
                group_neighbors = self.neighbors_arr[group_mask]
                nb_only = np.vectorize(lambda t: t if t in group else 0)
                group_neighbors = nb_only(group_neighbors)
                adjacent = [[val for val in gn if val] for gn in group_neighbors if any(gn[1:])]
                map(count_neighbor_types, adjacent, [label]*len(adjacent), [1]*len(adjacent))
                # Currently adjacent is a list of lists with double counting of the DNBs
                adjacent = merge_index_lists(adjacent)
                stop = datetime.datetime.now()
                time_diff = stop - start
                group_time += time_diff

                # Within each group of split DNBs, one is declared the "seed" and the rest are the extraneous mirages
                # To properly declare the seed, check CBIs of each index. The index with the highest CBI should be the seed.
                start = datetime.datetime.now()

                for adjgroup in adjacent:
                    if len(adjgroup) > 0:
                        # print adjgroup
                        length = len(adjgroup)
                        assert length > 1, "Group size less than 2"
                        num_split_dnbs += length - 1
                        # print(num_split_dnbs)
                        adj_mask = np.isin(group_neighbors[:, 0], adjgroup)
                        adj_array = group_neighbors[adj_mask]
                        try:
                            seed_id = max(adjgroup, key=lambda x: self.naCBI_data[x - 1])
                        except:
                            logger.warning(traceback.format_exc())
                            seed_id = adjgroup[0]
                        group_sequences = sequence_strings[[i - 1 for i in adjgroup]]
                        sequence_counts = np.unique(group_sequences, return_counts=True)[1]
                        num_spatdups += sum(sequence_counts - 1)
                        """
                        if len(sequence_counts) > 1 and max(sequence_counts) > 2:
                            print 'Multiple Sequences!'
                            for c, i in enumerate(adjgroup):
                                print 'DNB %02d [%s] (%s call) ' % (c, i.idx, self.label_arr[label_dict['Multicall']][i-1])
                                print self.neighbors_arr[i-1]-1
                                print sequence_strings[i-1]
                                print self.called_signals[i]
                        """
                        seed_nbs = dnb_lookup(seed_id, adj_array)[1:]
                        if length > 2:
                            multi_count += length - 1
                            """
                            # original directions
                            if seed.left: #3
                                horiz_count += 1
                            if seed.right: #4
                                horiz_count += 1
                            if seed.up: #1
                                vert_count += 1
                            if seed.down: #6
                                vert_count += 1
                            if seed.ul: #0
                                diag_count += 1
                            if seed.ur: #2
                                diag_count += 1
                            if seed.dl: #5
                                diag_count += 1
                            if seed.dr: #7
                                diag_count += 1
                            """
                            if seed_nbs[3]:
                                horiz_count += 1
                            if seed_nbs[4]:
                                horiz_count += 1
                            if seed_nbs[1]:
                                vert_count += 1
                            if seed_nbs[6]:
                                vert_count += 1
                            if seed_nbs[0]:
                                diag_count += 1
                            if seed_nbs[2]:
                                diag_count += 1
                            if seed_nbs[5]:
                                diag_count += 1
                            if seed_nbs[7]:
                                diag_count += 1

                            temp = [val - 1 for val in adjgroup]
                            multiple_splits.append(temp)

                        elif length == 2:
                            if seed_nbs[3] or seed_nbs[4]:
                                horiz_count += 1
                                if seed_nbs[3]:
                                    horizontal_splits.append([seed_nbs[3] - 1, seed_id - 1])
                                else:
                                    horizontal_splits.append([seed_id - 1, seed_nbs[4] - 1])
                            if seed_nbs[1] or seed_nbs[6]:
                                vert_count += 1
                                if seed_nbs[1]:
                                    vertical_splits.append([seed_nbs[1] - 1, seed_id - 1])
                                else:
                                    vertical_splits.append([seed_id - 1, seed_nbs[6] - 1])
                            if seed_nbs[2] or seed_nbs[0] or seed_nbs[5] or seed_nbs[7]:
                                diag_count += 1
                                temp = [seed_nbs[2], seed_nbs[0], seed_nbs[5], seed_nbs[7]]
                                temp = [val - 1 for val in temp if val]
                                if len(temp) == 1:
                                    diagonal_splits.append([seed_id - 1, temp[0]])

                        # labeling de facto parent according to number of children in family as a whole
                        if label:
                            self.label_arr[label_dict['FamilySize']][seed_id - 1] += length - 1
                stop = datetime.datetime.now()
                time_diff = stop - start
                adj_time += time_diff

        logger.info('%s - group_time: %s' % (self.fov, group_time))
        logger.info('%s - adj_time: %s' % (self.fov, adj_time))

        self.save_labels()

        logger.info('%s - num_split_dnbs: %s' % (self.fov, num_split_dnbs))
        logger.info('%s - num_spatdups: %s' % (self.fov, num_spatdups))
        logger.info('%s - num_spots: %s' % (self.fov, self.num_spots))

        return 100. * horiz_count / num_split_dnbs if num_split_dnbs else 'NA', \
               100. * vert_count / num_split_dnbs if num_split_dnbs else 'NA', \
               100. * diag_count / num_split_dnbs if num_split_dnbs else 'NA', \
               100. * multi_count / num_split_dnbs if num_split_dnbs else 'NA', \
               100. * num_spatdups / self.num_spots, \
               horizontal_splits, vertical_splits, diagonal_splits, multiple_splits

    def most_frequent_sequence(self, horiz, vert, diag, multi):
        horiz = unpack(horiz)
        vert = unpack(vert)
        diag = unpack(diag)
        multi = unpack(multi)

        idx = list(set(unpack([horiz, vert, diag, multi])))
        ht = {}

        with gzip.open(self.fastq_fp, 'r') as fq:
            contents = fq.readlines()
            for i in idx:
                line = contents[4 * i + 1]
                decamer = line.rstrip()[0:10]
                if decamer not in ht:
                    ht[decamer] = 1
                else:
                    ht[decamer] += 1

        seqs = ht.keys()
        counts = [ht[s] for s in seqs]
        seqs = [y for x, y in sorted(zip(counts, seqs), reverse=True)]
        counts = sorted(counts, reverse=True)

        logger.info('%s - Most frequent split sequences:' % self.fov)
        for i in range(10):
            logger.info('%s - %s - %s' % (self.fov, seqs[i], counts[i]))

        # return seq, 100.*ht[seq]/len(idx)
        return seqs[0], counts[0]

    def split_CBI_arr(self, naCBI_data, arr, multiplicity, direction, mixed_indices):
        """
        Input format: naCBI_data = np.array containing the 4 signals of the average of 10 cycles for each positional index
                      arr = array-like containing arrays/lists of groups of positional indices that share the same 10mer
                      multiplicity = string, either single or multiple

        Uses arr and naCBI_data to find the signal intensities at each index that is categorized as a mirages

        Output format: list of lists containing the 4 signal intensities of each mirage category
        """

        save_path = os.path.join(self.output_dp, '%s_%s_%s.npy' % (self.prefix, multiplicity, direction))
        if not os.path.exists(save_path):
            CBI_arr = []

            if multiplicity == 'single':
                for pair in arr:
                    if np.isin(pair, mixed_indices).any(): continue
                    assert (len(pair) == 2), "Single splits must be in pairs"
                    CBI_arr.append([naCBI_data[pair[0]], naCBI_data[pair[1]]])

            elif multiplicity == 'multiple':
                for i in arr:
                    if i in mixed_indices: continue
                    CBI_arr.append(naCBI_data[i])

            CBI_arr = np.array(CBI_arr)
            np.save(save_path, CBI_arr)
        else:
            CBI_arr = np.load(save_path)
        return CBI_arr

    def calculate_single_split_CBI_ratio(self, CBI_arr, direction):
        """
        Input format: CBI_arr = list of lists containing the 4 signal intensities of each mirage category
                                from split_CBI_arr function
                      direction = string indicating horizontal, vertical, or diagonal

        Based on the direction of the split, creates a list of ratios (small CBI)/(large CBI) for each pair
        For horizontal and vertical splits, also creates a list of values corresponding to left/right or up/down

        Output format: dictionary containing the list of ratios
        """
        res = {}
        res['ratio'] = []
        if direction == 'horizontal' or direction == 'vertical':
            # res[direction] = []
            CBI_arr[CBI_arr < 0.0001] = 0.0001
            res['ratio'] = np.min(CBI_arr, axis=1) / np.max(CBI_arr, axis=1)
            res[direction] = CBI_arr[:, 0] / CBI_arr[:, 1]
        else:
            CBI_arr = CBI_arr.tolist()
            for pair in CBI_arr:
                p1 = pair[0]
                p2 = pair[1]
                if p1 > 0.0001 or p2 > 0.0001:
                    if p1 < 0.0001:
                        p1 = 0.0001
                    if p2 < 0.0001:
                        p2 = 0.0001
                    res['ratio'].append(min([p1, p2]) / max([p1, p2]))


        # res['ratio'] = []
        # if direction == 'horizontal' or direction == 'vertical':
        #     res[direction] = []
        #     for pair in CBI_arr:
        #         lu = pair[0]
        #         rd = pair[1]
        #         if lu > 0.0001 or rd > 0.0001:
        #             if lu < 0.0001:
        #                 lu = 0.0001
        #             if rd < 0.0001:
        #                 rd = 0.0001
        #             res[direction].append(lu / rd)  # left/right or up/down
        #             res['ratio'].append(min([lu, rd]) / max([lu, rd]))
        # else:
        #     for pair in CBI_arr:
        #         p1 = pair[0]
        #         p2 = pair[1]
        #         if p1 > 0.0001 or p2 > 0.0001:
        #             if p1 < 0.0001:
        #                 p1 = 0.0001
        #             if p2 < 0.0001:
        #                 p2 = 0.0001
        #             res['ratio'].append(min([p1, p2]) / max([p1, p2]))
        return res

    def plot_single_split_CBI_ratio(self, CBI_arr, direction):
        plt.rcParams.update(mpl.rcParamsDefault)
        plt.style.use('seaborn-muted')
        x = [p[1] for p in CBI_arr]
        y = [p[0] for p in CBI_arr]

        fig = plt.figure(figsize=(10, 10))
        plt.scatter(x, y, s=1, c='k', marker=',')
        x1, x2, y1, y2 = plt.axis()
        xymin = min(x1, y1)
        xymax = max(x2, y2)
        plt.axis((xymin, xymax, xymin, xymax))
        if direction == 'horizontal':
            plt.ylabel('Left CBI')
            plt.xlabel('Right CBI')
            plt.title('Horizontal Pair Splits CBI')
        elif direction == 'vertical':
            plt.ylabel('Up CBI')
            plt.xlabel('Down CBI')
            plt.title('Vertical Pair Splits CBI')
        else:
            plt.xlabel('Child CBI')
            plt.ylabel('Parent CBI')
            plt.title('Diagonal Pair Splits CBI')

        plt.tight_layout()
        png_path = os.path.join(self.output_dp, '%s_%s%s_Scatter.png' % (self.prefix,
                                                                         direction[0].upper(), direction[1:]))
        try:
            fig.savefig(png_path)
        except IOError:
            fig.savefig('\\\\?\\' + png_path)
        plt.gcf().clear()
        plt.close()
        return


    def profile_family(self, CBI_arr):
        """
        Input format: list of lists containing the 4 signal intensities of each mirage category

        Takes the highest signal of each sublist and declares the highest value as the parent and the rest as children
        Can work for single splits as well as multiple splits

        Output format: 2 lists, each one containing signal values corresponding to parent or children respectively
        """
        CBI_arr = CBI_arr.tolist()
        parents = []
        children = []
        for group in CBI_arr:
            parent = max(group)
            parents.append(parent)
            children.extend([val for val in group if val is not parent])
        return parents, children

    def count_parents(self, CBI_arr):
        """
        Input format: CBI_arr = list of lists containing the 4 signal intensities from split_CBI_arr function
                                horizontal or vertical splits ONLY

        Counts parents for left, right or up, down

        Output format: a list of ints [left, right] or [up, down]
        """
        lu = np.sum(CBI_arr[:, 0] > CBI_arr[:, 1])
        rd = len(CBI_arr) - lu

        # lu = rd = 0
        # for pair in CBI_arr:
        #     if pair[0] > pair[1]:
        #         lu += 1
        #     else:
        #         rd += 1
        return [lu, rd]

    def calculate_multi_split_CBI_ratio(self, CBI_arr):
        """
        Input format: CBI_arr = list of lists containing the 4 signal intensities of each mirage category
                                from split_CBI_arr function

        Creates a list of sum(children intensities)/parent intensity.

        Output format: List of floats
        """
        res = []

        for g in CBI_arr:
            group = [val for val in g if val > 0.0001]
            if len(group) > 0:
                gmax = max(group)
                gsum_min = sum(group) - gmax
                res.append(gsum_min / gmax)
        return res

    def get_fov_min_max(self):
        plt.scatter(self.coords[0], self.coords[1], s=1, c='k', marker=',')
        x1, x2, y1, y2 = plt.axis()
        xymin = min(x1, y1)
        xymax = max(x2, y2)
        self.xyminmax = (xymin, xymax, xymin, xymax)
        plt.gcf().clear()
        plt.close()
        return

    def plot_profile(self, arr, directions, relation):
        plt.rcParams.update(mpl.rcParamsDefault)
        plt.style.use('seaborn-muted')
        """
        Input format: arr = a list of CBI_arrs
                      direction = string indicating horizontal, vertical, diagonal, or multiple
                      save_path = directory like Z:/kchoi/Zebra_Data/....
                      start_cycle = int value, 0 or 11
                      mode = float value of the mode of the overall CBI distribution
                      id = string indicating parent or child

        Saves an image of 2x2 subplots. Each plot is a PDF of the CBI, normalized to median = 1
        Plots for each parent/child & direction combination
        For parents, calculates the proportion of CBI < 1
        For children, calculates the proportion of CBI < 0.5

        Output format: no output. Saves a figure to the directory
        """
        th = float(self.empty_fth)
        f, axarr = plt.subplots(2, 2)
        ax = [axarr[0, 0], axarr[0, 1], axarr[1, 0], axarr[1, 1]]
        x = np.linspace(0, 4, 500)
        num_empty = 0
        num_total = 0

        data = np.zeros((4, 500))
        for i in range(len(ax)):
            if len(arr[i]) > 1:
                p = gaussian_kde(arr[i])(x)
                data[i] = p
                ax[i].plot(x, p)
                ax[i].set_title(directions[i] + ' ' + relation + ': Intensity')

                if 'parent' in relation.lower():
                    less_than, greater_than = area_under_curve(x, p)
                    less_than_th, greater_than_th = area_under_curve(x, p, th)
                    ax[i].plot([1, 1], [0, p[np.argwhere(x <= 1)[-1][0]]], '--', color='red')
                    ax[i].plot([th, th], [0, p[np.argwhere(x <= th)][-1][0]], '--', color='green')
                    ax[i].annotate('< 1: {0:.3}'.format(less_than) + '%\n> 1: {0:.3}'.format(greater_than) + '%',
                                   xy=(0.95, 0.95), xycoords='axes fraction', horizontalalignment='right',
                                   verticalalignment='top')
                    ax[i].annotate('< {0:.2}'.format(th) + ' : {0:.3}'.format(less_than_th) + '%\n> {0:.2}'.format(th) + \
                                   ' : {0:.3}'.format(greater_than_th) + '%', xy=(0.95, 0.05), xycoords='axes fraction',
                                   horizontalalignment='right', verticalalignment='bottom')
                elif 'child' in relation.lower():
                    less_than, greater_than = area_under_curve(x, p, 0.5)
                    less_than_th, greater_than_th = area_under_curve(x, p, th)
                    ax[i].plot([0.5, 0.5], [0, p[np.argwhere(x <= 0.5)[-1][0]]], '--', color='red')
                    ax[i].plot([th, th], [0, p[np.argwhere(x <= th)][-1][0]], '--', color='green')
                    ax[i].annotate('< 0.5: {0:.3}'.format(less_than) + '%\n> 0.5: {0:.3}'.format(greater_than) + '%',
                                   xy=(0.95, 0.95), xycoords='axes fraction', horizontalalignment='right',
                                   verticalalignment='top')
                    ax[i].annotate('< {0:.2}'.format(th) + ' : {0:.3}'.format(less_than_th) + '%\n> {0:.2}'.format(th) + \
                                   ' : {0:.3}'.format(greater_than_th) + '%', xy=(0.95, 0.05), xycoords='axes fraction',
                                   horizontalalignment='right', verticalalignment='bottom')

                num_total += len(arr[i])
                num_empty += less_than_th / 100 * len(arr[i])

        f.suptitle('%s CBI Distributions (w/o Mixed Splits)')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        png_path = os.path.join(self.output_dp, '%s_%s_CBI_Distributions.png' % (self.prefix, relation))
        try:
            plt.savefig(png_path)
        except IOError:
            plt.savefig('\\\\?\\' + png_path)
        plt.gcf().clear()
        plt.close()

        npy_path = os.path.join(self.output_dp, '%s_%s_CBI_Distributions.npy' % (self.prefix, relation))
        data = np.array(data)
        np.save(npy_path, data)
        return num_empty, num_total

    def plot_CBI_ratios(self, horiz, vert, diag, mult, hvd):
        plt.rcParams.update(mpl.rcParamsDefault)
        plt.style.use('seaborn-muted')
        """
        Input format: horiz, vert, diag = dictionaries containing the small/large CBI ratios,
                                          and up/down or left/right for vert and horiz respectively
                      mult = list of floats
                      hvd = list containing 3 float values - the relative proportions of
                            horizontal, vertical, and diagonal splits
                      save_path = directory path like Z:/kchoi/Zebra_Data/...
                      start_cycle = integer value 0 or 11

        Creates a single figure with 7 subplots.
        (1) Bar chart displaying relative proportions of horiz:vert:diag splits
        (2) small/large ratio distribution for horiz splits
        (3) left/right ratio distribution for horiz splits
        (4) small/large ratio distribution for vert splits
        (5) up/down ratio distribution for vert splits
        (6) small/large ratio distribution for diag splits
        (7) sum(children)/parent distribution for multi splits

        Output format: None. Saves image to directory
        """
        fig = plt.figure(1)
        gridspec.GridSpec(9, 9)
        mpl.rcParams.update({'font.size': 6})

        mMs = np.zeros((12, 500))
        mM = np.linspace(0, 1, 500)
        if len(horiz['ratio']) > 1:
            # horiz small/large subplot
            horiz_mM = gaussian_kde(horiz['ratio'])
            horiz_mM = horiz_mM(mM)

            plt.subplot2grid((9, 9), (0, 3), rowspan=3, colspan=3)
            plt.title('Horizontal Split:\nSmall/Large CBI', size=8, weight='bold')
            plt.xlabel('Ratio')
            plt.ylabel('Density')
            plt.plot(mM, horiz_mM)
            mMs[0] = mM
            mMs[1] = horiz_mM

        if len(horiz['horizontal']) > 1:
            # horiz L/R subplot
            horiz_LR = gaussian_kde(horiz['horizontal'])
            LR = np.linspace(0, 4 * np.median(horiz['horizontal']), 500)
            horiz_LR = horiz_LR(LR)
            less_than, greater_than = area_under_curve(LR, horiz_LR)
            mMs[2] = LR
            mMs[3] = horiz_LR

            plt.subplot2grid((9, 9), (0, 6), rowspan=3, colspan=3)
            plt.title('Horizontal Split:\nLeft/Right CBI', size=8, weight='bold')
            plt.xlabel('Ratio')
            plt.ylabel('Density')
            plt.plot(LR, horiz_LR)
            plt.plot([1, 1], [0, horiz_LR[np.argwhere(LR <= 1)[-1][0]]], '--', color='red')
            plt.annotate('< 1: {0:.3}'.format(less_than) + '%\n> 1: {0:.3}'.format(greater_than) + '%', xy=(0.95, 0.95),
                         xycoords='axes fraction', horizontalalignment='right', verticalalignment='top')

        if len(vert['ratio']) > 1:
            # vertical small/large subplot
            vert_mM = gaussian_kde(vert['ratio'])
            vert_mM = vert_mM(mM)

            plt.subplot2grid((9, 9), (3, 3), rowspan=3, colspan=3)
            plt.title('Vertical Split:\nSmall/Large CBI', size=8, weight='bold')
            plt.xlabel('Ratio')
            plt.ylabel('Density')
            plt.plot(mM, vert_mM)
            mMs[4] = mM
            mMs[5] = vert_mM

        if len(vert['vertical']) > 1:
            # vertical U/D subplot
            vert_UD = gaussian_kde(vert['vertical'])
            UD = np.linspace(0, 4 * np.median(vert['vertical']), 500)
            vert_UD = vert_UD(UD)
            less_than, greater_than = area_under_curve(UD, vert_UD)
            mMs[6] = UD
            mMs[7] = vert_UD

            plt.subplot2grid((9, 9), (3, 6), rowspan=3, colspan=3)
            plt.title('Vertical Split:\nUp/Down CBI', size=8, weight='bold')
            plt.xlabel('Ratio')
            plt.ylabel('Density')
            plt.plot(UD, vert_UD)
            plt.plot([1, 1], [0, vert_UD[np.argwhere(UD <= 1)[-1][0]]], '--', color='red')
            plt.annotate('< 1: {0:.3}'.format(less_than) + '%\n> 1: {0:.3}'.format(greater_than) + '%', xy=(0.95, 0.95),
                         xycoords='axes fraction', horizontalalignment='right', verticalalignment='top')

        if len(diag['ratio']) > 1:
            # diagonal small/large subplot
            diag_mM = gaussian_kde(diag['ratio'])
            diag_mM = diag_mM(mM)

            plt.subplot2grid((9, 9), (6, 3), rowspan=3, colspan=3)
            plt.title('Diagonal Split:\nSmall/Large CBI', size=8, weight='bold')
            plt.xlabel('Ratio')
            plt.ylabel('Density')
            plt.plot(mM, diag_mM)
            mMs[8] = mM
            mMs[9] = diag_mM

        if len(mult) > 1:
            # multi small/large subplot
            multi_mM = gaussian_kde(mult)
            m_mM = np.linspace(0, max(mult), 500)
            multi_mM = multi_mM(m_mM)
            mMs[10] = m_mM
            mMs[11] = multi_mM

            plt.subplot2grid((9, 9), (6, 6), rowspan=3, colspan=3)
            plt.title('Multi Split:\nSum(Small)/Large CBI', size=8, weight='bold')
            plt.xlabel('Ratio')
            plt.ylabel('Density')
            plt.plot(m_mM, multi_mM)

        # Bar chart
        plt.subplot2grid((9, 9), (0, 0), rowspan=9, colspan=2)
        plt.title('Ratio of Splits\nHorizonal:Vertical:Diagonal', size=8, weight='bold')
        plt.ylabel('%')
        plt.xticks([1], ' ')
        total = sum(hvd)
        hvd = [100. * i / total for i in hvd]
        p1 = plt.bar([1], hvd[0], color='r')
        p2 = plt.bar([1], hvd[1], bottom=hvd[0], color='b')
        p3 = plt.bar([1], hvd[2], bottom=hvd[1] + hvd[0], color='g')
        x1, x2, y1, y2 = plt.axis()
        plt.axis((x1, x2, 0, 100))
        plt.legend((p1[0], p2[0], p3[0]), ('{0:.3}% Horizontal'.format(hvd[0]),
                                           '{0:.3}% Vertical'.format(hvd[1]),
                                           '{0:.3}% Diagonal'.format(hvd[2])), loc='center left',
                   bbox_to_anchor=(1, 0.5))

        fig.tight_layout()
        png_path = os.path.join(self.output_dp, '%s_Split_CBI-Ratio_Distributions.png' % self.prefix)
        try:
            fig.savefig(png_path)
        except IOError:
            fig.savefig('\\\\?\\' + png_path)
        plt.gcf().clear()
        plt.close()

        mMs = np.array(mMs)
        np.save(self.split_cbi_ratio_dist_npy, mMs)
        return

    def plot_splits(self, horiz, vert, diag, multi, tag=''):
        plt.style.use('dark_background')
        #plt.rcParams.update(mpl.rcParamsDefault)
        fig = plt.figure(figsize=(60, 60))
        plt.scatter(self.coords[0], self.coords[1], s=1, c='w', marker=',')
        x1, x2, y1, y2 = plt.axis()
        xymin = min(x1, y1)
        xymax = max(x2, y2)
        plt.axis((xymin, xymax, xymin, xymax))
        hx = []
        hy = []
        for pair in horiz:
            for p in pair:
                hx.append(self.coords[0][p])
                hy.append(self.coords[1][p])

        vx = []
        vy = []
        for pair in vert:
            for p in pair:
                vx.append(self.coords[0][p])
                vy.append(self.coords[1][p])

        dx = []
        dy = []
        for pair in diag:
            for p in pair:
                dx.append(self.coords[0][p])
                dy.append(self.coords[1][p])

        mx = []
        my = []
        for group in multi:
            for i in group:
                mx.append(self.coords[0][i])
                my.append(self.coords[1][i])

        nx = []
        ny = []
        for i in np.where(self.label_arr[label_dict['HiddenSplit']] > 0)[0]:
            nx.append(self.coords[0][i])
            ny.append(self.coords[1][i])

        # change: color multi first
        plt.scatter(mx, my, s=1, c='m', marker=',', label='multiple')
        plt.scatter(hx, hy, s=1, c='r', marker=',', label='horizontal')
        plt.scatter(vx, vy, s=1, c='b', marker=',', label='vertical')
        plt.scatter(dx, dy, s=1, c='g', marker=',', label='diagonal')
        plt.scatter(nx, ny, s=1, c='w', marker='.', label='mixed split (int ID)')
        plt.gca().invert_yaxis()
        plt.legend(markerscale=30, prop={'size': 60})

        plt.tight_layout()
        png_path = os.path.join(self.output_dp, '%s_Splits%s.png' % (self.prefix, tag))
        try:
            fig.savefig(png_path)
        except IOError:
            fig.savefig('\\\\?\\' + png_path)
        plt.gcf().clear()
        plt.close()
        return

    def plot_splits_alt(self, horiz, vert, diag, multi, tag=''):
        plt.rcParams.update(mpl.rcParamsDefault)
        fig = plt.figure(figsize=(60, 60))
        plt.scatter(self.coords[0], self.coords[1], s=1, c='k', marker=',')
        x1, x2, y1, y2 = plt.axis()
        xymin = min(x1, y1)
        xymax = max(x2, y2)
        plt.axis((xymin, xymax, xymin, xymax))
        hx = []
        hy = []
        for pair in horiz:
            for p in pair:
                hx.append(self.coords[0][p])
                hy.append(self.coords[1][p])

        vx = []
        vy = []
        for pair in vert:
            for p in pair:
                vx.append(self.coords[0][p])
                vy.append(self.coords[1][p])

        dx = []
        dy = []
        for pair in diag:
            for p in pair:
                dx.append(self.coords[0][p])
                dy.append(self.coords[1][p])

        mx = []
        my = []
        for group in multi:
            for i in group:
                mx.append(self.coords[0][i])
                my.append(self.coords[1][i])

        nx = []
        ny = []
        for i in np.where(self.label_arr[label_dict['HiddenSplit']] > 0)[0]:
            nx.append(self.coords[0][i])
            ny.append(self.coords[1][i])

        # change: color multi first
        plt.scatter(mx, my, s=1, c='m', marker=',', label='multiple')
        plt.scatter(hx, hy, s=1, c='r', marker=',', label='horizontal')
        plt.scatter(vx, vy, s=1, c='b', marker=',', label='vertical')
        plt.scatter(dx, dy, s=1, c='g', marker=',', label='diagonal')
        plt.scatter(nx, ny, s=1, c='w', marker='.', label='mixed split (int ID)')
        plt.gca().invert_yaxis()
        plt.legend(markerscale=30, prop={'size': 60})

        plt.tight_layout()
        png_path = os.path.join(self.output_dp, '%s_Splits_Alt%s.png' % (self.prefix, tag))
        try:
            fig.savefig(png_path)
        except IOError:
            fig.savefig('\\\\?\\' + png_path)
        plt.gcf().clear()
        plt.close()
        return

    def plot_cbi_thresholds(self):
        plt.style.use('dark_background')
        fig = plt.figure(figsize=(60, 60))
        plt.axis(self.xyminmax)

        mask = self.naCBI_data <= 0.25
        plt.scatter(self.coords[0][mask], self.coords[1][mask], s=1, c='r', marker=',', label='CBI<=0.25 (Empty)')

        temp_data = np.append(0, self.naCBI_data)
        neighbor_ints = np.nan_to_num(temp_data[self.neighbors_arr])
        small_ints = np.logical_and(neighbor_ints > 0.25, neighbor_ints <= 0.50)[
            np.where(np.logical_and(self.naCBI_data > 0.25, self.naCBI_data <= 0.50))[0]]
        logger.info('%s - small_ints.shape: %s' % (self.fov, str(small_ints.shape)))
        adjacent_smalls = np.sum(small_ints[:,1:], 1)
        logger.info('%s - adjacent_smalls.shape: %s' % (self.fov, str(adjacent_smalls.shape)))
        as_nb_count, as_dnb_count = np.unique(adjacent_smalls, return_counts=True)
        for i in range(len(as_nb_count)):
            logger.info('%s - %s Adjacent Smalls: %s' % (self.fov, as_nb_count[i], as_dnb_count[i]))

        colors = ['C3', 'C5', 'b', 'C4']
        ranges = [[0.25, 0.50], [0.50, 0.75], [0.75, 1.25], [1.25, 2.00]]
        sizes = ['Small', 'Med-Small', 'Medium', 'Med-Large']
        for i, rang in enumerate(ranges):
            label = '%d<CBI<=%d (%s)' % (rang[0], rang[1], sizes[i])
            mask = np.logical_and(self.naCBI_data > rang[0], self.naCBI_data <= rang[1])
            plt.scatter(self.coords[0][mask], self.coords[1][mask], s=1, c=colors[i], marker=',', label=label)

        mask = self.naCBI_data > 2.00
        plt.scatter(self.coords[0][mask], self.coords[1][mask], s=1, c='c', marker=',', label='2.00<CBI (Large)')

        mask = np.isnan(self.naCBI_data)
        plt.scatter(self.coords[0][mask], self.coords[1][mask], s=1, c='m', marker='.', label='NaN')

        plt.gca().invert_yaxis()
        plt.legend(markerscale=30, prop={'size': 60})

        plt.tight_layout()
        png_path = os.path.join(self.output_dp, '%s_CBI_Thresholds.png' % self.prefix)
        try:
            fig.savefig(png_path)
        except IOError:
            fig.savefig('\\\\?\\' + png_path)
        plt.gcf().clear()
        plt.close()
        return

    def plot_cbi_thr_random(self):
        plt.style.use('dark_background')
        #fig = plt.figure(figsize=(60, 60))
        fig = plt.figure()
        plt.scatter(self.coords[0], self.coords[1], s=1, c='k', marker=',')
        x1, x2, y1, y2 = plt.axis()
        xymin = min(x1, y1)
        xymax = max(x2, y2)
        plt.axis((xymin, xymax, xymin, xymax))

        naCBI_data = self.naCBI_data.copy()
        np.random.shuffle(naCBI_data)
        # Empty CBI<=0.25
        x0 = []
        y0 = []
        for i in np.where(naCBI_data <= 0.25)[0]:
            x0.append(self.coords[0][i])
            y0.append(self.coords[1][i])

        # Small 0.25<CBI<=0.50
        x1 = []
        y1 = []
        for i in np.where(np.logical_and(naCBI_data > 0.25, naCBI_data <= 0.50))[0]:
            x1.append(self.coords[0][i])
            y1.append(self.coords[1][i])

        temp_data = np.append(0, naCBI_data)
        neighbor_ints = np.nan_to_num(temp_data[self.neighbors_arr])
        small_ints = np.logical_and(neighbor_ints > 0.25, neighbor_ints <= 0.50)[
            np.where(np.logical_and(naCBI_data > 0.25, naCBI_data <= 0.50))[0]]
        logger.info('%s - rand small_ints.shape: %s' % (self.fov, str(small_ints.shape)))
        adjacent_smalls = np.sum(small_ints[:,1:], 1)
        logger.info('%s - rand adjacent_smalls.shape: %s' % (self.fov, str(adjacent_smalls.shape)))
        as_nb_count, as_dnb_count = np.unique(adjacent_smalls, return_counts=True)
        for i in range(len(as_nb_count)):
            logger.info('%s - %s Adjacent Smalls: %s' % (self.fov, as_nb_count[i], as_dnb_count[i]))

        # Med-Small 0.50<CBI<=0.75
        x2 = []
        y2 = []
        for i in np.where(np.logical_and(naCBI_data > 0.50, naCBI_data <= 0.75))[0]:
            x2.append(self.coords[0][i])
            y2.append(self.coords[1][i])

        # Medium 0.75<CBI<=1.25
        x3 = []
        y3 = []
        for i in np.where(np.logical_and(naCBI_data > 0.75, naCBI_data <= 1.25))[0]:
            x3.append(self.coords[0][i])
            y3.append(self.coords[1][i])

        # Med-Large 1.25<CBI<=2.00
        x4 = []
        y4 = []
        for i in np.where(np.logical_and(naCBI_data > 1.25, naCBI_data <= 2.00))[0]:
            x4.append(self.coords[0][i])
            y4.append(self.coords[1][i])

        # Large  2.00<CBI
        x5 = []
        y5 = []
        for i in np.where(naCBI_data > 2.00)[0]:
            x5.append(self.coords[0][i])
            y5.append(self.coords[1][i])

        nx = []
        ny = []
        for i in np.where(np.where(np.isnan(naCBI_data)))[0]:
            nx.append(self.coords[0][i])
            ny.append(self.coords[1][i])

        plt.scatter(x0, y0, s=1, c='m', marker=',', label='CBI<=0.25 (Empty)')
        plt.scatter(x1, y1, s=1, c='r', marker=',', label='0.25<CBI<=0.50 (Small)')
        plt.scatter(x2, y2, s=1, c='y', marker=',', label='0.50<CBI<=0.75 (Med-Small)')
        plt.scatter(x3, y3, s=1, c='k', marker='.', label='0.75<CBI<=1.25 (Medium)')
        plt.scatter(x4, y4, s=1, c='w', marker='.', label='1.25<CBI<=2.00 (Med-Large)')
        plt.scatter(x5, y5, s=1, c='y', marker='.', label='2.00<CBI (Large)')
        plt.scatter(nx, ny, s=1, c='r', marker='.', label='NaN')
        plt.gca().invert_yaxis()
        plt.legend(markerscale=30, prop={'size': 60})

        plt.tight_layout()
        png_path = os.path.join(self.output_dp, '%s_Random_CBI.png' % self.prefix)
        try:
            fig.savefig(png_path)
        except IOError:
            fig.savefig('\\\\?\\' + png_path)
        plt.gcf().clear()
        plt.close()
        return

    def plot_cbi_rank(self):
        plt.style.use('dark_background')
        fig = plt.figure(figsize=(60, 60))
        plt.axis(self.xyminmax)

        mask = self.label_arr[label_dict['PercCBI']] < 0
        plt.scatter(self.coords[0][mask], self.coords[1][mask], s=1, c='r', marker=',', label='DNB<0')

        colors = ['C3', 'c', 'C4', 'b']
        ranges = [[0, 25], [25, 50], [50, 75], [75, 100]]
        for i, rang in enumerate(ranges):
            label = '%d<DNB<=%d' % (rang[0], rang[1])
            mask = np.logical_and(self.label_arr[label_dict['PercCBI']] > rang[0],
                                  self.label_arr[label_dict['PercCBI']] <= rang[1])
            plt.scatter(self.coords[0][mask], self.coords[1][mask], s=1, c=colors[i], marker=',', label=label)
            logger.debug('%s - %s : %s' % (self.fov, label, np.sum(mask)))

        mask = self.label_arr[label_dict['PercCBI']] > 100
        plt.scatter(self.coords[0][mask], self.coords[1][mask], s=1, c='w', marker=',', label='100<DNB')

        mask = np.isnan(self.label_arr[label_dict['PercCBI']])
        plt.scatter(self.coords[0][mask], self.coords[1][mask], s=1, c='m', marker='.', label='NaN')

        plt.gca().invert_yaxis()
        plt.legend(markerscale=30, prop={'size': 60})

        plt.tight_layout()
        png_path = os.path.join(self.output_dp, '%s_CBI_Rank.png' % self.prefix)
        try:
            fig.savefig(png_path)
        except IOError:
            fig.savefig('\\\\?\\' + png_path)
        plt.gcf().clear()
        plt.close()
        return

    def plot_multicalls(self):
        plt.style.use('dark_background')
        fig = plt.figure(figsize=(60, 60))
        plt.axis(self.xyminmax)

        colors = ['r', 'b', 'C1', 'C5', 'C3']
        labels = ['zero calls (empty)', 'single calls', 'double calls (mixed)',
                  'triple calls (mixed)', 'quadruple calls (mixed)']
        for i in range(5):
            mask = self.label_arr[label_dict['Multicall']] == i
            plt.scatter(self.coords[0][mask], self.coords[1][mask], s=1, c=colors[i], marker=',', label=labels[i])

        plt.gca().invert_yaxis()
        plt.legend(markerscale=30, prop={'size': 60})

        plt.tight_layout()
        png_path = os.path.join(self.output_dp, '%s_Call_Counts.png' % self.prefix)
        try:
            fig.savefig(png_path)
        except IOError:
            fig.savefig('\\\\?\\' + png_path)
        plt.gcf().clear()
        plt.close()
        return

    def plot_nonCBI(self):
        plt.style.use('dark_background')
        logger.debug('%s - plot_nonCBI initiated.' % self.fov)
        colors = ['r', 'C3', 'y', 'g', 'C4']
        labels = ['[60%, 100%)', '[35%, 60%)', '[20%, 35%)', '[10%, 20%)', '[ 5%, 10%)']
        ranges = [[60, 100], [35, 60], [20, 35], [10, 20], [5, 10]]
        for bi, base in enumerate('ACGT'):
            debug_string = '%s - %s -' % (self.fov, base)
            for i in range(1, 4):
                debug_string += ' %s: %s,' % \
                                (i * 25, np.percentile(self.label_arr[label_dict['%s-nonCBI' % base]], i * 25))
            logger.debug(debug_string[:-1])

            fig = plt.figure(figsize=(60, 60))
            plt.axis(self.xyminmax)

            mask = np.isnan(self.label_arr[label_dict['%s-nonCBI' % base]] == 100)
            plt.scatter(self.coords[0][mask], self.coords[1][mask], s=1, c='m', marker=',', label='100%')

            for i, rang in enumerate(ranges):
                mask = np.logical_and(self.label_arr[label_dict['%s-nonCBI' % base]] >= rang[0],
                                      self.label_arr[label_dict['%s-nonCBI' % base]] < rang[1])
                plt.scatter(self.coords[0][mask], self.coords[1][mask], s=1, c=colors[i], marker=',', label=labels[i])

            mask = self.label_arr[label_dict['%s-nonCBI' % base]] < 5
            plt.scatter(self.coords[0][mask], self.coords[1][mask], s=1, c='b', marker=',', label='< 5%')

            plt.gca().invert_yaxis()
            plt.legend(markerscale=30, prop={'size': 60})

            plt.tight_layout()
            png_path = os.path.join(self.output_dp, '%s_nonCBI_%s_Proportion.png' % (self.prefix, base))
            try:
                fig.savefig(png_path)
            except IOError:
                fig.savefig('\\\\?\\' + png_path)
            plt.gcf().clear()
            plt.close()
        logger.debug('%s - plot_nonCBI completed.' % self.fov)
        return

    def plot_chastity(self):
        logger.debug('%s - plot_chastity initiated.' % self.fov)
        fig = plt.figure(figsize=(60, 60))
        plt.style.use('dark_background')
        plt.axis(self.xyminmax)
        colors = ['C3', 'r', 'y', 'c', 'C4', 'b', 'w']
        labels = ['0.0', '[0.5, 0.6)', '[0.6, 0.7)', '[0.7, 0.8)', '[0.8, 0.9)', '[0.9, 1.0)', '1.0']
        chastities = [0, 5, 6, 7, 8, 9, 10]
        for i, chas in enumerate(chastities):
            mask = self.label_arr[label_dict['Chastity']] == chas
            plt.scatter(self.coords[0][mask], self.coords[1][mask], s=1, c=colors[i], marker=',', label=labels[i])

        mask = np.isnan(self.label_arr[label_dict['Chastity']])
        plt.scatter(self.coords[0][mask], self.coords[1][mask], s=1, c='m', marker=',', label='NaN')
        plt.axis(self.xyminmax)

        plt.gca().invert_yaxis()
        plt.legend(markerscale=30, prop={'size': 60})

        plt.tight_layout()
        png_path = os.path.join(self.output_dp, '%s_Chastity.png' % self.prefix)
        try:
            fig.savefig(png_path)
        except IOError:
            fig.savefig('\\\\?\\' + png_path)
        plt.gcf().clear()
        plt.close()
        logger.debug('%s - plot_chastity completed.' % self.fov)
        return

    def plot_SHI(self):
        logger.debug('%s - plot_SHI initiated.' % self.fov)
        fig = plt.figure(figsize=(60, 60))
        plt.style.use('dark_background')
        plt.axis(self.xyminmax)
        colors = ['w', 'r', 'y', 'c', 'g', 'C1', 'C5', 'C4', 'C2', 'C3', 'b'][::-1]
        for i in range(11):
            label = '100% SHI' if i == 10 else ('[%d - %d)' % (i * 10, (i + 1) * 10) + '% SHI')
            mask = self.label_arr[label_dict['SHI']] == i
            plt.scatter(self.coords[0][mask], self.coords[1][mask], s=1, c=colors[i], marker=',', label=label)

        mask = np.isnan(self.label_arr[label_dict['SHI']])
        plt.scatter(self.coords[0][mask], self.coords[1][mask], s=1, c='m', marker=',', label='NaN')

        plt.gca().invert_yaxis()
        plt.legend(markerscale=30, prop={'size': 40})

        plt.tight_layout()
        png_path = os.path.join(self.output_dp, '%s_SHI.png' % self.prefix)
        try:
            fig.savefig(png_path)
        except IOError:
            fig.savefig('\\\\?\\' + png_path)
        plt.gcf().clear()
        plt.close()
        logger.debug('%s - plot_shi completed.' % self.fov)
        return

    def ACGT_split(self, horiz, vert, diag, multi, ACGT_dist, rates):
        A = C = G = T = 0

        horiz = unpack(horiz)
        vert = unpack(vert)
        diag = unpack(diag)
        multi = unpack(multi)

        h_split = {'A': 0, 'C': 0, 'G': 0, 'T': 0}
        v_split = {'A': 0, 'C': 0, 'G': 0, 'T': 0}
        d_split = {'A': 0, 'C': 0, 'G': 0, 'T': 0}
        m_split = {'A': 0, 'C': 0, 'G': 0, 'T': 0}

        with gzip.open(self.fastq_fp, 'r') as fq:
            contents = fq.readlines()
            for i in horiz:
                line = contents[4 * i + 1]
                decamer = line.rstrip()[0:10]
                for base in decamer:
                    if base in h_split.keys():
                        h_split[base] += 1

            for i in vert:
                line = contents[4 * i + 1]
                decamer = line.rstrip()[0:10]
                for base in decamer:
                    if base in v_split.keys():
                        v_split[base] += 1

            for i in diag:
                line = contents[4 * i + 1]
                decamer = line.rstrip()[0:10]
                for base in decamer:
                    if base in d_split.keys():
                        d_split[base] += 1

            for i in multi:
                line = contents[4 * i + 1]
                decamer = line.rstrip()[0:10]
                for base in decamer:
                    if base in m_split.keys():
                        m_split[base] += 1

        num_bases = 0
        for base in ACGT_dist:
            num_bases += ACGT_dist[base]

        # fig = plt.figure(1)
        # gridspec.GridSpec(9, 12)
        # mpl.rcParams.update({'font.size': 6})
        #
        # # Total Plot
        # plt.subplot2grid((9, 12), (0, 0), rowspan=9, colspan=2)
        # plt.rcParams.update(mpl.rcParamsDefault)
        # plt.style.use('seaborn-muted')
        # plt.title('A:C:G:T Ratio in \nAll Split DNBs', size=8, weight='bold')
        # plt.ylabel('%')
        # plt.xticks([1], ' ')

        combined_split = {}
        for key in h_split:
            combined_split[key] = h_split[key] + v_split[key] + d_split[key] + m_split[key]
        total = 0
        for key in combined_split:
            total += combined_split[key]
        for key in combined_split:
            combined_split[key] *= 100. / total if total else 1.0

        splits = [combined_split, h_split, v_split, d_split, m_split]
        self.plot_ACGT_split(splits, total, num_bases, ACGT_dist)
        self.save_ACGT_split(splits, rates)
        return

    def save_ACGT_split(self, splits, rates):
        import pandas as pd
        all_rates = [100.]
        all_rates.extend(rates)

        names = ['All Split', 'Horizontal', 'Vertical', 'Diagonal', 'Multi']
        lane = [self.lane] * len(names)
        fov = [self.fov] * len(names)
        arrays = [lane, fov, names]
        tuples = list(zip(*arrays))
        idx = pd.MultiIndex.from_tuples(tuples, names=['Lane', 'FOV', 'Split Type'])
        df = pd.DataFrame(splits, index=idx)
        df['%ofSplits'] = all_rates
        df = df[['%ofSplits', 'A', 'C', 'G', 'T']]
        df.to_csv(self.ACGT_dist_csv)
        return

    def plot_ACGT_split(self, splits, total, num_bases, ACGT_dist):
        combined_split, h_split, v_split, d_split, m_split = splits

        fig = plt.figure(1)
        gridspec.GridSpec(9, 12)
        mpl.rcParams.update({'font.size': 6})

        # Total Plot
        plt.subplot2grid((9, 12), (0, 0), rowspan=9, colspan=2)
        plt.rcParams.update(mpl.rcParamsDefault)
        plt.style.use('seaborn-muted')
        plt.title('A:C:G:T Ratio in \nAll Split DNBs', size=8, weight='bold')
        plt.ylabel('%')
        plt.xticks([1], ' ')

        p1 = plt.bar([1], combined_split['A'], color='r')
        p2 = plt.bar([1], combined_split['T'], bottom=combined_split['A'], color='b')
        p3 = plt.bar([1], combined_split['C'], bottom=combined_split['T'] + combined_split['A'], color='g')
        p4 = plt.bar([1], combined_split['G'], bottom=combined_split['C'] + combined_split['T'] + combined_split['A'],
                     color='k')
        x1, x2, y1, y2 = plt.axis()
        plt.axis((x1, x2, 0, 100))

        names = ['A: ' + "{0:.3}".format(combined_split['A']) + '%',
                 'T: ' + "{0:.3}".format(combined_split['T']) + '%',
                 'C: ' + "{0:.3}".format(combined_split['C']) + '%',
                 'G: ' + "{0:.3}".format(combined_split['G']) + '%']

        for i, base in enumerate('ATCG'):
            names[i] += calculate_significance(combined_split[base] / 100., combined_split[base] * total / 100.,
                                               ACGT_dist[base] / num_bases, ACGT_dist[base])

        plt.legend((p1[0], p2[0], p3[0], p4[0]), names, loc='center left', bbox_to_anchor=(1, 0.5))

        # Horiz Plot
        plt.subplot2grid((9, 12), (0, 4), rowspan=4, colspan=2)
        plt.title('A:C:G:T Ratio in \nHorizontally Split DNBs', size=8, weight='bold')
        plt.ylabel('%')
        plt.xticks([1], ' ')

        total = 0
        for key in h_split:
            total += h_split[key]
        for key in h_split:
            h_split[key] *= 100. / total if total else 1.0

        h1 = plt.bar([1], h_split['A'], color='r')
        h2 = plt.bar([1], h_split['T'], bottom=h_split['A'], color='b')
        h3 = plt.bar([1], h_split['C'], bottom=h_split['T'] + h_split['A'], color='g')
        h4 = plt.bar([1], h_split['G'], bottom=h_split['C'] + h_split['T'] + h_split['A'], color='k')
        x1, x2, y1, y2 = plt.axis()
        plt.axis((x1, x2, 0, 100))

        names = ['A: ' + "{0:.3}".format(h_split['A']) + '%',
                 'T: ' + "{0:.3}".format(h_split['T']) + '%',
                 'C: ' + "{0:.3}".format(h_split['C']) + '%',
                 'G: ' + "{0:.3}".format(h_split['G']) + '%']

        for i, base in enumerate('ATCG'):
            names[i] += calculate_significance(h_split[base] / 100., h_split[base] * total / 100.,
                                               ACGT_dist[base] / num_bases,
                                               ACGT_dist[base])

        plt.legend((h1[0], h2[0], h3[0], h4[0]), names, loc='center left', bbox_to_anchor=(1, 0.5))

        # Vert Plot
        plt.subplot2grid((9, 12), (0, 8), rowspan=4, colspan=2)
        plt.title('A:C:G:T Ratio in \nVertically Split DNBs', size=8, weight='bold')
        plt.ylabel('%')
        plt.xticks([1], ' ')

        total = 0
        for key in v_split:
            total += v_split[key]
        for key in v_split:
            v_split[key] *= 100. / total if total else 1.0

        v1 = plt.bar([1], v_split['A'], color='r')
        v2 = plt.bar([1], v_split['T'], bottom=v_split['A'], color='b')
        v3 = plt.bar([1], v_split['C'], bottom=v_split['T'] + v_split['A'], color='g')
        v4 = plt.bar([1], v_split['G'], bottom=v_split['C'] + v_split['T'] + v_split['A'], color='k')
        x1, x2, y1, y2 = plt.axis()
        plt.axis((x1, x2, 0, 100))

        names = ['A: ' + "{0:.3}".format(v_split['A']) + '%',
                 'T: ' + "{0:.3}".format(v_split['T']) + '%',
                 'C: ' + "{0:.3}".format(v_split['C']) + '%',
                 'G: ' + "{0:.3}".format(v_split['G']) + '%']

        for i, base in enumerate('ATCG'):
            names[i] += calculate_significance(v_split[base] / 100., v_split[base] * total / 100.,
                                               ACGT_dist[base] / num_bases,
                                               ACGT_dist[base])

        plt.legend((v1[0], v2[0], v3[0], v4[0]), names, loc='center left', bbox_to_anchor=(1, 0.5))

        # Diag Plot
        plt.subplot2grid((9, 12), (5, 4), rowspan=4, colspan=2)
        plt.title('A:C:G:T Ratio in \nDiagonally Split DNBs', size=8, weight='bold')
        plt.ylabel('%')
        plt.xticks([1], ' ')

        total = 0
        for key in d_split:
            total += d_split[key]
        for key in d_split:
            d_split[key] *= 100. / total if total else 1.0

        d1 = plt.bar([1], d_split['A'], color='r')
        d2 = plt.bar([1], d_split['T'], bottom=d_split['A'], color='b')
        d3 = plt.bar([1], d_split['C'], bottom=d_split['T'] + d_split['A'], color='g')
        d4 = plt.bar([1], d_split['G'], bottom=d_split['C'] + d_split['T'] + d_split['A'], color='k')
        x1, x2, y1, y2 = plt.axis()
        plt.axis((x1, x2, 0, 100))

        names = ['A: ' + "{0:.3}".format(d_split['A']) + '%',
                 'T: ' + "{0:.3}".format(d_split['T']) + '%',
                 'C: ' + "{0:.3}".format(d_split['C']) + '%',
                 'G: ' + "{0:.3}".format(d_split['G']) + '%']

        for i, base in enumerate('ATCG'):
            names[i] += calculate_significance(d_split[base] / 100., d_split[base] * total / 100.,
                                               ACGT_dist[base] / num_bases,
                                               ACGT_dist[base])

        plt.legend((d1[0], d2[0], d3[0], d4[0]), names, loc='center left', bbox_to_anchor=(1, 0.5))

        # Multi Plot
        plt.subplot2grid((9, 12), (5, 8), rowspan=4, colspan=2)
        plt.title('A:C:G:T Ratio in \nMulti Split DNBs', size=8, weight='bold')
        plt.ylabel('%')
        plt.xticks([1], ' ')

        total = 0
        for key in m_split:
            total += m_split[key]
        for key in m_split:
            m_split[key] *= 100. / total if total else 1.0

        m1 = plt.bar([1], m_split['A'], color='r')
        m2 = plt.bar([1], m_split['T'], bottom=m_split['A'], color='b')
        m3 = plt.bar([1], m_split['C'], bottom=m_split['T'] + m_split['A'], color='g')
        m4 = plt.bar([1], m_split['G'], bottom=m_split['C'] + m_split['T'] + m_split['A'], color='k')
        x1, x2, y1, y2 = plt.axis()
        plt.axis((x1, x2, 0, 100))

        names = ['A: ' + "{0:.3}".format(m_split['A']) + '%',
                 'T: ' + "{0:.3}".format(m_split['T']) + '%',
                 'C: ' + "{0:.3}".format(m_split['C']) + '%',
                 'G: ' + "{0:.3}".format(m_split['G']) + '%']

        for i, base in enumerate('ATCG'):
            names[i] += calculate_significance(m_split[base] / 100., m_split[base] * total / 100.,
                                               ACGT_dist[base] / num_bases,
                                               ACGT_dist[base])

        plt.legend((m1[0], m2[0], m3[0], m4[0]), names, loc='center left', bbox_to_anchor=(1, 0.5))

        fig.tight_layout()
        png_path = os.path.join(self.output_dp, '%s_ACGT.png' % self.prefix)
        try:
            fig.savefig(png_path)
        except IOError:
            fig.savefig('\\\\?\\' + png_path)
        plt.gcf().clear()
        plt.close()
        return

    def count_mixed_mirages(self):
        mirages = self.called_signals[self.neighbors_arr]
        mirages = np.apply_along_axis(lambda c: [c[0] & d for d in c], 1, mirages)
        mirages = np.any(mirages, 2)
        mirages = np.sum(mirages, 2) == 10
        return np.sum(mirages[:,1:], 0)

    def save_outputs(self, summary, results):
        #np.save(self.na_summary_fp, summary)
        with open(self.na_summary_fp, 'w') as f:
            pickle.dump(summary, f)
        #np.save(self.na_results_fp, results)
        with open(self.na_results_fp, 'w') as f:
            pickle.dump(results, f)
        return

    def save_possible_split_groups(self, possible_split_groups):
        with open(self.possible_split_groups_fp, 'w') as f:
            pickle.dump(possible_split_groups, f)
        return

    def save_ACGT_dist(self, ACGT_dist):
        with open(self.ACGT_dist_fp, 'w') as f:
            pickle.dump(ACGT_dist, f)
        return

    def save_sequence_strings(self, sequence_strings):
        np.save(self.sequence_strings_fp, sequence_strings)
        return

    def load_possible_split_groups(self):
        with open(self.possible_split_groups_fp, 'r') as p:
            possible_split_groups = pickle.load(p)
        return possible_split_groups

    def load_ACGT_dist(self):
        with open(self.ACGT_dist_fp, 'r') as p:
            ACGT_dist = pickle.load(p)
        return ACGT_dist

    def load_sequence_strings(self):
        return np.load(self.sequence_strings_fp)

    def save_split_rates(self, horiz_rate, vert_rate, diag_rate, multi_rate, spatdup_rate):
        np.save(self.split_rates_fp,
                np.asarray([horiz_rate, vert_rate, diag_rate, multi_rate, spatdup_rate]))
        return

    def load_split_rates(self):
        split_rates = np.load(self.split_rates_fp)
        i = 0
        horiz_rate = split_rates[i];
        i += 1
        vert_rate = split_rates[i];
        i += 1
        diag_rate = split_rates[i];
        i += 1
        multi_rate = split_rates[i];
        i += 1
        spatdup_rate = split_rates[i];
        return horiz_rate, vert_rate, diag_rate, multi_rate, spatdup_rate

    def save_splits_lists(self, horizontal_splits, vertical_splits, diagonal_splits, multiple_splits):
        with open(self.horiz_splits_fp, 'w') as f:
            pickle.dump(horizontal_splits, f)
        with open(self.vert_splits_fp, 'w') as f:
            pickle.dump(vertical_splits, f)
        with open(self.diag_splits_fp, 'w') as f:
            pickle.dump(diagonal_splits, f)
        with open(self.multi_splits_fp, 'w') as f:
            pickle.dump(multiple_splits, f)
        return

    def load_splits_lists(self):
        with open(self.horiz_splits_fp, 'r') as p:
            horizontal_splits = pickle.load(p)
        with open(self.vert_splits_fp, 'r') as p:
            vertical_splits = pickle.load(p)
        with open(self.diag_splits_fp, 'r') as p:
            diagonal_splits = pickle.load(p)
        with open(self.multi_splits_fp, 'r') as p:
            multiple_splits = pickle.load(p)
        return horizontal_splits, vertical_splits, diagonal_splits, multiple_splits

    def run(self):
        start_time = datetime.datetime.now()
        logger.info('%s - Initiating neighbor analysis...' % self.fov)
        self.block_bool = self.load_block_bool()
        self.neighbors_arr, self.num_spots = self.load_neighbors()

        mixed_indices = np.where(self.label_arr[label_dict['Multicall']] == 2)[0]
        logger.info('%s - Mixed DNB Count: %s' % (self.fov, len(mixed_indices)))
        try:
            if self.bypass['get_possible_split_groups']:
                logger.debug('%s - Attempting to bypass get_possible_split_groups...' % self.fov)
                possible_split_groups = self.load_possible_split_groups()
                ACGT_dist = self.load_ACGT_dist()
                sequence_strings = self.load_sequence_strings()
                logger.debug('%s - get_possible_split_groups bypass successful.' % self.fov)
            else:
                # reset array
                self.label_arr[label_dict['MixedSplit']] = -1
                self.label_arr[label_dict['HiddenSplit']] = -1
                raise Exception('%s - Not bypassing get_possible_split_groups...' % self.fov)
        except:
            if self.bypass['get_possible_split_groups']:
                logger.warning(traceback.format_exc())
                logger.warning('%s - Unable to bypass get_possible_split_groups!' % self.fov)
            possible_split_groups, ACGT_dist, sequence_strings = \
                self.get_possible_split_groups(mixed_indices)
            self.save_possible_split_groups(possible_split_groups)
            self.save_ACGT_dist(ACGT_dist)
            self.save_sequence_strings(sequence_strings)

        try:
            if self.bypass['calculate_split_percentage']:
                logger.debug('%s - Attempting to bypass calculate_split_percentage...' % self.fov)
                horiz_rate, vert_rate, diag_rate, multi_rate, spatdup_rate = self.load_split_rates()
                horizontal_splits, vertical_splits, diagonal_splits, multiple_splits = self.load_splits_lists()
                logger.debug('%s - calculate_split_percentage bypass successful.' % self.fov)
            else:
                # reset array
                self.label_arr[label_dict['Children']] = 0
                self.label_arr[label_dict['Parents']] = 0
                self.label_arr[label_dict['FamilySize']] = 0
                raise Exception('%s - Not bypassing calculate_split_percentage...' % self.fov)
        except:
            if self.bypass['calculate_split_percentage']:
                logger.warning(traceback.format_exc())
                logger.warning('%s - Unable to bypass calculate_split_percentage!' % self.fov)
            horiz_rate, vert_rate, diag_rate, multi_rate, \
            spatdup_rate, horizontal_splits, vertical_splits, diagonal_splits, multiple_splits = \
                self.calculate_split_percentage(possible_split_groups, sequence_strings)
            self.save_split_rates(horiz_rate, vert_rate, diag_rate, multi_rate, spatdup_rate)
            self.save_splits_lists(horizontal_splits, vertical_splits, diagonal_splits, multiple_splits)
        logger.info('%s - Split counts: %s (hori), %s (vert), %s (diag), %s (mult)' %
                         (self.fov,
                          len(horizontal_splits), len(vertical_splits), len(diagonal_splits), len(multiple_splits)))

        results = [
            ['Horizontal Split (%ofSplits)', horiz_rate],
            ['Vertical Split (%ofSplits)', vert_rate],
            ['Diagonal Split (%ofSplits)', diag_rate],
            ['Multiple Split (%ofSplits)', multi_rate]
        ]
        seq = seq_freq = ''
        if horizontal_splits or vertical_splits or diagonal_splits or multiple_splits:
            seq, seq_freq = self.most_frequent_sequence(horizontal_splits, vertical_splits, diagonal_splits, multiple_splits)
        results += [
            ['Most Frequent 10-mer', seq],
            ['Frequency of 10-mer', seq_freq]
        ]

        hvd = [horiz_rate, vert_rate, diag_rate]

        logger.debug('%s - naCBI_data: %s' % (self.fov, self.naCBI_data))
        if self.naCBI_data is not None:
            logger.debug('%s - coords_fp: %s' % (self.fov, self.coords_fp))
            if self.coords_fp != None:
                self.get_coords(self.coords_fp)
                self.get_fov_min_max()
                logger.debug('%s - coords: %s' % (self.fov, self.coords))
                logger.debug('%s - bypass: %s' % (self.fov, self.bypass))
                if not self.bypass['plot_nonCBI']:
                    self.plot_nonCBI()
                if not self.bypass['plot_multicalls']:
                    self.plot_multicalls()
                if not self.bypass['plot_chastity']:
                    self.plot_chastity()
                if not self.bypass['plot_SHI']:
                    self.plot_SHI()
                if not self.bypass['plot_cbi_rank']:
                    self.plot_cbi_rank()
                if not self.bypass['plot_cbi_thresholds']:
                    self.plot_cbi_thresholds()
                if not self.bypass['plot_splits']:
                    self.plot_splits(horizontal_splits, vertical_splits, diagonal_splits, multiple_splits)
                    #self.plot_splits_alt(horizontal_splits, vertical_splits, diagonal_splits, multiple_splits)
                #self.plot_cbi_thr_random()

            h_CBI_arr = self.split_CBI_arr(self.naCBI_data, horizontal_splits, 'single', 'horizontal', mixed_indices)
            v_CBI_arr = self.split_CBI_arr(self.naCBI_data, vertical_splits, 'single', 'vertical', mixed_indices)
            d_CBI_arr = self.split_CBI_arr(self.naCBI_data, diagonal_splits, 'single', 'diagonal', mixed_indices)
            m_CBI_arr = self.split_CBI_arr(self.naCBI_data, multiple_splits, 'multiple', '', mixed_indices)

            left = right = up = down = 0

            if len(horizontal_splits) > 0:
                h = self.calculate_single_split_CBI_ratio(h_CBI_arr, 'horizontal')
                self.plot_single_split_CBI_ratio(h_CBI_arr, 'horizontal')
                h_parent, h_child = self.profile_family(h_CBI_arr)
                left, right = self.count_parents(h_CBI_arr)
            results += [
                ['Left Parent Count', left],
                ['Right Parent Count', right]
            ]
            if len(vertical_splits) > 0:
                v = self.calculate_single_split_CBI_ratio(v_CBI_arr, 'vertical')
                self.plot_single_split_CBI_ratio(v_CBI_arr, 'vertical')
                v_parent, v_child = self.profile_family(v_CBI_arr)
                up, down = self.count_parents(v_CBI_arr)
            results += [
                ['Up Parent Count', up],
                ['Down Parent Count', down]
            ]
            if len(diagonal_splits) > 0:
                d = self.calculate_single_split_CBI_ratio(d_CBI_arr, 'diagonal')
                self.plot_single_split_CBI_ratio(d_CBI_arr, 'diagonal')
                d_parent, d_child = self.profile_family(d_CBI_arr)
            if len(multiple_splits) > 0:
                m = self.calculate_multi_split_CBI_ratio(m_CBI_arr)
                m_parent, m_children = self.profile_family(m_CBI_arr)

            empty_parents_rate = empty_children_rate = empty_split_rate = 0
            if len(horizontal_splits) > 0 and len(vertical_splits) > 0 and len(diagonal_splits) > 0 and \
                    len(multiple_splits) > 0:
                directions = ['Horizontal', 'Vertical', 'Diagonal', 'Multiple']
                empty_parents, total_parents = self.plot_profile([h_parent, v_parent, d_parent, m_parent],
                                                                 directions, 'Parents')
                logger.info('%s - Parent profiles plotted.' % self.fov)
                empty_children, total_children = self.plot_profile([h_child, v_child, d_child, m_children],
                                                                   directions, 'Children')
                logger.info('%s - Children profiles plotted.' % self.fov)
                self.plot_CBI_ratios(h, v, d, m, hvd)
                logger.info('%s - plotted ratios' % self.fov)

                rates = [horiz_rate, vert_rate, diag_rate, multi_rate]
                self.ACGT_split(horizontal_splits, vertical_splits, diagonal_splits, multiple_splits, ACGT_dist, rates)

                empty_parents_rate = 100. * empty_parents / total_parents if total_parents else 'NA'
                empty_children_rate = 100. * empty_children / total_children if total_children else 'NA'
                empty_split_rate = 100. * (empty_parents + empty_children) / (total_parents + total_children) if \
                    (total_parents + total_children) else 'NA'

            results += [
                ['Empty Parents (%ofSingleParents)', empty_parents_rate],
                ['Empty Children (%ofSingleChildren)', empty_children_rate],
                ['Empty Splits (%ofSingleSplits)', empty_split_rate]
            ]

        """
        expanded_nonsplit_sequence_strings = self.expand_nonsplit_sequences(self.sequence_strings)
        expanded_split_groups = self.get_expanded_split_groups(expanded_nonsplit_sequence_strings)
        multi_count, horiz_count, vert_count, diag_count, \
        horizontal_splits, vertical_splits, diagonal_splits, multiple_splits = self.calculate_split_percentage(expanded_split_groups, label=False)
        logger.info('Expanded split counts: %s (hori), %s (vert), %s (diag), %s (mult)' %
                         (len(horizontal_splits), len(vertical_splits), len(diagonal_splits), len(multiple_splits)))

        expanded_nonsplit_sequence_strings = self.expand_nonsplit_sequences(self.sequence_strings, True)
        expanded_split_groups = self.get_expanded_split_groups(expanded_nonsplit_sequence_strings)
        multi_count, horiz_count, vert_count, diag_count, \
        horizontal_splits, vertical_splits, diagonal_splits, multiple_splits = self.calculate_split_percentage(expanded_split_groups, label=False)
        logger.info('Expanded split counts (including Parents): %s (hori), %s (vert), %s (diag), %s (mult)' %
                         (len(horizontal_splits), len(vertical_splits), len(diagonal_splits), len(multiple_splits)))

        for average_count in [50, 100, 250, 500, 1000]:
            expanded_ld_sequence_strings = self.expand_lowdiversity_sequences(self.sequence_strings, average_count)
            expanded_split_groups = self.get_expanded_split_groups(expanded_ld_sequence_strings)
            multi_count, horiz_count, vert_count, diag_count, \
            horizontal_splits, vertical_splits, diagonal_splits, multiple_splits = self.calculate_split_percentage(expanded_split_groups, label=False)
            logger.info('Expanded split counts (avg count: %s): %s (hori), %s (vert), %s (diag), %s (mult)' %
                             (average_count, len(horizontal_splits), len(vertical_splits), len(diagonal_splits), len(multiple_splits)))
        
        for adapter_count in [100, 3000, 10000, 80000]:
            expanded_ac_sequence_strings = self.expand_adapter_sequences(self.sequence_strings, adapter_count)
            expanded_ac_groups = self.get_expanded_split_groups(expanded_ac_sequence_strings)
            multi_count, horiz_count, vert_count, diag_count, \
            horizontal_splits, vertical_splits, diagonal_splits, multiple_splits = self.calculate_split_percentage(expanded_ac_groups, label=False)
            logger.info('Expanded split counts (avg count: %s): %s (hori), %s (vert), %s (diag), %s (mult)' %
                             (adapter_count, len(horizontal_splits), len(vertical_splits), len(diagonal_splits), len(multiple_splits)))
            self.plot_splits(horizontal_splits, vertical_splits, diagonal_splits, multiple_splits, '_%sac' % adapter_count)

        shuffled_sequences = self.shuffle_sequences(self.sequence_strings)
        shuffled_split_groups = self.get_expanded_split_groups(shuffled_sequences)
        multi_count, horiz_count, vert_count, diag_count, \
        horizontal_splits, vertical_splits, diagonal_splits, multiple_splits = self.calculate_split_percentage(shuffled_split_groups, label=False)
        logger.info('Shuffled split counts (including Parents): %s (hori), %s (vert), %s (diag), %s (mult)' %
                         (len(horizontal_splits), len(vertical_splits), len(diagonal_splits), len(multiple_splits)))
        """

        analysis_stop = datetime.datetime.now()
        time_diff = analysis_stop - self.start_time
        hours = time_diff.seconds // 3600
        minutes = (time_diff.seconds // 60) % 60
        seconds = time_diff.seconds % 60
        logger.info('%s - Neighbor analysis completed. (%s hrs, %s min, %s sec)' % (self.fov, hours, minutes, seconds))
        summary = [
            ['Spatial Duplicates (%ofTotal)', spatdup_rate]
        ]

        self.save_outputs(summary, results)
        time_diff = datetime.datetime.now() - start_time
        logger.info('%s Complete (%s)' % (self.fov, time_diff))
        return summary, results

    def complete_bypass(self):
        try:
            with open(self.na_summary_fp, 'r') as p:
                summary = pickle.load(p)
            with open(self.na_results_fp, 'r') as p:
                results = pickle.load(p)

            logger.info('%s - Bypass successful.' % self.fov)
        except:
            logger.warning(traceback.format_exc())
            logger.warning('%s - Could not bypass Neighbor Analysis!' % self.fov)
            summary, results = self.run()
        return summary, results

def main(slide, lane, fov, start_cycle, occupancy_range, int_fp, posinfo_fp, fastq_fp):
    inta = IntensityAnalysis(slide, lane, fov, start_cycle, occupancy_range, int_fp)
    # inta.load_data()

    ma = NeighborAnalysis(inta, posinfo_fp, fastq_fp=fastq_fp)
    summary, results = ma.run()
    print summary, results
    return summary, results

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7], sys.argv[8])