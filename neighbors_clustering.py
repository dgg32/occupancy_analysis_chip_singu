import os
import numpy as np
import pandas as pd
import math
from sklearn.neighbors import NearestNeighbors
from scipy.stats import norm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

label_order = [
    'PercCBI',
    'Multicall',
    'Chastity',
    'Children',
    'Parents',
    'MixedSplit',
    'HiddenSplit',
    'FamilySize',
    'PercReadSNR',
    'PercCycleSNR',
    'SHI',
    'A-nonCBI',
    'C-nonCBI',
    'G-nonCBI',
    'T-nonCBI'
]
label_dict = dict((label, c) for c, label in enumerate(label_order))
np.random.seed(1000)


class NeighborClustering(object):
    def __init__(self, int_analysis, neighbors_fp, blocks, report_name):
        self.report_name = report_name
        self.output_dp = int_analysis.output_dp
        self.prefix = int_analysis.prefix
        self.fov = int_analysis.fov
        self.start_cycle = int_analysis.start_cycle
        self.occupancy_range = int_analysis.cycle_range
        self.labels_fp = int_analysis.labels_fp
        self.snr_fp = int_analysis.snr_fp
        posinfo_fp = int_analysis.background_fp.replace('_background.npy', '.posiIndex.txt')
        neighbors_fp = neighbors_fp
        self.blocks = blocks
        self.inner = False if blocks else True

        if not os.path.exists(self.output_dp):
            os.makedirs(self.output_dp)
        if not os.path.exists(posinfo_fp) or not os.path.exists(neighbors_fp):
            int_fp, posinfo_fp, neighbors_fp = self.output_data(data_dp)

        self.posinfo_fp = posinfo_fp
        self.neighbors_fp = neighbors_fp
        self.neighbors_arr = np.load(self.neighbors_fp)
        return

    def output_data(self, data_dp):
        from occuint2npy import Int2npy
        import pos2neighbor
        i2n = Int2npy(data_dp, self.fov, self.start_cycle, self.occupancy_range, self.output_dp)
        int_fp, posinfo_fp, norm_paras_fp, background_fp = i2n.run()

        blocks = self.blocks
        coords_fp = os.path.join(self.output_dp, '%s_coords.npy' % self.fov)
        neighbors_fp = os.path.join(self.output_dp, '%s_neighbors.npy' % self.fov)
        blocks_fp = os.path.join(self.output_dp, '%s_blocks.npy' % self.fov) if blocks else None
        pos2neighbor.main(posinfo_fp, coords_fp, neighbors_fp, blocks_fp, 1, blocks, v1=False)
        return int_fp, posinfo_fp, neighbors_fp

    def subset_data(self, dnb_pos):
        if self.inner:
            edge_blocks = range(10) + range(10, 90, 10) + range(19, 99, 10) + range(90, 100)
            block_list = np.array(list(set(range(100)) - set(edge_blocks)))
            block_bool = np.in1d(dnb_pos[:, 0], block_list)
        else:
            block_bool = self.blocks

        self.block_bool = block_bool
        dnb_pos = dnb_pos[block_bool, 1:]
        self.dnb_pos = dnb_pos
        self.xyminmax = (dnb_pos[:, 0].min(), dnb_pos[:, 0].max(), dnb_pos[:, 1].min(), dnb_pos[:, 1].max())
        self.neighbors_arr = self.neighbors_arr[block_bool]
        self.num_dnbs = len(dnb_pos)
        return

    def get_mixed_dnbs(self, mixed):
        mixed_indices = np.where(mixed)[0]
        dnb_pos_mix = self.dnb_pos[mixed_indices, :]
        perc = len(dnb_pos_mix) * 100. / self.num_dnbs
        mixed_neighbors = self.neighbors_arr[mixed_indices, 1:]
        mixed_neighbors_num = np.sum(np.sum(np.isin(mixed_neighbors, dnb_pos_mix[:, 2]), axis=1) > 0)

        np.random.seed(1000)
        mixed_indices_shuffle = mixed.copy()
        np.random.shuffle(mixed_indices_shuffle)
        mixed_indices_shuffle = np.where(mixed_indices_shuffle)[0]
        dnb_pos_mix_shuffle = self.dnb_pos[mixed_indices_shuffle, :]
        mixed_neighbors_shuffle_num = np.sum(np.sum(
            np.isin(self.neighbors_arr[mixed_indices_shuffle, 1:], dnb_pos_mix_shuffle[:, 2]), axis=1) > 0)
        mixed_neighbors_ratio = float(mixed_neighbors_shuffle_num) / float(
            mixed_neighbors_num) if mixed_neighbors_num > 0 else -1
        return dnb_pos_mix, mixed_neighbors, [perc, mixed_neighbors_num, mixed_neighbors_shuffle_num,
                                              mixed_neighbors_ratio]

    def get_independent_distances(self, data, metric='euclidean', k_neighbors=1):
        p = NearestNeighbors(k_neighbors, metric=metric)
        p.fit(data)
        idx = np.arange(len(data)).reshape(len(data), k_neighbors)
        dist, nn = p.kneighbors()

        abc = np.concatenate((idx, nn, dist), axis=1)
        data = pd.DataFrame(data=abc, columns=['point', 'nn', 'dist']).drop_duplicates('nn').values
        _, inds = np.unique(np.sort(data, axis=1), return_index=True, axis=0)
        return data[inds, 2]

    def csr_test(self, data, df=2):
        np.random.seed(1000)

        n = len(data)
        density = n / float(self.num_dnbs)
        csr_mean = 1. / (2 * math.sqrt(density))
        csr_std = math.sqrt((4 - math.pi) / (n * 4 * math.pi * density))
        s = np.random.normal(csr_mean, csr_std, n)
        csr_var = np.var(s)

        mean = float(np.mean(data))
        std_dev = float(np.std(data))
        var = np.var(data)
        zscore = (mean - csr_mean) / csr_std
        count, bins = np.histogram(data, bins='auto')
        y = 1 / (std_dev * np.sqrt(2 * np.pi)) * np.exp(- (bins - mean) ** 2 / (2 * std_dev ** 2))
        left_tail = norm.cdf(zscore, len(bins) + len(y) - df)
        pvals = [left_tail * 2, left_tail, 1. - left_tail]

        means_ratio = float(mean) / float(csr_mean) if csr_mean > 0 else 0
        return [csr_mean, mean, means_ratio, csr_std, std_dev, csr_var, var, zscore, pvals[1]]

    def save_outputs(self, values, names, subsets):
        cols = ['%dnbs_of_inner',
                'neighbors', 'neighbors_shuffled', 'neighbors_ratios']
        df = pd.DataFrame(np.array(values), index=names, columns=cols).T
        df.T.to_csv(os.path.join(self.output_dp, '%s_Cluster_%s_Summary.csv') % (self.report_name, subsets))
        return

    def run(self, subsets='Mixed'):
        np.random.seed(1000)
        dnb_pos = get_dnb_pos(self.posinfo_fp)
        self.subset_data(dnb_pos)
        if subsets == 'SNR':
            snr_data = np.load(self.snr_fp)
            snr_data = snr_data[:, self.block_bool]
            mixed_data = [snr_data[0] < np.percentile(snr_data[0], 1),
                          np.logical_and(snr_data[0] >= np.percentile(snr_data[0], 0.5),
                                         snr_data[0] < np.percentile(snr_data[0], 1)),
                          snr_data[0] < np.percentile(snr_data[0], 0.5),
                          snr_data[0] < np.percentile(snr_data[0], 0.1),
                          snr_data[0] < np.percentile(snr_data[0], 0.01)]
            names = ['1% SNR ReadWise Rank', '[0.50-1)% SNR ReadWise Rank', '0.50% SNR ReadWise Rank',
                     '0.10% SNR ReadWise Rank', '0.01% SNR ReadWise Rank']

            mixed_data.extend([snr_data[1] < np.percentile(snr_data[1], 1),
                               np.logical_and(snr_data[1] >= np.percentile(snr_data[1], 0.5),
                                              snr_data[1] < np.percentile(snr_data[1], 1)),
                               snr_data[1] < np.percentile(snr_data[1], 0.5),
                               snr_data[1] < np.percentile(snr_data[1], 0.1),
                               snr_data[1] < np.percentile(snr_data[1], 0.01)])
            names.extend(['1% SNR CycleWise Rank', '[0.50-1)% SNR CycleWise Rank', '0.50% SNR CycleWise Rank',
                          '0.10% SNR CycleWise Rank', '0.01% SNR CycleWise Rank'])
        else:
            label_arr = np.load(self.labels_fp)
            self.label_arr = label_arr[:, self.block_bool]
            empties = self.label_arr[label_dict['Multicall']] == 0
            all_mixed = np.logical_or(self.label_arr[label_dict['Multicall']] > 1,
                                      self.label_arr[label_dict['Chastity']] < 7)
            all_mixed_nonempty = np.logical_and(np.logical_or(self.label_arr[label_dict['Multicall']] > 1,
                                                              self.label_arr[label_dict['Chastity']] < 7),
                                                self.label_arr[label_dict['Multicall']] != 0)
            mixed_mc_lc = np.logical_and(self.label_arr[label_dict['Multicall']] > 1,
                                         self.label_arr[label_dict['Chastity']] < 7)
            mixed_mc = self.label_arr[label_dict['Multicall']] > 1
            mixed_mc2 = self.label_arr[label_dict['Multicall']] == 2
            mixed_mc3 = self.label_arr[label_dict['Multicall']] == 3
            mixed_mc4 = self.label_arr[label_dict['Multicall']] == 4
            mixed_lc = self.label_arr[label_dict['Chastity']] < 7
            mixed_lc6 = self.label_arr[label_dict['Chastity']] == 6
            mixed_lc5 = self.label_arr[label_dict['Chastity']] == 5
            mixed_lcnan = self.label_arr[label_dict['Chastity']] == 0
            mixed_data = [empties, all_mixed, all_mixed_nonempty, mixed_mc_lc, mixed_mc, mixed_mc2, mixed_mc3, mixed_mc4,
                          mixed_lc, mixed_lc6, mixed_lc5, mixed_lcnan]
            names = ['Empty', 'Multicall or Low Chastity', 'Multicall or Low Chastity Non-Empty',
                     'Multicall and Low Chastity',
                     'Multicall', '2-Call', '3-Call', '4-Call',
                     'Low Chastity', '0.6-Chastity', '0.5-Chastity', 'NaN-Chastity']
        values = []
        for mixed, name in zip(mixed_data, names):
            mixed_dnb_pos, mixed_neighbors_arr, neighbors_stats = self.get_mixed_dnbs(mixed)
            values.append(neighbors_stats)
        self.save_outputs(values, names, subsets)
        return


def plot_dnbs_scatter(dnb_pos, xyminmax, num_dnbs, output_dp, prefix, name, neighbors=None):
    perc = len(dnb_pos) * 100. / num_dnbs
    dnb_label = 'empty' if 'Empty' in name else 'mixed'
    name = name.replace(' ', '_')
    title = '{}: {:.3f}% {} DNBs of {} inner'.format(prefix, perc, dnb_label, num_dnbs)
    png_path = os.path.join(output_dp, '%s_%s_dnbs.png' % (prefix, dnb_label))
    if name is not None:
        title = '{}: {:.3f}% {} DNBs\n({}) of {} inner'.format(prefix, perc, dnb_label, name.replace('_neighbors', ''), num_dnbs)
        png_path = png_path.replace('.png', '_%s.png' % name)

    s = 1
    ms = 15
    fig, ax = plt.subplots(figsize=(50, 50))
    plt.axis(xyminmax)
    ax.scatter(dnb_pos[:, 0], dnb_pos[:, 1], s=s, marker=',', label='%s DNBs' % dnb_label)
    if neighbors is not None:
        mixed_neighbors = np.ravel(neighbors)
        # mixed_neighbors_num = np.sum(np.isin(neighbors, dnb_pos[:, 2]))
        mixed_neighbors_num = np.sum(np.sum(np.isin(neighbors, dnb_pos[:, 2]), axis=1) > 0)
        mixed_neighbors_pos = dnb_pos[np.isin(dnb_pos[:, 2], np.unique(mixed_neighbors))]
        ax.scatter(mixed_neighbors_pos[:, 0], mixed_neighbors_pos[:, 1], s=s, marker=',',
                   label='%s neighbors' % dnb_label)
        title = title + ', {} DNB neighbors'.format(mixed_neighbors_num)

    ax.invert_yaxis()
    ax.legend(loc='upper right', markerscale=ms, prop={'size': 30})
    ax.set_title(title, fontsize=30)
    plt.tight_layout()  #rect=[0.02, 0.02, 0.98, 0.98])
    fig.savefig(png_path)
    plt.gcf().clear()
    plt.close()
    return


def get_dnb_pos(posinfo_fp):
    pos_idx = pd.read_csv(posinfo_fp, sep='\t', header=None)
    idx, block, x, y = [], [], [], []
    height = 0
    groups = pos_idx.groupby(1)
    group = groups.get_group(0)
    for r in range(10):
        width = 0
        for c in range(10):
            group = groups.get_group(int(str(r) + str(c)))
            idx.extend((group[0]).astype(int).tolist())
            block.extend((group[1]).astype(np.int16).tolist())
            x.extend(((group[3]) + width).astype(np.int32).tolist())
            y.extend(((group[2]) + height).astype(np.int32).tolist())
            width += len(np.unique(group[3])) + 2
        height += len(np.unique(group[2])) + 2

    dnb_pos = np.stack([np.array(block), np.array(x), np.array(y), np.array(idx)]).T
    dnb_pos = dnb_pos[dnb_pos[:, -1].argsort()]
    return dnb_pos


def main():
    slide = 'V300015377'  #'V300011269' #'V300015281'  #'V300011269'  #'V300015377'  #'V300015281'
    lane = 'L01'
    fovs = ['C001R006']
    start_cycle = 1
    occupancy_range = 10
    blocks = []
    data_dp = '//prod/hustor-01/zebra/ZebraV2.1/1.0.7.197_CPU/IntData/%s/%s' % (slide, lane)
    # fovs = ['C001R006', 'C002R012', 'C003R024', 'C004R036', 'C005R048', 'C006R060']
    for fov in fovs:
        temp_dp = '/home/wvien/occupancy_test/%s/temp/%s' % (slide, fov)
        nc = NeighborClustering(data_dp, slide, lane, fov, blocks, start_cycle, occupancy_range, temp_dp)
        nc.run(subsets='mixed')
    return


if __name__ == '__main__':
    main()
