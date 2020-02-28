from __future__ import division
import sys
import numpy as np
import os
import matplotlib as mpl

mpl.use('Agg')

import logging.config
logger = logging.getLogger(__name__)
from sap_funcs import setup_logging

from intensity_analysis import IntensityAnalysis
from scipy.stats import gaussian_kde

import matplotlib.pyplot as plt

from intensity_analysis import label_dict

import traceback

class LabelAnalysis(object):
    def __init__(self, int_analysis, bypass={}, center=False, log_dp='', log_overrides={}):
        self.fov = int_analysis.fov
        self.center = center
        if self.center:
            data_dir = os.path.dirname(int_analysis.int_fp)
            posIndex = os.path.join(data_dir, str(self.fov) + '.posiIndex.txt')
            pos_data = np.loadtxt(posIndex, delimiter='\t')
            blocks = pos_data[:, 1].astype(int)
            blocks_bool = np.zeros(len(blocks), dtype=bool)
            for center_block in [44, 45, 54, 55]:
                blocks_bool[blocks == center_block] = True
            # center_block_bool = ((blocks == 44) | (blocks == 45) | (blocks == 54) | (blocks == 55))
            self.center_str = 'Center_'
        else:
            blocks_bool = np.ones(int_analysis.naCBI_data.shape[0]).astype(bool)
            self.center_str = ''
        self.empty_fth = int_analysis.empty_fth
        self.small_fth = int_analysis.small_fth
        self.large_fth = int_analysis.large_fth
        self.outlier_fth = int_analysis.outlier_fth
        self.naCBI_data = int_analysis.naCBI_data[blocks_bool]
        self.label_arr = int_analysis.label_arr[:, blocks_bool]
        self.output_dp = int_analysis.output_dp
        self.prefix = int_analysis.prefix

        # output histogram npy paths
        self.avgCBI_hist_npy = os.path.join(self.output_dp, self.center_str + '%s_avgCBI_Hist.npy' % self.prefix)

        # output pickle paths
        self.size_results_fp = os.path.join(self.output_dp, self.center_str + '%s_Size_Results.p' % self.prefix)
        self.size_summary_fp = os.path.join(self.output_dp, self.center_str + '%s_Size_Summary.p' % self.prefix)

        self.multicall_results_fp = os.path.join(self.output_dp, self.center_str + '%s_Multicall_Results.p' % self.prefix)
        self.multicall_summary_fp = os.path.join(self.output_dp, self.center_str + '%s_Multicall_Summary.p' % self.prefix)

        self.chastity_results_fp = os.path.join(self.output_dp, self.center_str + '%s_Chastity_Results.p' % self.prefix)
        self.chastity_summary_fp = os.path.join(self.output_dp, self.center_str + '%s_Chastity_Summary.p' % self.prefix)

        self.SHI_results_fp = os.path.join(self.output_dp, self.center_str + '%s_SHI_Results.p' % self.prefix)
        self.SHI_summary_fp = os.path.join(self.output_dp, self.center_str + '%s_SHI_Summary.p' % self.prefix)
        self.mixed_summary_fp = os.path.join(self.output_dp, self.center_str + '%s_Mixed_Summary.p' % self.prefix)

        self.empty_splits_results_fp = os.path.join(self.output_dp,
                                                    self.center_str + '%s_Empty_Splits_Results.p' % self.prefix)
        self.mixed_splits_results_fp = os.path.join(self.output_dp,
                                                    self.center_str + '%s_Mixed_Splits_Results.p' % self.prefix)

        self.familial_results_fp = os.path.join(self.output_dp, self.center_str + '%s_Familial_Results.p' % self.prefix)
        self.singular_summary_fp = os.path.join(self.output_dp, self.center_str + '%s_Singular_Summary.p' % self.prefix)

        self.splits_results_fp = os.path.join(self.output_dp, self.center_str + '%s_Splits_Results.p' % self.prefix)
        self.splits_summary_fp = os.path.join(self.output_dp, self.center_str + '%s_splits_summary.p' % self.prefix)

        self.cbi_quartile_results_fp = os.path.join(self.output_dp,
                                                    self.center_str + '%s_CBI_Quartile_Results.p' % self.prefix)
        self.snr1_quartile_results_fp = os.path.join(self.output_dp,
                                                     self.center_str + '%s_SNR1_Quartile_Results.p' % self.prefix)
        self.snr2_quartile_results_fp = os.path.join(self.output_dp,
                                                     self.center_str + '%s_SNR2_Quartile_Results.p' % self.prefix)

        self.bypass = bypass
        self.bypass['plot_cbi_KDEs'] = self.bypass.pop('plot_cbi_KDEs', True)
        self.bypass['plot_cbi_hist'] = self.bypass.pop('plot_cbi_hist', True)

        self.log_dp = log_dp
        self.log_overrides = log_overrides
        sub_log_fn = os.path.join(log_dp, '%s.log' % self.fov)
        sub_error_log_fn = os.path.join(log_dp, '%s_errors.log' % self.fov)
        override_dict = {'sub.log': sub_log_fn, 'sub_errors.log': sub_error_log_fn}
        override_dict.update(log_overrides)
        setup_logging(overrides=override_dict)

        logger.info('Numpy version: %s' % np.version.version)
        return

    def plot_cbi_KDEs(self):
        xs = np.linspace(0, 3.5, 1000)
        full_density = gaussian_kde(self.naCBI_data)(xs)
        singular_density = gaussian_kde(self.naCBI_data[self.singular])(xs)

        fig, ax = plt.subplots(2, sharex=True, figsize=(12.8, 9.6))
        mpl.rcParams.update({'font.size': 10})

        ax[0].plot(xs, full_density, color='C0', label='All DNBs')
        ax[0].plot(xs, singular_density, color='C1', label='Singular DNBs')
        # empty, small, large
        ax[0].fill_between(xs, full_density, where=xs <= self.empty_fth, facecolor='red', alpha=0.4)
        ax[0].fill_between(xs, full_density, where=np.logical_and(xs > self.empty_fth, xs <= self.small_fth),
                           facecolor='blue',
                           alpha=0.4)
        ax[0].fill_between(xs, full_density, where=np.logical_and(xs > self.large_fth, xs <= self.outlier_fth),
                           facecolor='green',
                           alpha=0.4)
        ax[0].annotate('Empty (< {:,.2}): {:,.3}%'.format(self.empty_fth, self.empty_PofT) + \
                       '\nSmall (range({:,.2}, {:,.2})): {:,.3}%'.format(self.empty_fth, self.small_fth, self.small_PofT) + \
                       '\nLarge (> {:,.2}): {:,.3}%'.format(self.large_fth, self.large_PofT) + \
                       '\nHigh Intensity Outliers: {:,.3}%'.format(self.outlier_PofT),
                       xy=(0.9925, 0.83), xycoords='axes fraction', horizontalalignment='right',
                       verticalalignment='top')
        ax[0].xaxis.set_ticks_position('bottom')
        ax[0].yaxis.set_ticks_position('left')
        ax[0].legend(markerscale=20)

        ax[1].plot(xs, full_density, color='C0', label='All DNBs')
        ax[1].plot(xs, singular_density, color='C1', label='Singular DNBs')
        # 0.25, 0.5, 1.5
        ax[1].fill_between(xs, full_density, where=xs < 0.25, facecolor='red', alpha=0.4)
        ax[1].fill_between(xs, full_density, where=np.logical_and(xs >= 0.25, xs < 0.50), facecolor='blue', alpha=0.4)
        ax[1].fill_between(xs, full_density, where=xs > 1.50, facecolor='green', alpha=0.4)
        ax[1].annotate('DNBs < 0.25: {:,.3}%'.format(self.arb1_PofT) + \
                       '\nDNBs in range(0.25, 0.5): {:,.3}%'.format(self.arb2_PofT) + \
                       '\nDNBs in range(1.5, 3): {:,.3}%'.format(self.arb3_PofT),
                       xy=(0.9925, 0.83), xycoords='axes fraction', horizontalalignment='right',
                       verticalalignment='top')
        ax[1].xaxis.set_ticks_position('bottom')
        ax[1].yaxis.set_ticks_position('left')
        ax[1].legend(markerscale=20)

        fig.text(0.5, 0.04, 'CBI', ha='center', va='center')
        fig.text(0.03, 0.5, 'Density', ha='center', va='center', rotation='vertical')
        plt.suptitle('%s: Avg CBI' % self.prefix, size=14)

        png_path = os.path.join(self.output_dp, '%s_avgCBI_Sizing_KDE.png' % self.prefix)
        try:
            plt.savefig(png_path)
        except:
            plt.savefig('\\\\?\\' + png_path)
        plt.gcf().clear()
        plt.close()

        # split breakdown
        nchildren_density = gaussian_kde(self.naCBI_data[self.label_arr[label_dict['Parents']] == 0])(xs)
        nparent_density = gaussian_kde(self.naCBI_data[self.label_arr[label_dict['Children']] == 0])(xs)
        nsplit_density = gaussian_kde(self.naCBI_data[self.non_split])(xs)

        # mixed breakdown
        highchastity_density = gaussian_kde(self.naCBI_data[self.label_arr[label_dict['Chastity']] >= 7])(xs)
        singlecall_density = gaussian_kde(self.naCBI_data[self.label_arr[label_dict['Multicall']] == 1])(xs)
        nmixed_density = gaussian_kde(self.naCBI_data[self.non_mixed])(xs)

        # combined breakdown
        nsm_density = gaussian_kde(self.naCBI_data[self.non_split_and_non_mixed])(xs)

        fig, ax = plt.subplots(3, sharex=True, figsize=(12.8, 9.6))
        mpl.rcParams.update({'font.size': 10})

        ax[0].plot(xs, full_density, color='C0', label='All DNBs')
        ax[0].plot(xs, nchildren_density, color='C1', label='Non-Children DNBs')
        ax[0].plot(xs, nparent_density, color='C2', label='Non-Parent DNBs')
        ax[0].plot(xs, nsplit_density, color='C4', label='Non-Split DNBs')
        ax[0].axvline(x=self.empty_fth, color='red', linestyle='--')
        ax[0].xaxis.set_ticks_position('bottom')
        ax[0].yaxis.set_ticks_position('left')
        ax[0].legend(markerscale=20)

        ax[1].plot(xs, full_density, color='C0', label='All DNBs')
        ax[1].plot(xs, highchastity_density, color='C1', label='High Chastity DNBs')
        ax[1].plot(xs, singlecall_density, color='C2', label='Single-call DNBs')
        ax[1].plot(xs, nmixed_density, color='C4', label='Non-Mixed DNBs')
        ax[1].axvline(x=self.empty_fth, color='red', linestyle='--')
        ax[1].xaxis.set_ticks_position('bottom')
        ax[1].yaxis.set_ticks_position('left')
        ax[1].legend(markerscale=20)

        ax[2].plot(xs, full_density, color='C0', label='All DNBs')
        ax[2].plot(xs, nsplit_density, color='C1', label='Non-Split DNBs')
        ax[2].plot(xs, nmixed_density, color='C2', label='Non-Mixed DNBs')
        ax[2].plot(xs, nsm_density, color='C4', label='Non-Mixed/Split DNBs')
        ax[2].axvline(x=self.empty_fth, color='red', linestyle='--')
        ax[2].xaxis.set_ticks_position('bottom')
        ax[2].yaxis.set_ticks_position('left')
        ax[2].legend(markerscale=20)

        fig.text(0.5, 0.04, 'CBI', ha='center', va='center')
        fig.text(0.03, 0.5, 'Density', ha='center', va='center', rotation='vertical')
        plt.suptitle('%s: Avg CBI' % self.prefix, size=14)

        png_path = os.path.join(self.output_dp, '%s_avgCBI_Filtered_KDE.png' % self.prefix)
        try:
            plt.savefig(png_path)
        except:
            plt.savefig('\\\\?\\' + png_path)
        plt.gcf().clear()
        plt.close()
        return

    def mt_plot_cbi_KDEs(self):
        from multiprocessing.dummy import Pool as ThreadPool
        pool = ThreadPool(4)

        xs = np.linspace(0, 3.5, 1000)
        def calc_gaussian_kde(array):
            return gaussian_kde(array)(xs)

        full_density, singular_density, nchildren_density, nparent_density, nsplit_density, highchastity_density, \
        singlecall_density, nmixed_density, nsm_density = pool.map(calc_gaussian_kde,
                                                                   [self.naCBI_data, self.naCBI_data[self.singular],
                                                                    self.naCBI_data[
                                                                        self.label_arr[label_dict['Parents']] == 0],
                                                                    self.naCBI_data[
                                                                        self.label_arr[label_dict['Children']] == 0],
                                                                    self.naCBI_data[self.non_split],
                                                                    self.naCBI_data[
                                                                        self.label_arr[label_dict['Chastity']] >= 7],
                                                                    self.naCBI_data[
                                                                        self.label_arr[label_dict['Multicall']] == 1],
                                                                    self.naCBI_data[self.non_mixed],
                                                                    self.naCBI_data[self.non_split_and_non_mixed]])

        fig, ax = plt.subplots(2, sharex=True, figsize=(12.8, 9.6))
        mpl.rcParams.update({'font.size': 10})

        ax[0].plot(xs, full_density, color='C0', label='All DNBs')
        ax[0].plot(xs, singular_density, color='C1', label='Singular DNBs')
        # empty, small, large
        ax[0].fill_between(xs, full_density, where=xs <= self.empty_fth, facecolor='red', alpha=0.4)
        ax[0].fill_between(xs, full_density, where=np.logical_and(xs > self.empty_fth, xs <= self.small_fth),
                           facecolor='blue',
                           alpha=0.4)
        ax[0].fill_between(xs, full_density, where=np.logical_and(xs > self.large_fth, xs <= self.outlier_fth),
                           facecolor='green',
                           alpha=0.4)
        ax[0].annotate('Empty (< {:,.2}): {:,.3}%'.format(self.empty_fth, self.empty_PofT) + \
                       '\nSmall (range({:,.2}, {:,.2})): {:,.3}%'.format(self.empty_fth, self.small_fth, self.small_PofT) + \
                       '\nLarge (> {:,.2}): {:,.3}%'.format(self.large_fth, self.large_PofT) + \
                       '\nHigh Intensity Outliers: {:,.3}%'.format(self.outlier_PofT),
                       xy=(0.9925, 0.83), xycoords='axes fraction', horizontalalignment='right',
                       verticalalignment='top')
        ax[0].xaxis.set_ticks_position('bottom')
        ax[0].yaxis.set_ticks_position('left')
        ax[0].legend(markerscale=20)

        ax[1].plot(xs, full_density, color='C0', label='All DNBs')
        ax[1].plot(xs, singular_density, color='C1', label='Singular DNBs')
        # 0.25, 0.5, 1.5
        ax[1].fill_between(xs, full_density, where=xs < 0.25, facecolor='red', alpha=0.4)
        ax[1].fill_between(xs, full_density, where=np.logical_and(xs >= 0.25, xs < 0.50), facecolor='blue', alpha=0.4)
        ax[1].fill_between(xs, full_density, where=xs > 1.50, facecolor='green', alpha=0.4)
        ax[1].annotate('DNBs < 0.25: {:,.3}%'.format(self.arb1_PofT) + \
                       '\nDNBs in range(0.25, 0.5): {:,.3}%'.format(self.arb2_PofT) + \
                       '\nDNBs in range(1.5, 3): {:,.3}%'.format(self.arb3_PofT),
                       xy=(0.9925, 0.83), xycoords='axes fraction', horizontalalignment='right',
                       verticalalignment='top')
        ax[1].xaxis.set_ticks_position('bottom')
        ax[1].yaxis.set_ticks_position('left')
        ax[1].legend(markerscale=20)

        fig.text(0.5, 0.04, 'CBI', ha='center', va='center')
        fig.text(0.03, 0.5, 'Density', ha='center', va='center', rotation='vertical')
        plt.suptitle('%s: Avg CBI' % self.prefix, size=14)

        png_path = os.path.join(self.output_dp, '%s_avgCBI_Sizing_KDE.png' % self.prefix)
        try:
            plt.savefig(png_path)
        except:
            plt.savefig('\\\\?\\' + png_path)
        plt.gcf().clear()
        plt.close()

        fig, ax = plt.subplots(3, sharex=True, figsize=(12.8, 9.6))
        mpl.rcParams.update({'font.size': 10})

        ax[0].plot(xs, full_density, color='C0', label='All DNBs')
        ax[0].plot(xs, nchildren_density, color='C1', label='Non-Children DNBs')
        ax[0].plot(xs, nparent_density, color='C2', label='Non-Parent DNBs')
        ax[0].plot(xs, nsplit_density, color='C4', label='Non-Split DNBs')
        ax[0].axvline(x=self.empty_fth, color='red', linestyle='--')
        ax[0].xaxis.set_ticks_position('bottom')
        ax[0].yaxis.set_ticks_position('left')
        ax[0].legend(markerscale=20)

        ax[1].plot(xs, full_density, color='C0', label='All DNBs')
        ax[1].plot(xs, highchastity_density, color='C1', label='High Chastity DNBs')
        ax[1].plot(xs, singlecall_density, color='C2', label='Single-call DNBs')
        ax[1].plot(xs, nmixed_density, color='C4', label='Non-Mixed DNBs')
        ax[1].axvline(x=self.empty_fth, color='red', linestyle='--')
        ax[1].xaxis.set_ticks_position('bottom')
        ax[1].yaxis.set_ticks_position('left')
        ax[1].legend(markerscale=20)

        ax[2].plot(xs, full_density, color='C0', label='All DNBs')
        ax[2].plot(xs, nsplit_density, color='C1', label='Non-Split DNBs')
        ax[2].plot(xs, nmixed_density, color='C2', label='Non-Mixed DNBs')
        ax[2].plot(xs, nsm_density, color='C4', label='Non-Mixed/Split DNBs')
        ax[2].axvline(x=self.empty_fth, color='red', linestyle='--')
        ax[2].xaxis.set_ticks_position('bottom')
        ax[2].yaxis.set_ticks_position('left')
        ax[2].legend(markerscale=20)

        fig.text(0.5, 0.04, 'CBI', ha='center', va='center')
        fig.text(0.03, 0.5, 'Density', ha='center', va='center', rotation='vertical')
        plt.suptitle('%s: Avg CBI' % self.prefix, size=14)

        png_path = os.path.join(self.output_dp, self.center_str + '%s_avgCBI_Filtered_KDE.png' % self.prefix)
        try:
            plt.savefig(png_path)
        except:
            plt.savefig('\\\\?\\' + png_path)
        plt.gcf().clear()
        plt.close()
        return

    def plot_cbi_hist(self):
        xs = np.linspace(0, 3.5, 1000)

        full_density = self.naCBI_data
        singular_density = self.naCBI_data[self.singular]

        fig, ax = plt.subplots(2, sharex=True, figsize=(12.8, 9.6))
        mpl.rcParams.update({'font.size': 10})

        full_counts, _, _ = ax[0].hist(full_density, xs, color='C0', histtype='step', label='All DNBs')
        singular_counts, _, _ = ax[0].hist(singular_density, xs, color='C1', histtype='step', label='Singular DNBs')

        ax[0].axvline(x=self.empty_fth, color='red', linestyle='--')
        ax[0].axvline(x=self.small_fth, color='blue', linestyle='--')
        ax[0].axvline(x=self.large_fth, color='green', linestyle='--')
        ax[0].axvline(x=self.outlier_fth, color='red', linestyle='--')

        ax[0].annotate('Empty [{:,.2}): {:,.3}%'.format(self.empty_fth, self.empty_PofT) + \
                       '\nSmall [{:,.2}, {:,.2}): {:,.3}%'.format(self.empty_fth, self.small_fth,
                                                                  self.small_PofT) + \
                       '\nLarge [{:,.2}, {:,.2}): {:,.3}%'.format(self.large_fth, self.outlier_fth, self.large_PofT) + \
                       '\nOutliers [{:,.2}): {:,.3}%'.format(self.outlier_fth, self.outlier_PofT),
                       xy=(0.9925, 0.82), xycoords='axes fraction', horizontalalignment='right',
                       verticalalignment='top')
        ax[0].xaxis.set_ticks_position('bottom')
        ax[0].yaxis.set_ticks_position('left')
        ax[0].legend(markerscale=20)

        ax[1].hist(self.naCBI_data, xs, color='C0', histtype='step', label='All DNBs')
        ax[1].hist(self.naCBI_data[self.singular], xs, color='C1', histtype='step', label='Singular DNBs')

        # arbitrary thresholds
        empty_ath = 0.25
        small_ath = 0.50
        large_ath = 1.5
        ax[1].axvline(x=empty_ath, color='red', linestyle='--')
        ax[1].axvline(x=small_ath, color='blue', linestyle='--')
        ax[1].axvline(x=large_ath, color='green', linestyle='--')

        ax[1].annotate('Empty [{:,.2}): {:,.3}%'.format(empty_ath, self.arb1_PofT) + \
                       '\nSmall [{:,.2}, {:,.2}): {:,.3}%'.format(empty_ath, small_ath,
                                                                  self.arb2_PofT) + \
                       '\nLarge [{:,.2}): {:,.3}%'.format(large_ath, self.arb3_PofT),
                       xy=(0.9925, 0.82), xycoords='axes fraction', horizontalalignment='right',
                       verticalalignment='top')
        ax[1].xaxis.set_ticks_position('bottom')
        ax[1].yaxis.set_ticks_position('left')
        ax[1].legend(markerscale=20)

        fig.text(0.5, 0.04, 'CBI', ha='center', va='center')
        fig.text(0.03, 0.5, 'Density', ha='center', va='center', rotation='vertical')
        plt.suptitle('%s: Avg CBI' % self.prefix, size=14)

        png_path = os.path.join(self.output_dp, self.center_str + '%s_avgCBI_Sizing_Hist.png' % self.prefix)
        try:
            plt.savefig(png_path)
        except:
            plt.savefig('\\\\?\\' + png_path)
        plt.gcf().clear()
        plt.close()

        # split breakdown
        nchildren_density = self.naCBI_data[self.label_arr[label_dict['Parents']] == 0]
        nparent_density = self.naCBI_data[self.label_arr[label_dict['Children']] == 0]
        nsplit_density = self.naCBI_data[self.non_split]

        # mixed breakdown
        lowshi_density = self.naCBI_data[self.label_arr[label_dict['SHI']] < 3]
        highchastity_density = self.naCBI_data[self.label_arr[label_dict['Chastity']] >= 7]
        singlecall_density = self.naCBI_data[self.label_arr[label_dict['Multicall']] == 1]
        nmixed_density = self.naCBI_data[self.non_mixed]

        # combined breakdown
        nsm_density = self.naCBI_data[self.non_split_and_non_mixed]

        fig, ax = plt.subplots(3, sharex=True, figsize=(12.8, 9.6))
        mpl.rcParams.update({'font.size': 10})

        ax[0].hist(full_density, xs, histtype='step', color='C0', label='All DNBs')
        nchildren_counts, _, _ = ax[0].hist(nchildren_density, xs, histtype='step', color='C1',
                                            label='Non-Children DNBs')
        nparent_counts, _, _ = ax[0].hist(nparent_density, xs, histtype='step', color='C2', label='Non-Parent DNBs')
        nsplit_counts, _, _ = ax[0].hist(nsplit_density, xs, histtype='step', color='C4', label='Non-Split DNBs')
        ax[0].axvline(x=self.empty_fth, color='red', linestyle='--')
        ax[0].xaxis.set_ticks_position('bottom')
        ax[0].yaxis.set_ticks_position('left')
        ax[0].legend(markerscale=20)

        ax[1].hist(full_density, xs, histtype='step', color='C0', label='All DNBs')
        lowshi_counts, _, _ = ax[1].hist(lowshi_density, xs, histtype='step', color='C5', label='Low SHI DNBs')
        highchas_counts, _, _ = ax[1].hist(highchastity_density, xs, histtype='step', color='C1',
                                           label='High Chastity DNBs')
        singlecall_counts, _, _ = ax[1].hist(singlecall_density, xs, histtype='step', color='C2',
                                             label='Single-call DNBs')
        nmixed_counts, _, _ = ax[1].hist(nmixed_density, xs, histtype='step', color='C4', label='Non-Mixed DNBs')
        ax[1].axvline(x=self.empty_fth, color='red', linestyle='--')
        ax[1].xaxis.set_ticks_position('bottom')
        ax[1].yaxis.set_ticks_position('left')
        ax[1].legend(markerscale=20)

        ax[2].hist(full_density, xs, histtype='step', color='C0', label='All DNBs')
        ax[2].hist(nsplit_density, xs, histtype='step', color='C1', label='Non-Split DNBs')
        ax[2].hist(nmixed_density, xs, histtype='step', color='C2', label='Non-Mixed DNBs')
        nsm_counts, _, _ = ax[2].hist(nsm_density, xs, histtype='step', color='C4', label='Non-Mixed/Split DNBs')
        ax[2].axvline(x=self.empty_fth, color='red', linestyle='--')
        ax[2].xaxis.set_ticks_position('bottom')
        ax[2].yaxis.set_ticks_position('left')
        ax[2].legend(markerscale=20)

        fig.text(0.5, 0.04, 'CBI', ha='center', va='center')
        fig.text(0.03, 0.5, 'Density', ha='center', va='center', rotation='vertical')
        plt.suptitle('%s: Avg CBI' % self.prefix, size=14)

        png_path = os.path.join(self.output_dp, self.center_str + '%s_avgCBI_Filtered_Hist.png' % self.prefix)
        try:
            plt.savefig(png_path)
        except:
            plt.savefig('\\\\?\\' + png_path)
        plt.gcf().clear()
        plt.close()

        counts = [full_counts, singular_counts,
                  nchildren_counts, nparent_counts, nsplit_counts,
                  lowshi_counts, highchas_counts, singlecall_counts, nmixed_counts, nsm_counts]
        counts = np.array(counts).astype(np.int16)
        np.save(self.avgCBI_hist_npy, counts)
        return

    def save_outputs(self, size_summary, size_results,
                     multicall_summary, multicall_results,
                     chastity_summary, chastity_results,
                     SHI_summary, SHI_results, mixed_summary,
                     empty_splits_results, mixed_splits_results,
                     familial_results,
                     singular_summary,
                     splits_summary, splits_results,
                     cbi_quartile_results, snr1_quartile_results, snr2_quartile_results):
        import cPickle as pickle
        
        with open(self.size_results_fp, 'w') as f:
            pickle.dump(size_results, f)
        with open(self.size_summary_fp, 'w') as f:
            pickle.dump(size_summary, f)
            
        with open(self.multicall_results_fp, 'w') as f:
            pickle.dump(multicall_results, f)
        with open(self.multicall_summary_fp, 'w') as f:
            pickle.dump(multicall_summary, f)
        
        with open(self.chastity_results_fp, 'w') as f:
            pickle.dump(chastity_results, f)
        with open(self.chastity_summary_fp, 'w') as f:
            pickle.dump(chastity_summary, f)

        with open(self.SHI_summary_fp, 'w') as f:
            pickle.dump(SHI_summary, f)
        with open(self.SHI_results_fp, 'w') as f:
            pickle.dump(SHI_results, f)

        with open(self.mixed_summary_fp, 'w') as f:
            pickle.dump(mixed_summary, f)
            
        with open(self.empty_splits_results_fp, 'w') as f:
            pickle.dump(empty_splits_results, f)
        with open(self.mixed_splits_results_fp, 'w') as f:
            pickle.dump(mixed_splits_results, f)
            
        with open(self.familial_results_fp, 'w') as f:
            pickle.dump(familial_results, f)
        with open(self.singular_summary_fp, 'w') as f:
            pickle.dump(singular_summary, f)
            
        with open(self.splits_results_fp, 'w') as f:
            pickle.dump(splits_results, f)
        with open(self.splits_summary_fp, 'w') as f:
            pickle.dump(splits_summary, f)
            
        with open(self.cbi_quartile_results_fp, 'w') as f:
            pickle.dump(cbi_quartile_results, f)
        with open(self.snr1_quartile_results_fp, 'w') as f:
            pickle.dump(snr1_quartile_results, f)
        with open(self.snr2_quartile_results_fp, 'w') as f:
            pickle.dump(snr2_quartile_results, f)
        return

    def run(self):
        logger.info('Initiating %s label analysis...' % self.fov)
        num_DNBs = len(self.naCBI_data)
        logger.info('num_DNBs: %s' % num_DNBs)
        logger.info('label_arr.shape: %s' % str(self.label_arr.shape))

        empty = self.naCBI_data < self.empty_fth
        arb15_empty = self.naCBI_data < 0.15
        arb20_empty = self.naCBI_data < 0.20
        arb25_empty = self.naCBI_data < 0.25
        small = np.logical_and(self.naCBI_data >= self.empty_fth, self.naCBI_data < self.small_fth)
        med = np.logical_and(self.naCBI_data >= self.small_fth, self.naCBI_data < self.large_fth)
        nonempty_025 = np.logical_and(self.naCBI_data >= self.empty_fth, self.naCBI_data < 0.25)
        nonempty_050 = np.logical_and(self.naCBI_data >= 0.25, self.naCBI_data < 0.50)
        large = np.logical_and(self.naCBI_data >= self.large_fth, self.naCBI_data < self.outlier_fth)
        outlier = self.naCBI_data >= self.outlier_fth

        # empty, small, large
        self.empty_PofT = 100. * np.sum(empty) / num_DNBs
        arb15_empty_PofT = 100. * np.sum(arb15_empty) / num_DNBs
        arb20_empty_PofT = 100. * np.sum(arb20_empty) / num_DNBs
        arb25_empty_PofT = 100. * np.sum(arb25_empty) / num_DNBs
        ne_025_PofT = 100. * np.sum(nonempty_025) / num_DNBs
        ne_050_PofT = 100. * np.sum(nonempty_050) / num_DNBs
        self.small_PofT = 100. * np.sum(small) / num_DNBs
        med_PofT = 100. * np.sum(med) / num_DNBs
        self.large_PofT = 100. * np.sum(large) / num_DNBs
        self.outlier_PofT = 100. * np.sum(outlier) / num_DNBs
        size_summary = [
            ['Empty (%ofTotal)', self.empty_PofT],
            ['<0.20 (%ofTotal)', arb20_empty_PofT],
            ['NE<0.25 (%ofTotal)', ne_025_PofT],
            ['NE<0.50 (%ofTotal)', ne_050_PofT]
        ]
        size_results = [
            ['<0.15 (%ofTotal)', arb15_empty_PofT],
            ['<0.20 (%ofTotal)', arb20_empty_PofT],
            ['<0.25 (%ofTotal)', arb25_empty_PofT],
            ['Empty (%ofTotal)', self.empty_PofT],
            ['Small (%ofTotal)', self.small_PofT],
            ['Med (%ofTotal)', med_PofT],
            ['Large (%ofTotal)', self.large_PofT],
            ['Outlier (%ofTotal)', self.outlier_PofT]
        ]

        # 0.25, 0.5
        arb1 = self.naCBI_data < 0.25
        arb2 = np.logical_and(self.naCBI_data >= 0.25, self.naCBI_data < 0.50)
        arb3 = self.naCBI_data > 1.50
        self.arb1_PofT = 100. * np.sum(arb1) / num_DNBs
        self.arb2_PofT = 100. * np.sum(arb2) / num_DNBs
        self.arb3_PofT = 100. * np.sum(arb3) / num_DNBs

        # 0 percentile rankings - positive (non-outlier, non-empty) in single percent bins
        # 1 calls per DNB - # of intensities above empty threshold for majority of cycles
        # 2 chastity level - highest chastity value for majority of cycles
        # 3 children counts - number of split neighbors with lower CBI
        # 4 parent counts - number of split neighbors with higher CBI
        # 5 mixed split counts - number of intensity-based matching split neighbors
        # 6 hidden (intensity-based) mixed splits - intensity-based matches different from sequence-based matching
        # 7 cluster size - number of splits around seed

        # quartiles
        cbi_q1 = np.logical_and(self.label_arr[label_dict['PercCBI']] > 0,
                                self.label_arr[label_dict['PercCBI']] <= 25)
        cbi_q2 = np.logical_and(self.label_arr[label_dict['PercCBI']] > 25,
                                self.label_arr[label_dict['PercCBI']] <= 50)
        cbi_q3 = np.logical_and(self.label_arr[label_dict['PercCBI']] > 50,
                                self.label_arr[label_dict['PercCBI']] <= 75)
        cbi_q4 = np.logical_and(self.label_arr[label_dict['PercCBI']] > 75,
                                self.label_arr[label_dict['PercCBI']] <= 100)
        empty_alt = self.label_arr[label_dict['PercCBI']] <= 0
        outliers = self.label_arr[label_dict['PercCBI']] > 100
        assert np.sum(cbi_q1 + cbi_q2 + cbi_q3 + cbi_q4 + empty_alt + outliers) == num_DNBs
        valid_DNBs = non_empty_and_non_outlier = np.logical_and(self.label_arr[label_dict['PercCBI']] > 0,
                                                                self.label_arr[label_dict['PercCBI']] <= 100)
        num_valid = np.sum(valid_DNBs)
        assert np.sum(cbi_q1 + cbi_q2 + cbi_q3 + cbi_q4) == num_valid

        # multicall breakdown
        c0 = self.label_arr[label_dict['Multicall']] == 0
        c1 = self.label_arr[label_dict['Multicall']] == 1
        c2 = self.label_arr[label_dict['Multicall']] == 2
        c3 = self.label_arr[label_dict['Multicall']] == 3
        c4 = self.label_arr[label_dict['Multicall']] == 4
        assert np.sum(c0 + c1 + c2 + c3 + c4) == num_DNBs
        cS = np.logical_or.reduce((c0, c1))
        cM = np.logical_or.reduce((c2, c3, c4))
        num_cM = np.sum(cM)

        vc0 = np.logical_and(self.label_arr[label_dict['Multicall']] == 0, valid_DNBs)
        vc1 = np.logical_and(self.label_arr[label_dict['Multicall']] == 1, valid_DNBs)
        vc2 = np.logical_and(self.label_arr[label_dict['Multicall']] == 2, valid_DNBs)
        vc3 = np.logical_and(self.label_arr[label_dict['Multicall']] == 3, valid_DNBs)
        vc4 = np.logical_and(self.label_arr[label_dict['Multicall']] == 4, valid_DNBs)
        assert np.sum(vc0 + vc1 + vc2 + vc3 + vc4) == num_valid
        vcM = np.logical_or.reduce((vc2, vc3, vc4))
        num_vcM = np.sum(vcM)

        multicalls_PofT = 100. * np.sum(cM) / num_DNBs
        valid_mc_PofV = 100. * np.sum(vcM) / num_valid
        c0_PofT = 100. * np.sum(c0) / num_DNBs
        c1_PofT = 100. * np.sum(c1) / num_DNBs
        c2_PofT = 100. * np.sum(c2) / num_DNBs
        c3_PofT = 100. * np.sum(c3) / num_DNBs
        c4_PofT = 100. * np.sum(c4) / num_DNBs
        vc0_PofV = 100. * np.sum(vc0) / num_valid
        vc1_PofV = 100. * np.sum(vc1) / num_valid
        vc2_PofV = 100. * np.sum(vc2) / num_valid
        vc3_PofV = 100. * np.sum(vc3) / num_valid
        vc4_PofV = 100. * np.sum(vc4) / num_valid
        multicall_summary = [
            ['MultiCall (%ofTotal)', multicalls_PofT],
            ['vMultiCall (%ofValid)', valid_mc_PofV]
        ]
        multicall_results = [
            ['0-Call (%ofTotal)', c0_PofT],
            ['1-Call (%ofTotal)', c1_PofT],
            ['2-Call (%ofTotal)', c2_PofT],
            ['3-Call (%ofTotal)', c3_PofT],
            ['4-Call (%ofTotal)', c4_PofT],
            ['v0-Call (%ofValid)', vc0_PofV],
            ['v1-Call (%ofValid)', vc1_PofV],
            ['v2-Call (%ofValid)', vc2_PofV],
            ['v3-Call (%ofValid)', vc3_PofV],
            ['v4-Call (%ofValid)', vc4_PofV]
        ]

        # chs breakdown
        chs_nan = self.label_arr[label_dict['Chastity']] == 0
        chs_5 = self.label_arr[label_dict['Chastity']] == 5
        chs_6 = self.label_arr[label_dict['Chastity']] == 6
        chs_7 = self.label_arr[label_dict['Chastity']] == 7
        chs_8 = self.label_arr[label_dict['Chastity']] == 8
        chs_9 = self.label_arr[label_dict['Chastity']] == 9
        chs_10 = self.label_arr[label_dict['Chastity']] == 10
        assert np.sum(chs_nan + chs_5 + chs_6 + chs_7 + chs_8 + chs_9 + chs_10) == num_DNBs
        low_chastity = np.logical_or.reduce((chs_nan, chs_5, chs_6))
        num_lc = np.sum(low_chastity)
        high_chastity = np.logical_or.reduce((chs_7, chs_8, chs_9, chs_10))

        vchs_nan = np.logical_and(self.label_arr[label_dict['Chastity']] == 0, valid_DNBs)
        vchs_5 = np.logical_and(self.label_arr[label_dict['Chastity']] == 5, valid_DNBs)
        vchs_6 = np.logical_and(self.label_arr[label_dict['Chastity']] == 6, valid_DNBs)
        vchs_7 = np.logical_and(self.label_arr[label_dict['Chastity']] == 7, valid_DNBs)
        vchs_8 = np.logical_and(self.label_arr[label_dict['Chastity']] == 8, valid_DNBs)
        vchs_9 = np.logical_and(self.label_arr[label_dict['Chastity']] == 9, valid_DNBs)
        vchs_10 = np.logical_and(self.label_arr[label_dict['Chastity']] == 10, valid_DNBs)
        assert np.sum(vchs_nan + vchs_5 + vchs_6 + vchs_7 + vchs_8 + vchs_9 + vchs_10) == num_valid
        vlow_chastity = np.logical_or.reduce((vchs_nan, vchs_5, vchs_6))
        num_vlc = np.sum(vlow_chastity)

        low_chastity_PofT = 100. * np.sum(low_chastity) / num_DNBs
        valid_lc_PofV = 100. * np.sum(vlow_chastity) / num_valid
        chastity_nan_PofT = 100. * np.sum(chs_nan) / num_DNBs
        chastity_05_PofT = 100. * np.sum(chs_5) / num_DNBs
        chastity_06_PofT = 100. * np.sum(chs_6) / num_DNBs
        chastity_07_PofT = 100. * np.sum(chs_7) / num_DNBs
        chastity_08_PofT = 100. * np.sum(chs_8) / num_DNBs
        chastity_09_PofT = 100. * np.sum(chs_9) / num_DNBs
        chastity_10_PofT = 100. * np.sum(chs_10) / num_DNBs
        vchastity_nan_PofV = 100. * np.sum(vchs_nan) / num_valid
        vchastity_05_PofV = 100. * np.sum(vchs_5) / num_valid
        vchastity_06_PofV = 100. * np.sum(vchs_6) / num_valid
        vchastity_07_PofV = 100. * np.sum(vchs_7) / num_valid
        vchastity_08_PofV = 100. * np.sum(vchs_8) / num_valid
        vchastity_09_PofV = 100. * np.sum(vchs_9) / num_valid
        vchastity_10_PofV = 100. * np.sum(vchs_10) / num_valid
        chastity_summary = [
            ['LowChastity (%ofTotal)', low_chastity_PofT],
            ['vLowChastity (%ofValid)', valid_lc_PofV]
        ]
        chastity_results = [
            ['NaN-Chastity (%ofTotal)', chastity_nan_PofT],
            ['0.5-Chastity (%ofTotal)', chastity_05_PofT],
            ['0.6-Chastity (%ofTotal)', chastity_06_PofT],
            ['0.7-Chastity (%ofTotal)', chastity_07_PofT],
            ['0.8-Chastity (%ofTotal)', chastity_08_PofT],
            ['0.9-Chastity (%ofTotal)', chastity_09_PofT],
            ['1.0-Chastity (%ofTotal)', chastity_10_PofT],
            ['vNaN-Chastity (%ofValid)', vchastity_nan_PofV],
            ['v0.5-Chastity (%ofValid)', vchastity_05_PofV],
            ['v0.6-Chastity (%ofValid)', vchastity_06_PofV],
            ['v0.7-Chastity (%ofValid)', vchastity_07_PofV],
            ['v0.8-Chastity (%ofValid)', vchastity_08_PofV],
            ['v0.9-Chastity (%ofValid)', vchastity_09_PofV],
            ['v1.0-Chastity (%ofValid)', vchastity_10_PofV]
             ]

        # SHI breakdown
        SHI_10 = self.label_arr[label_dict['SHI']] == 10
        SHI_9 = self.label_arr[label_dict['SHI']] == 9
        SHI_8 = self.label_arr[label_dict['SHI']] == 8
        SHI_7 = self.label_arr[label_dict['SHI']] == 7
        SHI_6 = self.label_arr[label_dict['SHI']] == 6
        SHI_5 = self.label_arr[label_dict['SHI']] == 5
        SHI_4 = self.label_arr[label_dict['SHI']] == 4
        SHI_3 = self.label_arr[label_dict['SHI']] == 3
        SHI_2 = self.label_arr[label_dict['SHI']] == 2
        SHI_1 = self.label_arr[label_dict['SHI']] == 1
        SHI_0 = self.label_arr[label_dict['SHI']] == 0
        assert np.sum(SHI_10 + SHI_9 + SHI_8 + SHI_7 + SHI_6 + SHI_5 + SHI_4 + SHI_3 + SHI_2 + SHI_1 + SHI_0) == \
               num_DNBs
        high_shi = np.logical_or.reduce((SHI_10, SHI_9, SHI_8, SHI_7, SHI_6, SHI_5, SHI_4, SHI_3))

        vSHI_10 = np.logical_and(self.label_arr[label_dict['SHI']] == 10, valid_DNBs)
        vSHI_9 = np.logical_and(self.label_arr[label_dict['SHI']] == 9, valid_DNBs)
        vSHI_8 = np.logical_and(self.label_arr[label_dict['SHI']] == 8, valid_DNBs)
        vSHI_7 = np.logical_and(self.label_arr[label_dict['SHI']] == 7, valid_DNBs)
        vSHI_6 = np.logical_and(self.label_arr[label_dict['SHI']] == 6, valid_DNBs)
        vSHI_5 = np.logical_and(self.label_arr[label_dict['SHI']] == 5, valid_DNBs)
        vSHI_4 = np.logical_and(self.label_arr[label_dict['SHI']] == 4, valid_DNBs)
        vSHI_3 = np.logical_and(self.label_arr[label_dict['SHI']] == 3, valid_DNBs)
        vSHI_2 = np.logical_and(self.label_arr[label_dict['SHI']] == 2, valid_DNBs)
        vSHI_1 = np.logical_and(self.label_arr[label_dict['SHI']] == 1, valid_DNBs)
        vSHI_0 = np.logical_and(self.label_arr[label_dict['SHI']] == 0, valid_DNBs)

        assert np.sum(vSHI_10 + vSHI_9 + vSHI_8 + vSHI_7 + vSHI_6 + vSHI_5 + vSHI_4 + vSHI_3 + vSHI_2 + vSHI_1 +
                      vSHI_0) == num_valid
        vhigh_shi = np.logical_or.reduce((vSHI_10, vSHI_9, vSHI_8, vSHI_7, vSHI_6, vSHI_5, vSHI_4, vSHI_3))

        high_shi_PofT = 100. * np.sum(high_shi) / num_DNBs
        valid_hs_PofV = 100. * np.sum(vhigh_shi) / num_valid

        SHI_0_PofT = 100. * np.sum(SHI_0) / num_DNBs
        SHI_1_PofT = 100. * np.sum(SHI_1) / num_DNBs
        SHI_2_PofT = 100. * np.sum(SHI_2) / num_DNBs
        SHI_3_PofT = 100. * np.sum(SHI_3) / num_DNBs
        SHI_4_PofT = 100. * np.sum(SHI_4) / num_DNBs
        SHI_5_PofT = 100. * np.sum(SHI_5) / num_DNBs
        SHI_6_PofT = 100. * np.sum(SHI_6) / num_DNBs
        SHI_7_PofT = 100. * np.sum(SHI_7) / num_DNBs
        SHI_8_PofT = 100. * np.sum(SHI_8) / num_DNBs
        SHI_9_PofT = 100. * np.sum(SHI_9) / num_DNBs
        SHI_10_PofT = 100. * np.sum(SHI_10) / num_DNBs

        vSHI_0_PofT = 100. * np.sum(vSHI_0) / num_valid
        vSHI_1_PofT = 100. * np.sum(vSHI_1) / num_valid
        vSHI_2_PofT = 100. * np.sum(vSHI_2) / num_valid
        vSHI_3_PofT = 100. * np.sum(vSHI_3) / num_valid
        vSHI_4_PofT = 100. * np.sum(vSHI_4) / num_valid
        vSHI_5_PofT = 100. * np.sum(vSHI_5) / num_valid
        vSHI_6_PofT = 100. * np.sum(vSHI_6) / num_valid
        vSHI_7_PofT = 100. * np.sum(vSHI_7) / num_valid
        vSHI_8_PofT = 100. * np.sum(vSHI_8) / num_valid
        vSHI_9_PofT = 100. * np.sum(vSHI_9) / num_valid
        vSHI_10_PofT = 100. * np.sum(vSHI_10) / num_valid

        SHI_summary = [
            ['HighSHI (%ofTotal)', high_shi_PofT],
            ['vHighSHI (%ofValid)', valid_hs_PofV]
        ]
        SHI_results = [
            ['[0 10)% SHI (%ofTotal)', SHI_0_PofT],
            ['[10 20)% SHI (%ofTotal)', SHI_1_PofT],
            ['[20 30)% SHI (%ofTotal)', SHI_2_PofT],
            ['[30 40)% SHI (%ofTotal)', SHI_3_PofT],
            ['[40 50)% SHI (%ofTotal)', SHI_4_PofT],
            ['[50 60)% SHI (%ofTotal)', SHI_5_PofT],
            ['[60 70)% SHI (%ofTotal)', SHI_6_PofT],
            ['[70 80)% SHI (%ofTotal)', SHI_7_PofT],
            ['[80 90)% SHI (%ofTotal)', SHI_8_PofT],
            ['[90 100)% SHI (%ofTotal)', SHI_9_PofT],
            ['100% SHI (%ofTotal)', SHI_10_PofT],
            ['[0 10)% vSHI (%ofValid)', vSHI_0_PofT],
            ['[10 20)% vSHI (%ofValid)', vSHI_1_PofT],
            ['[20 30)% vSHI (%ofValid)', vSHI_2_PofT],
            ['[30 40)% vSHI (%ofValid)', vSHI_3_PofT],
            ['[40 50)% vSHI (%ofValid)', vSHI_4_PofT],
            ['[50 60)% vSHI (%ofValid)', vSHI_5_PofT],
            ['[60 70)% vSHI (%ofValid)', vSHI_6_PofT],
            ['[70 80)% vSHI (%ofValid)', vSHI_7_PofT],
            ['[80 90)% vSHI (%ofValid)', vSHI_8_PofT],
            ['[90 100)% vSHI (%ofValid)', vSHI_9_PofT],
            ['100% vSHI (%ofValid)', vSHI_10_PofT]
        ]

        # children breakdown
        no_children = self.label_arr[label_dict['Children']] == 0
        single_child = self.label_arr[label_dict['Children']] == 1
        multi_children = self.label_arr[label_dict['Children']] > 1
        assert np.sum(no_children + single_child + multi_children) == num_DNBs, self.label_arr[label_dict['Children']]
        parents = np.logical_and(self.label_arr[label_dict['Children']] > 0, self.label_arr[label_dict['Parents']] == 0)
        vparents = np.logical_and(parents, valid_DNBs)
        empty_parents = np.logical_and(parents, self.label_arr[label_dict['PercCBI']] <= 0)
        c0_parents = np.logical_and(parents, self.label_arr[label_dict['Multicall']] == 0)

        # parents breakdown
        no_parents = self.label_arr[label_dict['Parents']] == 0
        single_parent = self.label_arr[label_dict['Parents']] == 1
        multi_parents = self.label_arr[label_dict['Parents']] > 1
        assert np.sum(no_parents + single_parent + multi_parents) == num_DNBs
        children = np.logical_and(self.label_arr[label_dict['Parents']] > 0,
                                  self.label_arr[label_dict['Children']] == 0)
        vchildren = np.logical_and(children, valid_DNBs)
        empty_children = np.logical_and(children, self.label_arr[label_dict['PercCBI']] <= 0)
        c0_children = np.logical_and(children, self.label_arr[label_dict['Multicall']] == 0)

        parent_and_child = np.logical_and(self.label_arr[label_dict['Parents']] > 0,
                                          self.label_arr[label_dict['Children']] > 0)
        vparent_and_child = np.logical_and(parent_and_child, valid_DNBs)
        empty_pc = np.logical_and(parent_and_child, self.label_arr[label_dict['PercCBI']] <= 0)
        c0_pc = np.logical_and(parent_and_child, self.label_arr[label_dict['Multicall']] == 0)

        empty_parents_PofT = 100. * np.sum(empty_parents) / num_DNBs
        c0_parents_PofT = 100. * np.sum(c0_parents) / num_DNBs
        empty_children_PofT = 100. * np.sum(empty_children) / num_DNBs
        c0_children_PofT = 100. * np.sum(c0_children) / num_DNBs
        c0_pc_PofT = 100. * np.sum(c0_pc) / num_DNBs

        empty_splits_results = [
            ['Empty Parents (%ofTotal)', empty_parents_PofT],
            ['0-Call Parents (%ofTotal)', c0_parents_PofT],
            ['Empty Children (%ofTotal)', empty_children_PofT],
            ['0-Call Children (%ofTotal)', c0_children_PofT]
        ]

        split = np.logical_or(self.label_arr[label_dict['Children']] != 0,
                              self.label_arr[label_dict['Parents']] != 0)
        valid_split = np.logical_and(split, valid_DNBs)
        self.non_split = np.logical_and(self.label_arr[label_dict['Children']] == 0,
                                        self.label_arr[label_dict['Parents']] == 0)
        mixed = np.logical_or(cM, low_chastity)
        valid_mixed = np.logical_and(valid_DNBs, mixed)
        num_mixed = np.sum(mixed)
        num_vmixed = np.sum(valid_mixed)
        self.non_mixed = np.logical_and(cS, high_chastity)
        self.non_split_and_non_mixed = np.logical_and(self.non_split, self.non_mixed)
        # singular is not split/mixed/empty/outlier
        self.singular = np.logical_and(self.non_split_and_non_mixed, non_empty_and_non_outlier)

        mixed_PofT = 100. * num_mixed / num_DNBs
        vmixed_PofV = 100. * num_vmixed / num_valid

        mixed_shi = np.logical_or(cM, high_shi)
        valid_mixed_shi = np.logical_and(valid_DNBs, mixed_shi)
        mixed_shi_PofT = 100. * np.sum(mixed_shi) / num_DNBs
        vmixed_shi_PofV = 100. * np.sum(valid_mixed_shi) / num_valid
        mixed_summary = [
            ['Mixed (%ofTotal)', mixed_PofT],
            ['vMixed (%ofValid)', vmixed_PofV],
            ['Mixed (MultiCall or HighSHI) (%ofTotal)', mixed_shi_PofT],
            ['vMixed (MultiCall or HighSHI) (%ofValid)', vmixed_shi_PofV]
        ]

        split_or_mixed = np.logical_or(split, mixed)
        small_singular = np.logical_and(small, self.non_split_and_non_mixed)
        small_split_or_mixed = np.logical_and(small, split_or_mixed)
        assert np.sum(small_singular + small_split_or_mixed) == np.sum(small)

        mixed_non_split = np.logical_and(mixed, self.non_split)
        mixed_children = np.logical_and(mixed, children)
        mixed_parents = np.logical_and(mixed, parents)
        mixed_parent_and_child = np.logical_and(mixed, parent_and_child)
        vmixed_non_split = np.logical_and(valid_DNBs, mixed_non_split)
        vmixed_children = np.logical_and(valid_DNBs, mixed_children)
        vmixed_parents = np.logical_and(valid_DNBs, mixed_parents)
        vmixed_parent_and_child = np.logical_and(valid_DNBs, mixed_parent_and_child)
        assert np.sum(mixed_non_split + mixed_children + mixed_parents + mixed_parent_and_child) == np.sum(mixed)
        if num_mixed != 0:
            non_split_PofM = 100. * np.sum(mixed_non_split) / num_mixed
            children_PofM = 100. * np.sum(mixed_children) / num_mixed
            parents_PofM = 100. * np.sum(mixed_parents) / num_mixed
            pc_PofM = 100. * np.sum(mixed_parent_and_child) / num_mixed
        else:
            non_split_PofM = 0
            children_PofM = 0
            parents_PofM = 0
            pc_PofM = 0
        if num_vmixed != 0:
            vnon_split_PofVM = 100. * np.sum(vmixed_non_split) / num_vmixed
            vchildren_PofVM = 100. * np.sum(vmixed_children) / num_vmixed
            vparents_PofVM = 100. * np.sum(vmixed_parents) / num_vmixed
            vpc_PofVM = 100. * np.sum(vmixed_parent_and_child) / num_vmixed
        else:
            vnon_split_PofVM = 0
            vchildren_PofVM = 0
            vparents_PofVM = 0
            vpc_PofVM = 0

        multicall_non_split = np.logical_and(cM, self.non_split)
        multicall_children = np.logical_and(cM, children)
        multicall_parents = np.logical_and(cM, parents)
        multicall_parent_and_child = np.logical_and(cM, parent_and_child)
        vmulticall_non_split = np.logical_and(valid_DNBs, multicall_non_split)
        vmulticall_children = np.logical_and(valid_DNBs, multicall_children)
        vmulticall_parents = np.logical_and(valid_DNBs, multicall_parents)
        vmulticall_parent_and_child = np.logical_and(valid_DNBs, multicall_parent_and_child)
        assert np.sum(multicall_non_split + multicall_children +
                      multicall_parents + multicall_parent_and_child) == np.sum(cM)
        if num_cM != 0:
            non_split_PofMC = 100. * np.sum(multicall_non_split) / num_cM
            children_PofMC = 100. * np.sum(multicall_children) / num_cM
            parents_PofMC = 100. * np.sum(multicall_parents) / num_cM
            pc_PofMC = 100. * np.sum(multicall_parent_and_child) / num_cM
        else:
            non_split_PofMC = 0
            children_PofMC = 0
            parents_PofMC = 0
            pc_PofMC = 0
        if num_vcM !=0:
            vnon_split_PofVMC = 100. * np.sum(vmulticall_non_split) / num_vcM
            vchildren_PofVMC = 100. * np.sum(vmulticall_children) / num_vcM
            vparents_PofVMC = 100. * np.sum(vmulticall_parents) / num_vcM
            vpc_PofVMC = 100. * np.sum(vmulticall_parent_and_child) / num_vcM
        else:
            vnon_split_PofVMC = 0
            vchildren_PofVMC = 0
            vparents_PofVMC = 0
            vpc_PofVMC = 0

        low_chastity_non_split = np.logical_and(low_chastity, self.non_split)
        low_chastity_children = np.logical_and(low_chastity, children)
        low_chastity_parents = np.logical_and(low_chastity, parents)
        low_chastity_parent_and_child = np.logical_and(low_chastity, parent_and_child)
        vlow_chastity_non_split = np.logical_and(valid_DNBs, low_chastity_non_split)
        vlow_chastity_children = np.logical_and(valid_DNBs, low_chastity_children)
        vlow_chastity_parents = np.logical_and(valid_DNBs, low_chastity_parents)
        vlow_chastity_parent_and_child = np.logical_and(valid_DNBs, low_chastity_parent_and_child)
        assert np.sum(low_chastity_non_split + low_chastity_children +
                      low_chastity_parents + low_chastity_parent_and_child) == np.sum(low_chastity)
        if num_lc !=0:
            non_split_PofLC = 100. * np.sum(low_chastity_non_split) / num_lc
            children_PofLC = 100. * np.sum(low_chastity_children) / num_lc
            parents_PofLC = 100. * np.sum(low_chastity_parents) / num_lc
            pc_PofLC = 100. * np.sum(low_chastity_parent_and_child) / num_lc
        else:
            non_split_PofLC = 0
            children_PofLC = 0
            parents_PofLC = 0
            pc_PofLC = 0
        if num_vlc != 0:
            vnon_split_PofVLC = 100. * np.sum(vlow_chastity_non_split) / num_vlc
            vchildren_PofVLC = 100. * np.sum(vlow_chastity_children) / num_vlc
            vparents_PofVLC = 100. * np.sum(vlow_chastity_parents) / num_vlc
            vpc_PofVLC = 100. * np.sum(vlow_chastity_parent_and_child) / num_vlc
        else:
            vnon_split_PofVLC = 0
            vchildren_PofVLC = 0
            vparents_PofVLC = 0
            vpc_PofVLC = 0

        # intensity-based split neighbors of mixed DNBs
        mixed_split0 = self.label_arr[label_dict['MixedSplit']] == 0
        mixed_split1 = self.label_arr[label_dict['MixedSplit']] == 1
        mixed_splitM = self.label_arr[label_dict['MixedSplit']] > 1
        # only 2-call DNBs are considered for mix splits due to false positive rate of 3+
        assert np.sum(mixed_split0 + mixed_split1 + mixed_splitM) == np.sum(c2), '%s %s' % (
            np.sum(mixed_split0 + mixed_split1 + mixed_splitM), np.sum(c2)
        )

        mixed_splits_results = [
            ['NonSplit Mixed (%ofMixed)', non_split_PofM],
            ['Mixed Children (%ofMixed)', children_PofM],
            ['Mixed Parents (%ofMixed)', parents_PofM],
            ['Mixed MidParent (%ofMixed)', pc_PofM],
            ['vNonSplit Mixed (%ofValidMixed)', vnon_split_PofVM],
            ['vMixed Children (%ofValidMixed)', vchildren_PofVM],
            ['vMixed Parents (%ofValidMixed)', vparents_PofVM],
            ['vMixed MidParent (%ofValidMixed)', vpc_PofVM],
            ['NonSplit (%ofMultiCall)', non_split_PofMC],
            ['mcChildren (%ofMultiCall)', children_PofMC],
            ['mcParents (%ofMultiCall)', parents_PofMC],
            ['mcMidParent (%ofMultiCall)', pc_PofMC],
            ['vNonSplit (%ofValidMultiCall)', vnon_split_PofVMC],
            ['vmcChildren (%ofValidMultiCall)', vchildren_PofVMC],
            ['vmcParents (%ofValidMultiCall)', vparents_PofVMC],
            ['vmcMidParent (%ofValidMultiCall)', vpc_PofVMC],
            ['lcNonSplit (%ofLowChastity)', non_split_PofLC],
            ['lcChildren (%ofLowChastity)', children_PofLC],
            ['lcParents (%ofLowChastity)', parents_PofLC],
            ['lcMidParent (%ofLowChastity)', pc_PofLC],
            ['vlcNonSplit (%ofValidLowChastity)', vnon_split_PofVLC],
            ['vlcChildren (%ofValidLowChastity)', vchildren_PofVLC],
            ['vlcParents (%ofValidLowChastity)', vparents_PofVLC],
            ['vlcMidParent (%ofValidLowChastity)', vpc_PofVLC]
        ]

        # hidden splits
        mixed_split10 = np.logical_and(self.label_arr[label_dict['MixedSplit']] == 1,
                                       self.label_arr[label_dict['HiddenSplit']] == 0)
        mixed_split11 = np.logical_and(self.label_arr[label_dict['MixedSplit']] == 1,
                                       self.label_arr[label_dict['HiddenSplit']] == 1)
        assert np.sum(mixed_split10 + mixed_split11) == np.sum(mixed_split1)
        mixed_splitM0 = np.logical_and(self.label_arr[label_dict['MixedSplit']] > 1,
                                       self.label_arr[label_dict['HiddenSplit']] == 0)
        mixed_splitM1 = np.logical_and(self.label_arr[label_dict['MixedSplit']] > 1,
                                       self.label_arr[label_dict['HiddenSplit']] == 1)
        mixed_splitMM = np.logical_and(self.label_arr[label_dict['MixedSplit']] > 1,
                                       self.label_arr[label_dict['HiddenSplit']] > 1)
        assert np.sum(mixed_splitM0 + mixed_splitM1 + mixed_splitMM) == np.sum(mixed_splitM)

        # the seed is the highest CBI in a cluster - they are designated by number in their cluster
        seeds = self.label_arr[label_dict['FamilySize']] > 0
        num_families = np.sum(seeds)
        num_familial_DNBs = np.sum(self.label_arr[label_dict['FamilySize']][seeds])
        # families have overlapping intensities and/or a shared called sequence
        fmly2 = self.label_arr[label_dict['FamilySize']][self.label_arr[label_dict['FamilySize']] == 1]
        fmly3 = self.label_arr[label_dict['FamilySize']][self.label_arr[label_dict['FamilySize']] == 2]
        fmly4 = self.label_arr[label_dict['FamilySize']][self.label_arr[label_dict['FamilySize']] == 3]
        fmly5 = self.label_arr[label_dict['FamilySize']][self.label_arr[label_dict['FamilySize']] == 4]
        fmly6 = self.label_arr[label_dict['FamilySize']][self.label_arr[label_dict['FamilySize']] == 5]
        fmly7 = self.label_arr[label_dict['FamilySize']][self.label_arr[label_dict['FamilySize']] == 6]
        fmly8 = self.label_arr[label_dict['FamilySize']][self.label_arr[label_dict['FamilySize']] == 7]
        fmly9 = self.label_arr[label_dict['FamilySize']][self.label_arr[label_dict['FamilySize']] == 8]
        fmly9plus = self.label_arr[label_dict['FamilySize']][self.label_arr[label_dict['FamilySize']] > 8]
        fmly2_PofT = 100. * np.sum(fmly2) / num_DNBs
        if num_familial_DNBs != 0:
            fmly2_PofCl = 100. * np.sum(fmly2) / num_familial_DNBs
        else:
            fmly2_PofCl = 0
        fmly3_PofT = 100. * np.sum(fmly3) / num_DNBs
        fmly4_PofT = 100. * np.sum(fmly4) / num_DNBs
        fmly5_PofT = 100. * np.sum(fmly5) / num_DNBs
        fmly6_PofT = 100. * np.sum(fmly6) / num_DNBs
        fmly7_PofT = 100. * np.sum(fmly7) / num_DNBs
        fmly8_PofT = 100. * np.sum(fmly8) / num_DNBs
        fmly9_PofT = 100. * np.sum(fmly9) / num_DNBs
        fmly9plus_PofT = 100. * np.sum(fmly9plus) / num_DNBs
        familial_results = [
            ['Families Count', num_families],
            ['2n DNBs (%ofTotal)', fmly2_PofT],
            ['2n DNBs (%ofFamilies)', fmly2_PofCl],
            ['3n DNBs (%ofTotal)', fmly3_PofT],
            ['4n DNBs (%ofTotal)', fmly4_PofT],
            ['5n DNBs (%ofTotal)', fmly5_PofT],
            ['6n DNBs (%ofTotal)', fmly6_PofT],
            ['7n DNBs (%ofTotal)', fmly7_PofT],
            ['8n DNBs (%ofTotal)', fmly8_PofT],
            ['9n DNBs (%ofTotal)', fmly9_PofT],
            ['9+n DNBs (%ofTotal)', fmly9plus_PofT]
        ]

        non_mixed_seeds = np.logical_and(seeds, self.non_mixed)
        small_non_mixed_seeds = np.logical_and(small, non_mixed_seeds)
        seeds_and_singular = np.logical_or(non_mixed_seeds, self.singular)
        seeds_and_singular_PofT = 100. * np.sum(seeds_and_singular) / num_DNBs
        singular_PofT = 100. * np.sum(self.singular) / num_DNBs
        snm_seeds_and_sm_singular = np.logical_or(small_non_mixed_seeds, small_singular)
        # seeds and singular *should* be exclusive, so the sum of the two should equal the union
        assert np.sum(small_non_mixed_seeds) + np.sum(small_singular) == np.sum(snm_seeds_and_sm_singular), \
            '%s + %s != %s' % (np.sum(small_non_mixed_seeds), np.sum(small_singular), np.sum(snm_seeds_and_sm_singular))
        small_sss_PofSm = 100. * np.sum(snm_seeds_and_sm_singular) / np.sum(small) if np.sum(small) else 'NA'
        small_singular_PofSm = 100. * np.sum(small_singular) / np.sum(small) if np.sum(small) else 'NA'
        singular_summary = [
            ['Seeds+Singular (%ofTotal)', seeds_and_singular_PofT],
            ['Singular (%ofTotal)', singular_PofT],
            ['smSeeds+smSingular (%ofSmall)', small_sss_PofSm],
            ['smSingular (%ofSmall)', small_singular_PofSm]
        ]

        non_seed_splits = np.logical_and(split, ~seeds)
        # seq_split are non-seed splits that match at least one neighbor by sequence
        # seq_split are so called because they do not provide non-redundant sequence information
        seq_split = np.logical_and(non_seed_splits,
                                 np.logical_or(self.label_arr[label_dict['MixedSplit']] == -1,
                                               self.label_arr[label_dict['HiddenSplit']] <
                                               self.label_arr[label_dict['MixedSplit']]))
        shady_nseed_splits = np.logical_and(non_seed_splits, self.label_arr[label_dict['HiddenSplit']] > 0)

        valid_nseed_split = np.logical_and(non_seed_splits, valid_DNBs)
        valid_seq_split = np.logical_and(valid_nseed_split,
                                       np.logical_or(self.label_arr[label_dict['MixedSplit']] == -1,
                                                     self.label_arr[label_dict['HiddenSplit']] <
                                                     self.label_arr[label_dict['MixedSplit']]))
        valid_snseed_split = np.logical_and(valid_nseed_split, self.label_arr[label_dict['HiddenSplit']] > 0)

        seq_split_PofT = 100. * np.sum(seq_split) / num_DNBs
        hns_split_PofT = 100. * np.sum(shady_nseed_splits) / num_DNBs
        vseq_split_PofV = 100. * np.sum(valid_seq_split) / num_valid
        vhns_split_PofV = 100. * np.sum(valid_snseed_split) / num_valid

        split_PofT = 100. * np.sum(split) / num_DNBs
        vsplit_PofV = 100. * np.sum(valid_split) / num_valid
        # child splits
        children_PofT = 100. * np.sum(children) / num_DNBs
        vchildren_PofV = 100. * np.sum(vchildren) / num_valid
        # parents splits
        parents_PofT = 100. * np.sum(parents) / num_DNBs
        vparents_PofV = 100. * np.sum(vparents) / num_valid
        # child+parent splits
        parent_and_child_PofT = 100. * np.sum(parent_and_child) / num_DNBs
        vparent_and_child_PofV = 100. * np.sum(vparent_and_child) / num_valid
        splits_summary = [
            ['Seq Splits (%ofTotal)', seq_split_PofT],
            ['Int Splits (%ofTotal)', hns_split_PofT],
            ['vSeq Splits (%ofValid)', vseq_split_PofV],
            ['vInt Splits (%ofValid)', vhns_split_PofV]
        ]
        splits_results = [
            ['Splits (%ofTotal)', split_PofT],
            ['vSplits (%ofValid)', vsplit_PofV],
            ['Children (%ofTotal)', children_PofT],
            ['vChildren (%ofValid)', vchildren_PofV],
            ['Parents (%ofTotal)', parents_PofT],
            ['vParents (%ofValid)', vparents_PofV],
            ['MidParent (%ofTotal)', parent_and_child_PofT],
            ['vMidParent (%ofValid)', vparent_and_child_PofV]
        ]

        # new quartiles
        snr1_q1 = np.logical_and(valid_DNBs, np.logical_and(self.label_arr[label_dict['PercReadSNR']] >= 0,
                                                            self.label_arr[label_dict['PercReadSNR']] <= 25))
        snr1_q2 = np.logical_and(valid_DNBs, np.logical_and(self.label_arr[label_dict['PercReadSNR']] > 25,
                                                            self.label_arr[label_dict['PercReadSNR']] <= 50))
        snr1_q3 = np.logical_and(valid_DNBs, np.logical_and(self.label_arr[label_dict['PercReadSNR']] > 50,
                                                            self.label_arr[label_dict['PercReadSNR']] <= 75))
        snr1_q4 = np.logical_and(valid_DNBs, np.logical_and(self.label_arr[label_dict['PercReadSNR']] > 75,
                                                            self.label_arr[label_dict['PercReadSNR']] <= 100))
        assert np.sum(snr1_q1 + snr1_q2 + snr1_q3 + snr1_q4) == num_valid

        snr2_q1 = np.logical_and(valid_DNBs, np.logical_and(self.label_arr[label_dict['PercCycleSNR']] >= 0,
                                                            self.label_arr[label_dict['PercCycleSNR']] <= 25))
        snr2_q2 = np.logical_and(valid_DNBs, np.logical_and(self.label_arr[label_dict['PercCycleSNR']] > 25,
                                                            self.label_arr[label_dict['PercCycleSNR']] <= 50))
        snr2_q3 = np.logical_and(valid_DNBs, np.logical_and(self.label_arr[label_dict['PercCycleSNR']] > 50,
                                                            self.label_arr[label_dict['PercCycleSNR']] <= 75))
        snr2_q4 = np.logical_and(valid_DNBs, np.logical_and(self.label_arr[label_dict['PercCycleSNR']] > 75,
                                                            self.label_arr[label_dict['PercCycleSNR']] <= 100))
        assert np.sum(snr2_q1 + snr2_q2 + snr2_q3 + snr2_q4) == num_valid

        quartile_metrics = [
            'vMixed (%ofValid)',
            'vMultiCall (%ofValid)',
            'vLowChastity (%ofValid)',
            'v1-Call (%ofValid)',
            'v2-Call (%ofValid)',
            'v3-Call (%ofValid)',
            'v4-Call (%ofValid)',
            'vNaN-Chas (%ofValid)',
            'v0.5-Chas (%ofValid)',
            'v0.6-Chas (%ofValid)',
            'v0.7-Chas (%ofValid)',
            'v0.8-Chas (%ofValid)',
            'v0.9-Chas (%ofValid)',
            'v1.0-Chas (%ofValid)',
            '[0 10)% vSHI (%ofValid)',
            '[10 20)% vSHI (%ofValid)',
            '[20 30)% vSHI (%ofValid)',
            '[30 40)% vSHI (%ofValid)',
            '[40 50)% vSHI (%ofValid)',
            '[50 60)% vSHI (%ofValid)',
            '[60 70)% vSHI (%ofValid)',
            '[70 80)% vSHI (%ofValid)',
            '[80 90)% vSHI (%ofValid)',
            '[90 100)% vSHI (%ofValid)',
            '100% vSHI (%ofValid)',
            'vNon-Split (%ofValid)',
            'vChildren (%ofValid)',
            'vParents (%ofValid)',
            'vParent&Child (%ofValid)',
            'vNon-Split-Mixed (%ofValid)',
            'vMixed-Children (%ofValid)',
            'vMixed-Parents (%ofValid)',
            'vMixed-Parent&Child (%ofValid)'
        ]

        qtrl_dict = {
            'cbi': {
                1: cbi_q1,
                2: cbi_q2,
                3: cbi_q3,
                4: cbi_q4,
                'results': [[m] for m in quartile_metrics]
            },
            'snr1': {
                1: snr1_q1,
                2: snr1_q2,
                3: snr1_q3,
                4: snr1_q4,
                'results': [[m] for m in quartile_metrics]
            },
            'snr2': {
                1: snr2_q1,
                2: snr2_q2,
                3: snr2_q3,
                4: snr2_q4,
                'results': [[m] for m in quartile_metrics]
            }
        }

        for mode in ['cbi', 'snr1', 'snr2']:
            for qtrl in [1,2,3,4]:
                subset = qtrl_dict[mode][qtrl]

                subset_mixed = np.logical_and(subset, mixed)
                subset_mixed_PofV = 100. * np.sum(subset_mixed) / num_valid
                subset_cM = np.logical_and(subset, cM)
                subset_cM_PofV = 100. * np.sum(subset_cM) / num_valid
                subset_low_chastity = np.logical_and(subset, low_chastity)
                subset_low_chastity_PofV = 100. * np.sum(subset_low_chastity) / num_valid

                subset_c1 = np.logical_and(subset, c1)
                subset_c1_PofV = 100. * np.sum(subset_c1) / num_valid
                subset_c2 = np.logical_and(subset, c2)
                subset_c2_PofV = 100. * np.sum(subset_c2) / num_valid
                subset_c3 = np.logical_and(subset, c3)
                subset_c3_PofV = 100. * np.sum(subset_c3) / num_valid
                subset_c4 = np.logical_and(subset, c4)
                subset_c4_PofV = 100. * np.sum(subset_c4) / num_valid

                subset_chs_nan = np.logical_and(subset, chs_nan)
                subset_chs_nan_PofV = 100. * np.sum(subset_chs_nan) / num_valid
                subset_chs_5 = np.logical_and(subset, chs_5)
                subset_chs_5_PofV = 100. * np.sum(subset_chs_5) / num_valid
                subset_chs_6 = np.logical_and(subset, chs_6)
                subset_chs_6_PofV = 100. * np.sum(subset_chs_6) / num_valid
                subset_chs_7 = np.logical_and(subset, chs_7)
                subset_chs_7_PofV = 100. * np.sum(subset_chs_7) / num_valid
                subset_chs_8 = np.logical_and(subset, chs_8)
                subset_chs_8_PofV = 100. * np.sum(subset_chs_8) / num_valid
                subset_chs_9 = np.logical_and(subset, chs_9)
                subset_chs_9_PofV = 100. * np.sum(subset_chs_9) / num_valid
                subset_chs_10 = np.logical_and(subset, chs_10)
                subset_chs_10_PofV = 100. * np.sum(subset_chs_10) / num_valid

                subset_SHI_10 = np.logical_and(subset, SHI_10)
                subset_SHI_10_PofV = 100. * np.sum(subset_SHI_10) / num_valid
                subset_SHI_9 = np.logical_and(subset, SHI_9)
                subset_SHI_9_PofV = 100. * np.sum(subset_SHI_9) / num_valid
                subset_SHI_8 = np.logical_and(subset, SHI_8)
                subset_SHI_8_PofV = 100. * np.sum(subset_SHI_8) / num_valid
                subset_SHI_7 = np.logical_and(subset, SHI_7)
                subset_SHI_7_PofV = 100. * np.sum(subset_SHI_7) / num_valid
                subset_SHI_6 = np.logical_and(subset, SHI_6)
                subset_SHI_6_PofV = 100. * np.sum(subset_SHI_6) / num_valid
                subset_SHI_5 = np.logical_and(subset, SHI_5)
                subset_SHI_5_PofV = 100. * np.sum(subset_SHI_5) / num_valid
                subset_SHI_4 = np.logical_and(subset, SHI_4)
                subset_SHI_4_PofV = 100. * np.sum(subset_SHI_4) / num_valid
                subset_SHI_3 = np.logical_and(subset, SHI_3)
                subset_SHI_3_PofV = 100. * np.sum(subset_SHI_3) / num_valid
                subset_SHI_2 = np.logical_and(subset, SHI_2)
                subset_SHI_2_PofV = 100. * np.sum(subset_SHI_2) / num_valid
                subset_SHI_1 = np.logical_and(subset, SHI_1)
                subset_SHI_1_PofV = 100. * np.sum(subset_SHI_1) / num_valid
                subset_SHI_0 = np.logical_and(subset, SHI_0)
                subset_SHI_0_PofV = 100. * np.sum(subset_SHI_0) / num_valid

                subset_non_split = np.logical_and(subset, self.non_split)
                subset_non_split_PofV = 100. * np.sum(subset_non_split) / num_valid
                subset_children = np.logical_and(subset, children)
                subset_children_PofV = 100. * np.sum(subset_children) / num_valid
                subset_parents = np.logical_and(subset, parents)
                subset_parents_PofV = 100. * np.sum(subset_parents) / num_valid
                subset_parent_and_child = np.logical_and(subset, parent_and_child)
                subset_parent_and_child_PofV = 100. * np.sum(subset_parent_and_child) / num_valid

                subset_mixed_non_split = np.logical_and(subset, mixed_non_split)
                subset_mixed_non_split_PofV = 100. * np.sum(subset_mixed_non_split) / num_valid
                subset_mixed_children = np.logical_and(subset, mixed_children)
                subset_mixed_children_PofV = 100. * np.sum(subset_mixed_children) / num_valid
                subset_mixed_parents = np.logical_and(subset, mixed_parents)
                subset_mixed_parents_PofV = 100. * np.sum(subset_mixed_parents) / num_valid
                subset_mixed_parent_and_child = np.logical_and(subset, mixed_parent_and_child)
                subset_mixed_parent_and_child_PofV = 100. * np.sum(subset_mixed_parent_and_child) / num_valid

                subset_results = [subset_mixed_PofV, subset_cM_PofV, subset_low_chastity_PofV,
                                  subset_c1_PofV, subset_c2_PofV, subset_c3_PofV, subset_c4_PofV,
                                  subset_chs_nan_PofV, subset_chs_5_PofV, subset_chs_6_PofV, subset_chs_7_PofV,
                                  subset_chs_8_PofV, subset_chs_9_PofV, subset_chs_10_PofV,
                                  subset_SHI_0_PofV, subset_SHI_1_PofV, subset_SHI_2_PofV,
                                  subset_SHI_3_PofV, subset_SHI_4_PofV, subset_SHI_5_PofV, subset_SHI_6_PofV,
                                  subset_SHI_7_PofV, subset_SHI_8_PofV, subset_SHI_9_PofV, subset_SHI_10_PofV,
                                  subset_non_split_PofV, subset_children_PofV, subset_parents_PofV,
                                  subset_parent_and_child_PofV, subset_mixed_non_split_PofV, subset_mixed_children_PofV,
                                  subset_mixed_parents_PofV, subset_mixed_parent_and_child_PofV]

                assert len(subset_results) == len(qtrl_dict[mode]['results']), 'Mismatch in length of quartile metrics!'

                for m, metric in enumerate(qtrl_dict[mode]['results']):
                    metric.append(subset_results[m])

        # plots
        logger.info('Plotting...')
        # if not self.center:
        if not self.bypass['plot_cbi_KDEs']:
            #self.plot_cbi_KDEs()
            self.mt_plot_cbi_KDEs()
        if not self.bypass['plot_cbi_hist']:
            self.plot_cbi_hist()

        self.save_outputs(size_summary, size_results,
                          multicall_summary, multicall_results,
                          chastity_summary, chastity_results,
                          SHI_summary, SHI_results, mixed_summary,
                          empty_splits_results, mixed_splits_results,
                          familial_results,
                          singular_summary,
                          splits_summary, splits_results,
                          qtrl_dict['cbi']['results'], qtrl_dict['snr1']['results'], qtrl_dict['snr2']['results'])
        logger.info('Label analysis completed.')
        return size_summary, size_results, \
               multicall_summary, multicall_results, \
               chastity_summary, chastity_results, \
               SHI_summary, SHI_results, \
               mixed_summary, \
               empty_splits_results, mixed_splits_results, \
               familial_results, \
               singular_summary, \
               splits_summary, splits_results, \
               qtrl_dict['cbi']['results'], qtrl_dict['snr1']['results'], qtrl_dict['snr2']['results']

    def complete_bypass(self):
        import cPickle as pickle

        try:
            with open(self.size_results_fp, 'r') as f:
                size_results = pickle.load(f)
            with open(self.size_summary_fp, 'r') as f:
                size_summary = pickle.load(f)

            with open(self.multicall_results_fp, 'r') as f:
                multicall_results = pickle.load(f)
            with open(self.multicall_summary_fp, 'r') as f:
                multicall_summary = pickle.load(f)

            with open(self.chastity_results_fp, 'r') as f:
                chastity_results = pickle.load(f)
            with open(self.chastity_summary_fp, 'r') as f:
                chastity_summary = pickle.load(f)

            with open(self.SHI_summary_fp, 'r') as f:
                SHI_summary = pickle.load(f)
            with open(self.SHI_results_fp, 'r') as f:
                SHI_results = pickle.load(f)
            with open(self.mixed_summary_fp, 'r') as f:
                mixed_summary = pickle.load(f)

            with open(self.empty_splits_results_fp, 'r') as f:
                empty_splits_results = pickle.load(f)
            with open(self.mixed_splits_results_fp, 'r') as f:
                mixed_splits_results = pickle.load(f)

            with open(self.familial_results_fp, 'r') as f:
                familial_results = pickle.load(f)
            with open(self.singular_summary_fp, 'r') as f:
                singular_summary = pickle.load(f)

            with open(self.splits_results_fp, 'r') as f:
                splits_results = pickle.load(f)
            with open(self.splits_summary_fp, 'r') as f:
                splits_summary = pickle.load(f)

            with open(self.cbi_quartile_results_fp, 'r') as f:
                cbi_quartile_results = pickle.load(f)
            with open(self.snr1_quartile_results_fp, 'r') as f:
                snr1_quartile_results = pickle.load(f)
            with open(self.snr2_quartile_results_fp, 'r') as f:
                snr2_quartile_results = pickle.load(f)

            logger.info('Bypass successful.')
        except:
            logger.warning(traceback.format_exc())
            size_summary, size_results, \
            multicall_summary, multicall_results, \
            chastity_summary, chastity_results, \
            SHI_summary, SHI_results, \
            mixed_summary, \
            empty_splits_results, mixed_splits_results, \
            familial_results, \
            singular_summary, \
            splits_summary, splits_results, \
            cbi_quartile_results, snr1_quartile_results, snr2_quartile_results = self.run()
        return size_summary, size_results, \
               multicall_summary, multicall_results, \
               chastity_summary, chastity_results, \
               SHI_summary, SHI_results, \
               mixed_summary, \
               empty_splits_results, mixed_splits_results, \
               familial_results, \
               singular_summary, \
               splits_summary, splits_results, \
               cbi_quartile_results, snr1_quartile_results, snr2_quartile_results

def main(slide, lane, fov, start_cycle, occupancy_range, int_fp):
    inta = IntensityAnalysis(slide, lane, fov, start_cycle, occupancy_range, int_fp)
    # inta.load_data()
    la = LabelAnalysis(inta)
    la.run()

    return

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6])
