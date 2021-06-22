from __future__ import division
import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from sklearn.mixture import GaussianMixture
import multiprocessing as mp
CPU_COUNT = mp.cpu_count()
np.random.seed(3)

import logging.config
logger = logging.getLogger(__name__)
from sap_funcs import setup_logging

BASE_LIST = ['A', 'C', 'G', 'T']
DYE_LIST = ['Cy5', 'TxR', 'Cy3', 'Fit']

class IntensitiesGMM(object):
    def __init__(self, data, prefix, fov, cycles, output_dp, log_dp='', log_overrides={}):
        self.data = data
        self.prefix = prefix
        self.fov = fov
        self.cycles = np.array(cycles)+1 #shift to 1 indexing for plot labeling and human readable
        self.output_dp = output_dp
        self.log_dp = log_dp
        self.log_overrides = log_overrides

        sub_log_fn = os.path.join(log_dp, '%s.log' % fov)
        sub_error_log_fn = os.path.join(log_dp, '%s_errors.log' % fov)
        override_dict = {'sub.log': sub_log_fn, 'sub_errors.log': sub_error_log_fn}
        override_dict.update(log_overrides)
        setup_logging(overrides=override_dict)
        logger.info('Initiating %s intensities GMM...' % fov)
        logger.info('Numpy version: %s' % np.version.version)

        logger.debug(self.__dict__)
        return

    def gauss_function(self, x, amp, x0, sigma):
        return amp * np.exp(-(x - x0) ** 2. / (2. * sigma ** 2.))

    def std_range(self, include, arr):
        return np.std(arr[include]), np.max(arr[include]) - np.min(arr[include])

    def add_subplot_axes(self, ax, rect=[.3, .3, .5, .5], facecolor='w'):
        fig = plt.gcf()
        box = ax.get_position()
        width = box.width
        height = box.height
        inax_position = ax.transAxes.transform(rect[0:2])
        transFigure = fig.transFigure.inverted()
        infig_position = transFigure.transform(inax_position)
        x = infig_position[0]
        y = infig_position[1]
        width *= rect[2]
        height *= rect[3]  # <= Typo was here
        subax = fig.add_axes([x, y, width, height], facecolor=facecolor)
        x_labelsize = subax.get_xticklabels()[0].get_size()
        y_labelsize = subax.get_yticklabels()[0].get_size()
        x_labelsize *= rect[2] ** 0.5
        y_labelsize *= rect[3] ** 0.5
        subax.xaxis.set_tick_params(labelsize=x_labelsize)
        subax.yaxis.set_tick_params(labelsize=y_labelsize)
        return subax

    def bin_generator(self, xmin, xmax, bin_count=100):
        temp_max = xmax - xmin
        factor = temp_max / float(bin_count)
        return [ (x * factor) + xmin for x in  range(bin_count + 1) ]

    def get_gmm_attributes(self, gmm, x):
        weights = []
        means = []
        covs = []
        separate = []

        for m, c, w in zip(gmm.means_.ravel(), gmm.covariances_.ravel(), gmm.weights_.ravel()):
            gauss = self.gauss_function(x, amp=1, x0=m, sigma=np.sqrt(c))
            means.append(m)
            covs.append(c)
            weights.append(w)
            separate.append(gauss / np.trapz(gauss, x) * w)
        covs = [y for x, y in sorted(zip(means, covs), reverse=False)]
        weights = [y for x, y in sorted(zip(means, weights), reverse=False)]
        means = sorted(means, reverse=False)
        return weights, means, covs, separate

    def dev_run(self):
        data = self.data.copy()

        gmm_parameters = ((cyndex, channel, data[:, channel, cyndex]) for
                          cyndex in range(len(data[0][0])) for channel in range(4))

        pool = mp.Pool(processes=CPU_COUNT, maxtasksperchild=1)
        logger.info('Launching fov occupancy subprocess pool...')

        gmm_outputs = pool.imap_unordered(single_channel_gmm_star, gmm_parameters)

        gmm_outputs = list(gmm_outputs)
        empty_th = [gmm_outputs[0][i*4:(i+1)*4] for i in range(10)]
        small_th = [gmm_outputs[1][i * 4:(i + 1) * 4] for i in range(10)]
        large_th = [gmm_outputs[2][i * 4:(i + 1) * 4] for i in range(10)]
        outlier_th = [gmm_outputs[3][i * 4:(i + 1) * 4] for i in range(10)]
        medians = [gmm_outputs[4][i * 4:(i + 1) * 4] for i in range(10)]
        models = [gmm_outputs[5][i * 4:(i + 1) * 4] for i in range(10)]
        #x_spaces = [gmm_outputs[6][i * 4:(i + 1) * 4] for i in range(10)]
        #gmm_results = [gmm_outputs[7][i * 4:(i + 1) * 4] for i in range(10)]

        exclude = []
        called_signals = np.zeros(data.shape)
        logger.info('Running mixture models for channels one after another.')
        for cyndex,cycle_number in enumerate(self.cycles):
            cycle_pass = True
            fig, ax = plt.subplots(2, 2, figsize=(12, 12))
            fig2, ax2 = plt.subplots(2, 2, figsize=(12, 12))

            for channel in range(4):
                logger.info('Cycle (Index) %01d Channel %s' % (cyndex, channel))
                base = BASE_LIST[channel]
                channel_pass == (empty_th[cyndex][channel] ==
                                 small_th[cyndex][channel] ==
                                 large_th[cyndex][channel] ==
                                 outlier_th[cyndex][channel] ==
                                 medians[cyndex][channel] == 0)
                if not channel_pass: cycle_pass = False
            if not cycle_pass: exclude.append(cyndex)
        """

                # Plotting
                plt.figure(1)
                gmm_y = np.exp(gmm.score_samples(gmm_x.reshape(-1, 1)))
                del gmm

                if channel == 0:
                    i = j = 0
                elif channel == 1:
                    i = 0
                    j = 1
                elif channel == 2:
                    i = 1
                    j = 0
                elif channel == 3:
                    i = j = 1

                xmin = means[0] - 3 * np.sqrt(covs[0])
                xmax = means[-1] + 3 * np.sqrt(covs[-1])

                if os.name == 'posix':
                    ax[i, j].hist(ch, bins=self.bin_generator(xmin, xmax), density=True)
                else:
                    ax[i, j].hist(ch, bins=self.bin_generator(xmin, xmax), normed=True)
                ax[i, j].plot(gmm_x, gmm_y, color='black')
                ax[i, j].set_title(base)
                for g in separate:
                    ax[i, j].plot(gmm_x, g, color='red', linestyle='--')
                [x1, x2, y1, y2] = ax[i, j].axis()
                ax[i, j].axis((xmin, xmax, y1, y2))
                # zoomed in subplot
                subax = self.add_subplot_axes(ax[i, j])
                #sub_2 = np.where(ch < 2.0)[0]
                sub_ch = ch[ch < 2.0]
                if os.name == 'posix':
                    subax.hist(sub_ch, bins=self.bin_generator(xmin, 2.0), density=True)
                else:
                    subax.hist(sub_ch, bins=self.bin_generator(xmin, 2.0), normed=True)
                subax.plot(gmm_x, gmm_y, color='black')
                for g in separate:
                    subax.plot(gmm_x, g, color='red', linestyle='--')
                sub_perc = len(sub_ch) / float(len(ch)) * 100.
                subax.axis((xmin, 2.0, y1, 1.0))
                subax.annotate('int<2.0:\n%0.02f%%' % sub_perc, xy=(0.95, 0.95), fontsize='smaller',
                               xycoords='axes fraction', horizontalalignment='right',
                               verticalalignment='top')
                sorted_weights = sorted(weights, reverse=True)
                noise_str = '%0.2f%% Noise' % (float(sorted_weights[0]) * 100.)
                signal_percents = ', '.join(map(lambda x: '%0.2f%%' % (float(x) * 100.), sorted_weights[1:]))
                signal_str = '%s Signal' % signal_percents
                threshold_str = ''
                if channel_pass:
                    empty_th[cyndex][channel] = np.sqrt(covs[0]) * 3
                    called_signals[:, channel, cyndex] = data[:, channel, cyndex] > empty_th[cyndex][channel]
                    threshold_str += '%0.4f Threshold' % empty_th[cyndex][channel]
                    # pre-normalized threshold line
                    subax.axvline(x=empty_th[cyndex][channel], color='green', linestyle='--')
                    ax[i, j].axvline(x=empty_th[cyndex][channel], color='green', linestyle='--')

                    idx = np.where(np.logical_and(ch > min(means) + empty_th[cyndex][channel], np.isfinite(ch)))[0]
                    if len(idx) > 0.05 * len(ch):
                        positive = ch[idx]
                    else:
                        positive = ch

                    median = np.median(positive)
                    if abs(median - min(means)) < abs(median - max(means)):
                        # outlier protection?
                        median = max(means)
                    logger.info('Median: %s' % median)

                    # Reject Outliers and Find Small/Large Threshold (IQR Method)
                    q1 = np.percentile(positive, 25)
                    q3 = np.percentile(positive, 75)
                    outlier_th[cyndex][channel] = q3 + 3 * (q3 - q1)
                    small_th[cyndex][channel] = q1
                    large_th[cyndex][channel] = q3

                    # Continue with normalization
                    if median != 0:
                        outlier_th[cyndex][channel] /= median
                        small_th[cyndex][channel] /= median
                        empty_th[cyndex][channel] /= median
                        large_th[cyndex][channel] /= median

                        ch = (ch - min(means)) / median
                        data[:, channel, cyndex] = ch

                    plt.figure(2)
                    xmin /= median
                    xmax /= median

                    if os.name == 'posix':
                        ax2[i, j].hist(ch, bins=self.bin_generator(xmin, xmax), density=True)
                    else:
                        ax2[i, j].hist(ch, bins=self.bin_generator(xmin, xmax), normed=True)
                    ax2[i, j].set_title(base)
                    [x1, x2, y1, y2] = ax2[i, j].axis()
                    ax2[i, j].axis((xmin, xmax, y1, y2))
                    # zoomed in subplot
                    subax2 = self.add_subplot_axes(ax2[i, j])
                    #sub_2 = np.where(ch < 2.0)[0]
                    sub_ch = ch[ch < 2.0]
                    if os.name == 'posix':
                        subax2.hist(sub_ch, bins=self.bin_generator(xmin, 2.0), density=True)
                    else:
                        subax2.hist(sub_ch, bins=self.bin_generator(xmin, 2.0), normed=True)
                    subax2.axvline(x=empty_th[cyndex][channel], color='green', linestyle='--')
                    ax2[i, j].axvline(x=empty_th[cyndex][channel], color='green', linestyle='--')
                    norm_factor_str = '%0.4f Normalization Factor' % (1.0/median)
                    norm_thresh_str = '%0.4f Threshold' % empty_th[cyndex][channel]
                    ax2[i, j].annotate('%s\n%s' % (norm_factor_str, norm_thresh_str),
                                      xy=(0.95, 0.95), xycoords='axes fraction', horizontalalignment='right',
                                      verticalalignment='top')
                plt.setp(subax, file=open(os.devnull, 'w'))
                ax[i, j].annotate('%s\n%s\n%s' % (noise_str, signal_str, threshold_str),
                                  xy=(0.95, 0.95), xycoords='axes fraction', horizontalalignment='right',
                                  verticalalignment='top')

                norm_factor = 1.0 if not channel_pass else 1.0/median
                for vl in [means, covs, weights]:
                    while len(vl) < 3:
                        vl.append('')
                gmm_results.append([cycle_number, base] + weights + means + covs + [norm_factor, 'final'])

            plt.figure(1)
            plt.suptitle('%s C%02d finInts GMM' % (self.prefix, cycle_number))
            if cycle_pass:
                png_path = os.path.join(self.output_dp, '%s_C%02d_GMM_Separation.png' % (self.prefix, cycle_number))
            else:
                png_path = os.path.join(self.output_dp, '%s_C%02d_GMM_Failure.png' % (self.prefix, cycle_number))
                exclude.append(cyndex)
            try:
                plt.savefig(png_path)
            except:
                plt.savefig('\\\\?\\' + png_path)
            plt.gcf().clear()
            plt.close()

            plt.figure(2)
            plt.suptitle('%s C%02d Normalized Histogram' % (self.prefix, cycle_number))
            png_path = os.path.join(self.output_dp, '%s_C%02d_Normalized_Intensities.png' % (self.prefix, cycle_number))
            try:
                plt.savefig(png_path)
            except:
                plt.savefig('\\\\?\\' + png_path)
            plt.gcf().clear()
            plt.close()
        """

        logger.info('Modeling complete.')
        include = [val for val in range(len(data[0][0])) if val not in exclude]
        assert include, 'No cycles remaining!'

        empty = np.sum(empty_th[include]) / (len(include) * 4)
        small = np.sum(small_th[include]) / (len(include) * 4)
        large = np.sum(large_th[include]) / (len(include) * 4)
        outlier = np.sum(outlier_th[include]) / (len(include) * 4)

        th_std, th_range = self.std_range(include, empty_th)
        small_std, small_range = self.std_range(include, small_th)
        large_std, large_range = self.std_range(include, large_th)
        outlier_std, outlier_range = self.std_range(include, outlier_th)

        logger.debug('Average empty threshold: %s' % empty)
        logger.debug('Empty_th stdev: %s' % th_std)
        logger.debug('Empty_th range: %s\n' % th_range)
        logger.debug('Average small threshold: %s' % small)
        logger.debug('Small_th stdev: %s' % small_std)
        logger.debug('Small_th range: %s\n' % small_range)
        logger.debug('Average large threshold: %s' % large)
        logger.debug('Large_th stdev: %s' % large_std)
        logger.debug('Large_th range: %s\n' % large_range)
        logger.debug('Average outlier threshold: %s' % outlier)
        logger.debug('Outlier_th stdev: %s' % outlier_std)
        logger.debug('Outlier_th range: %s\n' % outlier_range)

        gmm_fp = os.path.join(self.output_dp, '%s_GMM_Results.csv' % self.fov)
        gmm_header = ['Cycle', 'Channel',
                      'Weight1', 'Weight2', 'Weight3',
                      'Mean1', 'Mean2', 'Mean3',
                      'Cov1', 'Cov2', 'Cov3', 'NormFactor', 'Note']
        with open(gmm_fp, 'w') as gmm_f:
            gmm_f.write(','.join(gmm_header) + '\n')
            for gmm_result in gmm_results:
                gmm_f.write(','.join(map(str, gmm_result)) + '\n')

        if len(exclude) > 0:
            logger.info('Excluded cyndex(es): %s' % exclude)
            flag_fp = os.path.join(self.output_dp, '%s_%s_Excluded_Cycles.csv' % (self.fov, len(exclude)))
            with open(flag_fp, 'w') as flag_f:
                flag_f.write(','.join(map(str, exclude)))

        thresholds = {'empty': empty, 'small': small, 'large': large, 'outlier': outlier}
        statistics = {'th_std': th_std, 'th_range': th_range, 'small_std': small_std, 'small_range': small_range,
                      'large_std': large_std, 'large_range': large_range, 'outlier_std': outlier_std,
                      'outlier_range': outlier_range}
        logger.info('Completed.')
        return data, called_signals.astype(np.int8), thresholds, exclude

    def run(self):
        data = self.data.copy()
        three_sigma = np.asarray([[0 for _ in range(4)] for _ in range(len(data[0][0]))], dtype=np.float64)
        small_th = three_sigma.copy()
        large_th = three_sigma.copy()
        outlier_th = three_sigma.copy()

        exclude = []
        gmm_results = []
        called_signals = np.zeros(data.shape)
        logger.info('Running mixture models for channels one after another.')
        for cyndex,cycle_number in enumerate(self.cycles):
            cycle_pass = True
            fig, ax = plt.subplots(2, 2, figsize=(12, 12))
            fig2, ax2 = plt.subplots(2, 2, figsize=(12, 12))

            for channel in range(4):
                logger.info('Cycle (Index) %01d Channel %s' % (cyndex, channel))
                ch = data[:, channel, cyndex]
                base = BASE_LIST[channel]
                channel_pass = not (np.max(ch) <= 0 or np.max(ch) == np.min(ch))
                if not channel_pass:
                    logger.warning('Bad channel data!')

                q1 = np.percentile(ch, 75 + 25 * 0.25)
                q3 = np.percentile(ch, 75 + 25 * 0.75)
                temp_outlier_th = q3 + 3 * (q3 - q1)
                non_outliers = ch[np.where(ch < temp_outlier_th)[0]]

                n_comp = 3
                gmm = GaussianMixture(n_components=n_comp, covariance_type='full')
                gmm = gmm.fit(X=non_outliers.reshape(-1,1))
                gmm_x = np.linspace(min(non_outliers), max(non_outliers), 1000)

                weights, means, covs, separate = self.get_gmm_attributes(gmm, gmm_x)
                gmm_results.append([cycle_number, base] + weights + means + covs)

                if any([w / sum(weights[1:]) < 0.2 for w in weights[1:]]):
                    n_comp = 2
                    gmm = GaussianMixture(n_components=n_comp, covariance_type='full', tol=0.0001)
                    gmm = gmm.fit(X=non_outliers.reshape(-1,1))

                    weights, means, covs, separate = self.get_gmm_attributes(gmm, gmm_x)

                logger.debug('GMM n_components=%s' % n_comp)
                logger.debug('means: %s, covs: %s, weights: %s' % (means, covs, weights))

                if max(weights) > 0.95 or abs(min(means)) > 0.1:
                    logger.warning('weight: %s' % max(weights))
                    logger.warning('cov: %s' % max(covs))
                    logger.warning('means: %s' % abs(min(means)))
                    # raise bad_data
                    channel_pass = False
                    cycle_pass = False

                # Plotting
                plt.figure(1)
                gmm_y = np.exp(gmm.score_samples(gmm_x.reshape(-1, 1)))
                del gmm

                if channel == 0:
                    i = j = 0
                elif channel == 1:
                    i = 0
                    j = 1
                elif channel == 2:
                    i = 1
                    j = 0
                elif channel == 3:
                    i = j = 1

                xmin = means[0] - 3 * np.sqrt(covs[0])
                xmax = means[-1] + 3 * np.sqrt(covs[-1])

                if os.name == 'posix':
                    ax[i, j].hist(ch, bins=self.bin_generator(xmin, xmax), density=True)
                else:
                    ax[i, j].hist(ch, bins=self.bin_generator(xmin, xmax), normed=True)
                ax[i, j].plot(gmm_x, gmm_y, color='black')
                ax[i, j].set_title(base)
                for g in separate:
                    ax[i, j].plot(gmm_x, g, color='red', linestyle='--')
                [x1, x2, y1, y2] = ax[i, j].axis()
                ax[i, j].axis((xmin, xmax, y1, y2))
                # zoomed in subplot
                subax = self.add_subplot_axes(ax[i, j])
                #sub_2 = np.where(ch < 2.0)[0]
                sub_ch = ch[ch < 2.0]
                if os.name == 'posix':
                    subax.hist(sub_ch, bins=self.bin_generator(xmin, 2.0), density=True)
                else:
                    subax.hist(sub_ch, bins=self.bin_generator(xmin, 2.0), normed=True)
                subax.plot(gmm_x, gmm_y, color='black')
                for g in separate:
                    subax.plot(gmm_x, g, color='red', linestyle='--')
                sub_perc = len(sub_ch) / float(len(ch)) * 100.
                subax.axis((xmin, 2.0, y1, 1.0))
                subax.annotate('int<2.0:\n%0.02f%%' % sub_perc, xy=(0.95, 0.95), fontsize='smaller',
                               xycoords='axes fraction', horizontalalignment='right',
                               verticalalignment='top')
                sorted_weights = sorted(weights, reverse=True)
                noise_str = '%0.2f%% Noise' % (float(sorted_weights[0]) * 100.)
                signal_percents = ', '.join(map(lambda x: '%0.2f%%' % (float(x) * 100.), sorted_weights[1:]))
                signal_str = '%s Signal' % signal_percents
                threshold_str = ''
                if channel_pass:
                    three_sigma[cyndex][channel] = np.sqrt(covs[0]) * 3
                    called_signals[:, channel, cyndex] = data[:, channel, cyndex] > three_sigma[cyndex][channel]
                    threshold_str += '%0.4f Threshold' % three_sigma[cyndex][channel]
                    # pre-normalized threshold line
                    subax.axvline(x=three_sigma[cyndex][channel], color='green', linestyle='--')
                    ax[i, j].axvline(x=three_sigma[cyndex][channel], color='green', linestyle='--')

                    idx = np.where(np.logical_and(ch > min(means) + three_sigma[cyndex][channel], np.isfinite(ch)))[0]
                    if len(idx) > 0.05 * len(ch):
                        positive = ch[idx]
                    else:
                        positive = ch

                    median = np.median(positive)
                    if abs(median - min(means)) < abs(median - max(means)):
                        # outlier protection?
                        median = max(means)
                    logger.info('Median: %s' % median)

                    # Reject Outliers and Find Small/Large Threshold (IQR Method)
                    q1 = np.percentile(positive, 25)
                    q3 = np.percentile(positive, 75)
                    outlier_th[cyndex][channel] = q3 + 3 * (q3 - q1)
                    small_th[cyndex][channel] = q1
                    large_th[cyndex][channel] = q3

                    # Continue with normalization
                    if median != 0:
                        outlier_th[cyndex][channel] /= median
                        small_th[cyndex][channel] /= median
                        three_sigma[cyndex][channel] /= median
                        large_th[cyndex][channel] /= median

                        ch = (ch - min(means)) / median
                        data[:, channel, cyndex] = ch

                    plt.figure(2)
                    xmin /= median
                    xmax /= median

                    if os.name == 'posix':
                        ax2[i, j].hist(ch, bins=self.bin_generator(xmin, xmax), density=True)
                    else:
                        ax2[i, j].hist(ch, bins=self.bin_generator(xmin, xmax), normed=True)
                    ax2[i, j].set_title(base)
                    [x1, x2, y1, y2] = ax2[i, j].axis()
                    ax2[i, j].axis((xmin, xmax, y1, y2))
                    # zoomed in subplot
                    subax2 = self.add_subplot_axes(ax2[i, j])
                    #sub_2 = np.where(ch < 2.0)[0]
                    sub_ch = ch[ch < 2.0]
                    if os.name == 'posix':
                        subax2.hist(sub_ch, bins=self.bin_generator(xmin, 2.0), density=True)
                    else:
                        subax2.hist(sub_ch, bins=self.bin_generator(xmin, 2.0), normed=True)
                    subax2.axvline(x=three_sigma[cyndex][channel], color='green', linestyle='--')
                    ax2[i, j].axvline(x=three_sigma[cyndex][channel], color='green', linestyle='--')
                    norm_factor_str = '%0.4f Normalization Factor' % (1.0/median)
                    norm_thresh_str = '%0.4f Threshold' % three_sigma[cyndex][channel]
                    ax2[i, j].annotate('%s\n%s' % (norm_factor_str, norm_thresh_str),
                                      xy=(0.95, 0.95), xycoords='axes fraction', horizontalalignment='right',
                                      verticalalignment='top')
                # plt.setp(subax, file=open(os.devnull, 'w')) #is this necessary?
                ax[i, j].annotate('%s\n%s\n%s' % (noise_str, signal_str, threshold_str),
                                  xy=(0.95, 0.95), xycoords='axes fraction', horizontalalignment='right',
                                  verticalalignment='top')

                norm_factor = 1.0 if not channel_pass else 1.0/median
                for vl in [means, covs, weights]:
                    while len(vl) < 3:
                        vl.append('')
                gmm_results.append([cycle_number, base] + weights + means + covs + [norm_factor, 'final'])

            plt.figure(1)
            plt.suptitle('%s C%02d finInts GMM' % (self.prefix, cycle_number))
            if cycle_pass:
                png_path = os.path.join(self.output_dp, '%s_C%02d_GMM_Separation.png' % (self.prefix, cycle_number))
            else:
                png_path = os.path.join(self.output_dp, '%s_C%02d_GMM_Failure.png' % (self.prefix, cycle_number))
                exclude.append(cyndex)
            try:
                plt.savefig(png_path)
            except:
                plt.savefig('\\\\?\\' + png_path)
            plt.gcf().clear()
            plt.close()

            plt.figure(2)
            plt.suptitle('%s C%02d Normalized Histogram' % (self.prefix, cycle_number))
            png_path = os.path.join(self.output_dp, '%s_C%02d_Normalized_Intensities.png' % (self.prefix, cycle_number))
            try:
                plt.savefig(png_path)
            except:
                plt.savefig('\\\\?\\' + png_path)
            plt.gcf().clear()
            plt.close()

        logger.info('Modeling complete.')
        include = [val for val in range(len(data[0][0])) if val not in exclude]
        assert include, 'No cycles remaining!'

        empty = np.sum(three_sigma[include]) / (len(include) * 4)
        small = np.sum(small_th[include]) / (len(include) * 4)
        large = np.sum(large_th[include]) / (len(include) * 4)
        outlier = np.sum(outlier_th[include]) / (len(include) * 4)

        th_std, th_range = self.std_range(include, three_sigma)
        small_std, small_range = self.std_range(include, small_th)
        large_std, large_range = self.std_range(include, large_th)
        outlier_std, outlier_range = self.std_range(include, outlier_th)

        logger.debug('Average empty threshold: %s' % empty)
        logger.debug('Empty_th stdev: %s' % th_std)
        logger.debug('Empty_th range: %s\n' % th_range)
        logger.debug('Average small threshold: %s' % small)
        logger.debug('Small_th stdev: %s' % small_std)
        logger.debug('Small_th range: %s\n' % small_range)
        logger.debug('Average large threshold: %s' % large)
        logger.debug('Large_th stdev: %s' % large_std)
        logger.debug('Large_th range: %s\n' % large_range)
        logger.debug('Average outlier threshold: %s' % outlier)
        logger.debug('Outlier_th stdev: %s' % outlier_std)
        logger.debug('Outlier_th range: %s\n' % outlier_range)

        gmm_fp = os.path.join(self.output_dp, '%s_GMM_Results.csv' % self.fov)
        gmm_header = ['Cycle', 'Channel',
                      'Weight1', 'Weight2', 'Weight3',
                      'Mean1', 'Mean2', 'Mean3',
                      'Cov1', 'Cov2', 'Cov3', 'NormFactor', 'Note']
        with open(gmm_fp, 'w') as gmm_f:
            gmm_f.write(','.join(gmm_header) + '\n')
            for gmm_result in gmm_results:
                gmm_f.write(','.join(map(str, gmm_result)) + '\n')

        if len(exclude) > 0:
            logger.info('Excluded cyndex(es): %s' % exclude)
            flag_fp = os.path.join(self.output_dp, '%s_%s_Excluded_Cycles.csv' % (self.fov, len(exclude)))
            with open(flag_fp, 'w') as flag_f:
                flag_f.write(','.join(map(str, exclude)))

        thresholds = {'empty': empty, 'small': small, 'large': large, 'outlier': outlier}
        statistics = {'th_std': th_std, 'th_range': th_range, 'small_std': small_std, 'small_range': small_range,
                      'large_std': large_std, 'large_range': large_range, 'outlier_std': outlier_std,
                      'outlier_range': outlier_range}
        logger.info('Completed.')
        return data, called_signals.astype(np.int8), thresholds, exclude

def bin_generator(xmin, xmax, bin_count=100):
    temp_max = xmax - xmin
    factor = temp_max / float(bin_count)
    return map(lambda x: (x * factor) + xmin, range(bin_count + 1))

def get_gmm_attributes(gmm, x):
    def gauss_function(x, amp, x0, sigma):
        return amp * np.exp(-(x - x0) ** 2. / (2. * sigma ** 2.))

    weights = []
    means = []
    covs = []
    separate = []

    for m, c, w in zip(gmm.means_.ravel(), gmm.covariances_.ravel(), gmm.weights_.ravel()):
        gauss = gauss_function(x, amp=1, x0=m, sigma=np.sqrt(c))
        means.append(m)
        covs.append(c)
        weights.append(w)
        separate.append(gauss / np.trapz(gauss, x) * w)
    covs = [y for x, y in sorted(zip(means, covs), reverse=False)]
    weights = [y for x, y in sorted(zip(means, weights), reverse=False)]
    means = sorted(means, reverse=False)
    return weights, means, covs, separate

def single_channel_gmm(cyndex, channel, channel_intensities, ax, ax2):
    #np.random.seed(3)

    channel_pass = not (np.max(channel_intensities) <= 0 or np.max(channel_intensities) == np.min(channel_intensities))
    if not channel_pass:
        logger.warning('C%02d ch%s - Bad channel data!' % (cyndex, channel))

    q1 = np.percentile(channel_intensities, 75 + 25 * 0.25)
    q3 = np.percentile(channel_intensities, 75 + 25 * 0.75)
    temp_outlier_th = q3 + 3 * (q3 - q1)
    non_outliers = channel_intensities[np.where(channel_intensities < temp_outlier_th)[0]]

    n_comp = 3
    gmm = GaussianMixture(n_components=n_comp, covariance_type='full')
    gmm = gmm.fit(X=non_outliers.reshape(-1,1))
    gmm_x = np.linspace(min(non_outliers), max(non_outliers), 1000)

    weights, means, covs, separate = get_gmm_attributes(gmm, gmm_x)
    gmm_result = [weights, means, covs]

    if any([w / sum(weights[1:]) < 0.2 for w in weights[1:]]):
        n_comp = 2
        gmm = GaussianMixture(n_components=n_comp, covariance_type='full', tol=0.0001)
        gmm = gmm.fit(X=non_outliers.reshape(-1,1))

        weights, means, covs, separate = get_gmm_attributes(gmm, gmm_x)

    logger.debug('C%02d ch%s - GMM n_components=%s' % (cyndex, channel,n_comp))
    logger.debug('C%02d ch%s - means: %s, covs: %s, weights: %s' % (cyndex, channel, means, covs, weights))

    if max(weights) > 0.95 or abs(min(means)) > 0.1:
        logger.warning('C%02d ch%s - weight: %s' % ( cyndex, channel,max(weights)))
        logger.warning('C%02d ch%s - cov: %s' % (cyndex, channel,max(covs)))
        logger.warning('C%02d ch%s - means: %s' % (cyndex, channel,abs(min(means))))
        # raise bad_data
        channel_pass = False

    if channel_pass:
        empty_th = np.sqrt(covs[0]) * 3

        idx = np.where(np.logical_and(channel_intensities > min(means) + empty_th, np.isfinite(channel_intensities)))[0]
        if len(idx) > (0.05 * len(channel_intensities)):
            positive = channel_intensities[idx]
        else:
            logger.warning('C%02d ch%s - "Positive" data less than 5% of total!' % (cyndex, channel))
            positive = channel_intensities

        median = np.median(positive)
        if abs(median - min(means)) < abs(median - max(means)):
            # outlier protection?
            median = max(means)
        logger.info('Median: %s' % median)

        # Reject Outliers and Find Small/Large Threshold (IQR Method)
        q1 = np.percentile(positive, 25)
        q3 = np.percentile(positive, 75)
        small_th = q1
        large_th = q3
        outlier_th = q3 + 3 * (q3 - q1)

        # Continue with normalization
        if median != 0:
            empty_th /= median
            small_th /= median
            large_th /= median
            outlier_th /= median
    else:
        empty_th = small_th = large_th = outlier_th = median = 0
    return empty_th, small_th, outlier_th, large_th, median, gmm

def single_channel_gmm_star(args):
    single_channel_gmm(*args)

def main(data, prefix, fov, output_dp):
    ig = IntensitiesGMM(data, prefix, fov, output_dp)
    return ig.run()

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
