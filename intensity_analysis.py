from __future__ import division
import sys
import numpy as np
import os

from scipy.stats import gaussian_kde
import scipy.integrate
from scipy.stats import trim_mean

import logging.config
logger = logging.getLogger(__name__)
from sap_funcs import setup_logging

from sap_funcs import make_dir
import traceback
import datetime

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

class IntensityAnalysis(object):

    def __init__(self, slide, lane, fov, cycles, cal_fp, int_fp, norm_paras_fp, background_fp,
                 blocks_fp, output_dp='', bypass={},platform='V2',
                 log_dp='', log_overrides={}):
        self.lane = lane
        self.fov = fov
        self.cal_fp = cal_fp
        self.int_fp = int_fp
        self.norm_paras_fp = norm_paras_fp
        self.background_fp = background_fp
        self.blocks_fp = blocks_fp
        self.output_dp = output_dp
        self.platform = platform
        make_dir(self.output_dp)
        self.cycles = cycles # already sorted?
        self.start_cycle = self.cycles.min()+1 # used in file names and labels (1 indexing)
        self.end_cycle = self.cycles.max() + 1 #used in file names and labels (1 indexing) #self.start_cycle+self.cycle_range-1
        self.prefix = '%s_%s_%s_C%02d-C%02d' % ('-'.join(slide.split('_')), lane, fov, self.start_cycle,
                                                self.end_cycle)
        self.normCBI_fp = self.int_fp.replace('.npy', '_C%02d-C%02d_norm_CBI.npy' % (
                                                        self.start_cycle, self.end_cycle))
        self.cyndexes_fp = self.int_fp.replace('.npy', '_C%02d-C%02d_Cyndexes.npy' % (
                                                        self.start_cycle, self.end_cycle))
        self.calls_fp = self.int_fp.replace('.npy', '_C%02d-C%02d_Calls.npy' % (
                                                        self.start_cycle, self.end_cycle))
        self.thresholds_fp = self.int_fp.replace('.npy', '_C%02d-C%02d_Thresholds.npy' % (
                                                        self.start_cycle, self.end_cycle))
        self.final_thresholds_fp = self.int_fp.replace('.npy', '_C%02d-C%02d_Final_Thresholds.npy' % (
                                                        self.start_cycle, self.end_cycle))
        self.naCBI_fp = self.int_fp.replace('.npy', '_C%02d-C%02d_avg_CBI.npy' % (
                                                        self.start_cycle, self.end_cycle))
        self.labels_fp = os.path.join(self.output_dp, '%s_Labels.npy' % self.prefix)
        self.snr_fp = os.path.join(self.output_dp, '%s_SNR_Values.npy' % self.prefix)

        # output pickle paths
        self.rho_results_fp = os.path.join(self.output_dp, '%s_RHO_Results.p' % self.prefix)
        self.snr_results_fp = os.path.join(self.output_dp, '%s_SNR_Results.p' % self.prefix)
        self.thresholds_summary_fp = os.path.join(self.output_dp, '%s_Thresholds_Summary.p' % self.prefix)

        self.bypass = bypass
        self.bypass['calculate_thresholds'] = self.bypass.pop('calculate_thresholds', False)

        self.log_dp = log_dp
        self.log_overrides = log_overrides
        sub_log_fn = os.path.join(log_dp, '%s.log' % self.fov)
        sub_error_log_fn = os.path.join(log_dp, '%s_errors.log' % self.fov)
        override_dict = {'sub.log': sub_log_fn, 'sub_errors.log': sub_error_log_fn}
        override_dict.update(log_overrides)
        setup_logging(overrides=override_dict)

        logger.info('%s - Intensity Analysis initialized.' % self.fov)
        logger.info('%s - Numpy version: %s' % (self.fov, np.version.version))
        return

    def load_block_bool(self):
        block_bool = np.load(self.blocks_fp) if self.blocks_fp is not None else None
        logger.debug('%s - block bool DNB count: %s' % (self.fov, np.sum(block_bool)))
        return block_bool

    def lite_cal_obj(self):
        from fovReaders_py3.calReaderLite import CalReaderLite
        class cal_obj(object):
            def __init__(self,cal_fp,end_cycle):
                calReader = CalReaderLite(cal_fp)
                bases, score = calReader._decodeCal(cycles=end_cycle, ascii_fmt=True)
                print ("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!RHO", bases, score)
                bases[np.where(score == "!")] = "N"

                #mask = ~np.all(np.all(score == "!", axis=2), axis=1)
                self.basesDigit = {cycle+1:(bases[:,cycle]).astype('S1').view('uint8') for cycle in range(end_cycle)}
                # self.basesDigit
        return cal_obj(self.cal_fp,self.end_cycle)

    def calculate_RHO(self, label_mask=None):
        

        from intensity_index_multi import CalQCStats
        from calReader import Cal

        fin_ints = np.load(self.int_fp)
        norm_paras = np.load(self.norm_paras_fp)
        background = np.load(self.background_fp)
        if np.all(background==0) and np.all(norm_paras==0):
            logger.error('RHO Calc Failure : no background or normalization params')
            logger.error('Filling RHO With Zeros')
            zero_list = np.zeros(fin_ints.shape[2])
            rho = {'A': zero_list, 'C': zero_list, 'G': zero_list, 'T': zero_list}
        
        else:
            raw_ints = np.zeros_like(fin_ints)
            ctc_ints = np.zeros_like(fin_ints)
            if self.platform=="Lite":
                
                cal_obj = self.lite_cal_obj()
            
            else:
                from calReader import Cal
                cal_obj = Cal()
                if (self.platform.upper() == 'V40') or ('DP' in self.cal_fp) or ('cap_integ' in self.cal_fp) or (self.platform.upper() == 'V0.2'):
                    v40 = True
                else:
                    v40 = False
                cal_obj.load(self.cal_fp, V40=v40)
            
            if label_mask is None:
                label_mask = np.ones(len(fin_ints), dtype=bool)

            qc = CalQCStats(fin_ints[label_mask, :, :], raw_ints[label_mask, :, :],
                            ctc_ints[label_mask, :, :], norm_paras, cal_obj, label_mask,
                            background, cycle_range=self.cycles+1)
            try:
                logger.info('Calculating RHO')
                rho = qc.get_rho_phi()
            except Exception as e:
                tb = str(traceback.format_exc())
                logger.error('RHO Calc Failure : ' + tb)
                logger.error('Filling RHO With Zeros')
                zero_list = np.zeros(fin_ints.shape[2])
                rho = {'A': zero_list, 'C': zero_list, 'G': zero_list, 'T': zero_list}

        rho['avg'] = np.mean(np.array(list(rho.values())), axis=0)
        for base in ['A', 'C', 'G', 'T', 'avg']:
            try:
                rho['%s C%02d' % (base, self.start_cycle)] = rho[base][0]
                rho[base] = np.polyfit(range(len(self.cycles)), rho[base], 1)[1]
            except:
                rho['%s C%02d' % (base, self.start_cycle)] = rho[base][0]
                rho[base] = rho[base][0]
        return rho

    def calculate_SNR(self, normalized_data, trim=0.02):
        logger.info('%s - Calculating SNR...' % self.fov)
        signal = {'A': [], 'C': [], 'G': [], 'T': []}
        noise = {'A': [], 'C': [], 'G': [], 'T': []}

        # max_ints = np.apply_along_axis(lambda x: x == max(x), 1, normalized_data)
        argmax_ints = np.argmax(normalized_data, 1)
        max_ints = np.zeros_like(normalized_data, dtype=bool)
        for i in range(4):
            max_ints[:, i, :] = argmax_ints == i

        for i, base in enumerate('ACGT'):
            signal[base] = normalized_data[:, i, self.cyndexes][max_ints[:, i, self.cyndexes]]
            noise[base] = normalized_data[:, i, self.cyndexes][~max_ints[:, i, self.cyndexes]]

        snr = {'A': 0, 'C': 0, 'G': 0, 'T': 0, 'avg': 0}

        for base in 'ACGT':
            snr[base] = trim_mean(signal[base], trim) / np.std(noise[base])
        for base in 'ACGT':
            snr['avg'] += snr[base] / 4.0
        return snr

    def label_SNR(self, SNR_vector):
        SNR_labels = np.zeros(len(SNR_vector))
        for p in range(100):
            left = np.percentile(SNR_vector, p) if p else -float('inf')
            right = np.percentile(SNR_vector, p + 1) if p <= 99 else float('inf')
            subset = np.where(np.logical_and(SNR_vector >= left, SNR_vector < right))[0]
            SNR_labels[subset] = p + 1
        return SNR_labels

    def calculate_DNB_SNR(self, sorted_data):
        logger.info('%s - Calculating SNR at DNB level...' % self.fov)
        signal = sorted_data[:, 3, :].mean(axis=1)
        noise = sorted_data[:, :3, :].reshape(sorted_data.shape[0], 3 * len(self.cycles)).std(axis=1)
        SNR1_values = signal / (noise + 0.001)
        SNR1_labels = self.label_SNR(SNR1_values)

        signal = sorted_data[:, 3, :]
        noise = sorted_data[:, :3, :].std(axis=1)
        SNR2_values = (signal / (noise + .001)).mean(axis=1)
        SNR2_labels = self.label_SNR(SNR2_values)

        return SNR1_values, SNR1_labels, SNR2_values, SNR2_labels

    # Takes a probability distribution (1D numpy array) as input
    # Returns index of maximum density
    def calculate_mode(self, distribution, delta=0.1, n_iter=1):
        maxtab = []
        mintab = []

        x = np.arange(len(distribution))

        mn, mx = np.Inf, -np.Inf
        mnpos, mxpos = np.NaN, np.NaN

        lookformax = True

        for i in np.arange(len(distribution)):
            this = distribution[i]
            if this > mx:
                mx = this
                mxpos = x[i]
            if this < mn:
                mn = this
                mnpos = x[i]
            if lookformax:
                if this < mx - delta:
                    maxtab.append(mxpos)
                    mn = this
                    mnpos = x[i]
                    lookformax = False
            else:
                if this > mn + delta:
                    mintab.append(mnpos)
                    mx = this
                    mxpos = x[i]
                    lookformax = True

        if len(maxtab) > 1 or n_iter == 3:
            return maxtab[-1]
        else:
            return self.calculate_mode(distribution, delta / 2, n_iter + 1)

    def calculate_chastity(self, sorted_data):
        logger.info('%s - Calculating chastity...' % self.fov)
        cycle_chastities = np.apply_along_axis(lambda x: x[-1]/float(sum(x[-2:])), 1, sorted_data)
        return np.apply_along_axis(get_max_chastity, 1, cycle_chastities, self.major_count)

    def calculate_SHI(self, sorted_data):
        logger.info('%s - Calculating SHI...' % self.fov)
        cycle_SHIs = np.apply_along_axis(lambda x: x[-2]/x[-1] if x[-1] else 1.0, 1, sorted_data)
        return np.apply_along_axis(get_max_SHI, 1, cycle_SHIs, self.major_count)

    def count_multiple_calls(self, called_signals):
        logger.info('%s - Counting multiple calls...' % self.fov)
        cycle_multicalls = np.sum(called_signals, 1)
        return np.apply_along_axis(get_max_multicalls, 1, cycle_multicalls, self.major_count)

    def calculate_nonCBI(self, normalized_data):
        logger.info('%s - Calculating average nonCBI...' % self.fov)
        non_neg_norm_data = normalized_data.copy()
        non_neg_norm_data[non_neg_norm_data < 0] = 0
        CBI_proportion = np.apply_along_axis(lambda x: x / max(x), 1, non_neg_norm_data)
        argmax_ints = np.argmax(non_neg_norm_data, 1)
        max_ints = np.zeros_like(non_neg_norm_data, dtype=bool)
        for i in range(4):
            max_ints[:, i, :] = argmax_ints == i
        # max_ints = np.apply_along_axis(lambda x: x == max(x), 1, non_neg_norm_data)
        # nan out 1 (CBI) values
        CBI_proportion[max_ints] = np.nan
        average_nonCBI = np.nanmean(CBI_proportion, 2)
        average_nonCBI[np.isnan(average_nonCBI)] = 1
        return average_nonCBI * 100.
    
    def load_npy(self):
        logger.debug(f'Sixing self.int_fp: {self.int_fp}')
        data = np.load(self.int_fp, mmap_mode='r')
        if self.block_bool is not None: data = data[self.block_bool]
        num_dnbs = len(data)
        # expected shape ~570000, 4, 10
        logger.debug('%s - data shape: %s' % (self.fov, str(data.shape)))
        cycle_count = data.shape[2]
        if cycle_count > len(self.cycles):
            logger.warning('%s - Excess cycles detected! data shape: %s' % (self.fov, str(data.shape)))
            data = data[:,:,self.cycles]
            logger.warning('%s - Using slice (0 indexing): %s:%s' %
                           (self.fov, self.start_cycle-1, self.end_cycle-1))
            logger.warning('%s - Final data shape: %s' % (self.fov, str(data.shape)))
        if data.shape[2] > 1:
            major_count = int(data.shape[2] / 2.0 + 0.5) + 1
        else:
            major_count = 1
        return data, num_dnbs, major_count

    def calculate_thresholds(self, data):
        # normalize and determine thresholds
        from intensity_modeling import IntensitiesGMM

        gmm_output_dp = os.path.join(self.output_dp, 'GMM_Results')
        make_dir(gmm_output_dp)
        ig = IntensitiesGMM(data, self.prefix, self.fov, self.cycles, gmm_output_dp,
                            log_dp=self.log_dp, log_overrides=self.log_overrides)
        data, called_signals, thresholds, exclude = ig.run()
        cyndexes = np.delete(np.arange(len(self.cycles)), exclude)
        logger.debug('%s - cyndexes: %s' % (self.fov, cyndexes))
        return data, called_signals, cyndexes, \
               thresholds['empty'], thresholds['small'], thresholds['large'], thresholds['outlier']

    def save_naCBI(self, naCBI_data):
        np.save(self.naCBI_fp, naCBI_data)
        return

    def save_thresholds(self, empty_th, small_th, large_th, outlier_th):
        np.save(self.thresholds_fp,
                np.asarray([empty_th, small_th, large_th, outlier_th]))
        return

    def save_final_thresholds(self, empty_fth, small_fth, large_fth, outlier_fth):
        np.save(self.final_thresholds_fp,
                np.asarray([empty_fth, small_fth, large_fth, outlier_fth]))
        return

    def normalize(self, aCBI_data):
        logger.info('%s - Normalizing aCBI...' % self.fov)
        logger.debug('%s - aCBI_data shape: %s' % (self.fov, str(aCBI_data.shape)))
        non_outlier = np.where(aCBI_data < self.outlier_th)[0]
        outlier_perc = 100. * (self.num_dnbs - len(non_outlier)) / self.num_dnbs
        logger.debug('%s - aCBI_data[non_outlier] shape: %s' % (self.fov, str(aCBI_data[non_outlier].shape)))
        average_distribution = gaussian_kde(aCBI_data[non_outlier])
        xs = np.linspace(0, self.outlier_th, 500)
        density = average_distribution(xs)

        start_idx = np.where(xs <= self.empty_th)[0][-1] + 1
        half_area = scipy.integrate.simps(density[start_idx:], xs[start_idx:]) / 2
        i = start_idx + 1
        area = scipy.integrate.simps(density[start_idx:i], xs[start_idx:i])
        while area < half_area:
            i += 1
            area = scipy.integrate.simps(density[start_idx:i], xs[start_idx:i])
        norm_factor = xs[i]

        empty_fth = float(self.empty_th / norm_factor)
        small_fth = float(self.small_th / norm_factor)
        large_fth = float(self.large_th / norm_factor)
        outlier_fth = float(self.outlier_th / norm_factor)

        return aCBI_data / norm_factor, outlier_perc, norm_factor, empty_fth, small_fth, large_fth, outlier_fth

    def save_labels(self, label_arr):
        # fp gets passed down the line
        np.save(self.labels_fp, label_arr)
        return

    def save_snr(self, snr_arr):
        # fp gets passed down the line
        np.save(self.snr_fp, snr_arr)
        return

    def label_intensities(self, naCBI_data, empty_threshold, outlier_threshold):
        logger.info('%s - Labeling percentiles...' % self.fov)
        pl = np.zeros(len(naCBI_data))
        positive_data = naCBI_data[np.where(
            np.logical_and(naCBI_data >= empty_threshold, naCBI_data < outlier_threshold))[0]]
        for p in range(100):
            left = np.percentile(positive_data, p)
            right = np.percentile(positive_data, p + 1)
            p_subset = np.where(np.logical_and(naCBI_data >= left, naCBI_data < right))[0]
            pl[p_subset] = p + 1
        pl[np.where(naCBI_data >= outlier_threshold)[0]] = 101

        empty_data = naCBI_data[np.where(naCBI_data < empty_threshold)[0]]
        for p in range(10):
            try:
                left = np.percentile(empty_data, p * 10)
                right = np.percentile(empty_data, (p + 1) * 10)
                p_subset = np.where(np.logical_and(naCBI_data >= left, naCBI_data < right))[0]
                pl[p_subset] = -10 + p
            except:
                pass
        return pl

    def save_normalized_data(self, normalized_data):
        np.save(self.normCBI_fp, normalized_data)
        return

    def save_calls(self, called_signals):
        np.save(self.calls_fp, called_signals)
        return

    def save_cyndexes(self, cyndexes):
        np.save(self.cyndexes_fp, cyndexes)
        return

    def save_outputs(self, thresholds_summary, snr_results, rho_results):
        import pickle
        #np.save(self.thresholds_summary_fp, np.asarray(thresholds_summary))
        with open(self.thresholds_summary_fp, 'wb') as f:
            pickle.dump(thresholds_summary, f)
        #np.save(self.snr_results_fp, np.asarray(snr_results))
        with open(self.snr_results_fp, 'wb') as f:
            pickle.dump(snr_results, f)

        with open(self.rho_results_fp, 'wb') as f:
            pickle.dump(rho_results, f)
        return

    def load_normalized_data(self):
        normalized_data = np.load(self.normCBI_fp, mmap_mode='r')
        num_dnbs = len(normalized_data)
        major_count = int(normalized_data.shape[2] / 2.0 + 0.5) + 1
        return normalized_data, num_dnbs, major_count

    def run(self):
        start_time = datetime.datetime.now()
        logger.info('%s - Running Intensity Analysis...' % self.fov)

        self.block_bool = self.load_block_bool()

        try:
            if self.bypass['calculate_thresholds']:
                normalized_data, self.num_dnbs, self.major_count = self.load_normalized_data()
                self.called_signals = np.load(self.calls_fp, mmap_mode='r')
                self.cyndexes = np.load(self.cyndexes_fp)
                self.empty_th, self.small_th, self.large_th, self.outlier_th = self.load_thresholds()
            else:
                raise Exception('Not bypassing calculate_thresholds...')
                #print ("Not bypassing calculate_thresholds...")
        except:
            logger.warning(traceback.format_exc())
            if self.bypass['calculate_thresholds']:
                logger.error('%s - Unable to bypass calculate_thresholds!' % self.fov)
            # USING GMM_NORM TO NORMALIZE AND SAVE DATA
            data, self.num_dnbs, self.major_count = self.load_npy()
            #print ("data", data, data.shape, self.num_dnbs, self.major_count)
            print ("Here is in except")
            normalized_data, self.called_signals, self.cyndexes, \
            self.empty_th, self.small_th, self.large_th, self.outlier_th = \
                self.calculate_thresholds(data)
            self.save_normalized_data(normalized_data)
            self.save_calls(self.called_signals)
            self.save_cyndexes(self.cyndexes)
            self.save_thresholds(self.empty_th, self.small_th, self.large_th, self.outlier_th)
            del data

        rho = self.calculate_RHO(label_mask=None)

        multicall_data = self.count_multiple_calls(self.called_signals)

        snr = self.calculate_SNR(normalized_data)

        nonCBI = self.calculate_nonCBI(normalized_data)

        sorted_data = np.sort(normalized_data, 1)
        non_neg_sorted_data = sorted_data.copy()
        non_neg_sorted_data[non_neg_sorted_data < 0] = 0
        del normalized_data

        chastity_data = self.calculate_chastity(non_neg_sorted_data)
        SHI_data = self.calculate_SHI(non_neg_sorted_data)
        del non_neg_sorted_data

        SNR1_values, SNR1_labels, SNR2_values, SNR2_labels = self.calculate_DNB_SNR(sorted_data)
        # Average CBI per DNB for the first 10 cycles
        aCBI_data = np.mean(sorted_data[:, -1, :], 1) # use non_neg_sorted data?
        del sorted_data
        self.naCBI_data, outlier_perc, norm_factor, \
        self.empty_fth, self.small_fth, self.large_fth, self.outlier_fth = self.normalize(aCBI_data)
        del aCBI_data
        self.save_final_thresholds(self.empty_fth, self.small_fth, self.large_fth, self.outlier_fth)

        percentile_labels = self.label_intensities(self.naCBI_data, self.empty_fth, self.outlier_fth)
        self.save_naCBI(self.naCBI_data)

        # Labels Array
        # [0] percentile rankings - positive (non-outlier, non-empty) in single percent bins
        # [1] calls per DNB - # of intensities above empty threshold for majority of cycles
        # [2] chastity level - highest chastity value for majority of cycles
        # [3] children counts - number of split neighbors with lower CBI
        # [4] parent counts - number of split neighbors with higher CBI
        # [5] mixed split counts - number of split neighbors
        # [6] hidden (intensity-based) mixed splits - intensity-based matches different from sequence-based matching
        # [7] family size - number of splits around seed
        # [8] SNR labels calculated using intensities from all cycles altogether
        # [9] SNR labels calculated per cycle then averaged
        # [10] Average SHI proportion labels
        # [11-14] Average non-CBI proportions for channels ACGT

        # SNR Array
        # [0] SNR per DNB calculated using intensities from all cycles altogether
        # [1] SNR per DNB calculated per cycle then averaged

        # Mirages
        # Parent or Child ([3] > 1 or [4] > 1)
        # Not completely hidden ([5] == [6] and [5] > 0
        # Not hidden by a mixed ([7]

        self.label_arr = np.asarray((
            # 0
            percentile_labels,
            # 1
            multicall_data,
            # 2
            chastity_data,
            # 3
            np.zeros(*percentile_labels.shape),
            # 4
            np.zeros(*percentile_labels.shape),
            # 5
            np.zeros(*percentile_labels.shape)-1,
            # 6
            np.zeros(*percentile_labels.shape)-1,
            # 7
            np.zeros(*percentile_labels.shape),
            # 8
            SNR1_labels,
            # 9
            SNR2_labels,
            # 10
            SHI_data,
            # 11
            nonCBI[:,0],
            # 12
            nonCBI[:,1],
            # 13
            nonCBI[:,2],
            # 14
            nonCBI[:,3]
        )).astype(np.int8)
        self.save_labels(self.label_arr)

        self.snr_arr = np.asarray((
            # 0
            SNR1_values,
            # 1
            SNR2_values,
        )).astype(np.float16)
        self.save_snr(self.snr_arr)

        thresholds_summary = [
            ['Normalization Factor', norm_factor],
            ['Total # DNBs', self.num_dnbs],
            ['Empty Threshold', self.empty_fth],
            ['Small Threshold', self.small_fth],
            ['Large Threshold', self.large_fth],
            ['Outlier Threshold', self.outlier_fth]
        ]

        snr_results = [
            ['Avg SNR', snr['avg']],
            ['SNR A', snr['A']],
            ['SNR C', snr['C']],
            ['SNR G', snr['G']],
            ['SNR T', snr['T']]
        ]

        rho_results = [
            ['Avg RHO Intercept', rho['avg']],
            ['RHO A Intercept', rho['A']],
            ['RHO C Intercept', rho['C']],
            ['RHO G Intercept', rho['G']],
            ['RHO T Intercept', rho['T']],
            ['Avg RHO C%02d' % self.start_cycle, rho['avg C%02d' % self.start_cycle]],
            ['RHO A C%02d' % self.start_cycle, rho['A C%02d' % self.start_cycle]],
            ['RHO C C%02d' % self.start_cycle, rho['C C%02d' % self.start_cycle]],
            ['RHO G C%02d' % self.start_cycle, rho['G C%02d' % self.start_cycle]],
            ['RHO T C%02d' % self.start_cycle, rho['T C%02d' % self.start_cycle]]
        ]

        print ("((((((((((((((((((((((((((((((((((((((((((((((((((RHO", rho_results)

        self.save_outputs(thresholds_summary, snr_results, rho_results)
        time_diff = datetime.datetime.now() - start_time
        logger.info('%s - Intensity Analysis completed. (%s)' % (self.fov, time_diff))
        cbi_bypassed = False
        return rho_results, snr_results, thresholds_summary, cbi_bypassed

    def load_thresholds(self):
        thresholds = np.load(self.thresholds_fp)
        i = 0
        empty_th = thresholds[i];
        i += 1
        small_th = thresholds[i];
        i += 1
        large_th = thresholds[i];
        i += 1
        outlier_th = thresholds[i];
        return empty_th, small_th, large_th, outlier_th

    def load_final_thresholds(self):
        thresholds = np.load(self.final_thresholds_fp)
        i = 0
        empty_fth = thresholds[i];
        i += 1
        small_fth = thresholds[i];
        i += 1
        large_fth = thresholds[i];
        i += 1
        outlier_fth = thresholds[i];
        return empty_fth, small_fth, large_fth, outlier_fth

    def complete_bypass(self):
        import pickle
        # bypass run (to run mirage_analysis directly)
        try:
            self.naCBI_data = np.load(self.naCBI_fp, mmap_mode='r')
            self.called_signals = np.load(self.calls_fp, mmap_mode='r')
            self.label_arr = np.load(self.labels_fp)
            self.empty_fth, self.small_fth, self.large_fth, self.outlier_fth = self.load_final_thresholds()
            with open(self.snr_results_fp, 'rb') as p:
                snr_results = pickle.load(p)
            with open(self.thresholds_summary_fp, 'rb') as p:
                thresholds_summary = pickle.load(p)
            with open(self.rho_results_fp, 'rb') as p:
                rho_results = pickle.load(p)

            logger.info('%s - Bypass successful.' % self.fov)
            cbi_bypassed = True
        except:
            logger.warning(traceback.format_exc())
            logger.warning('%s - Could not bypass Intensity Analysis!' % self.fov)
            rho_results, snr_results, thresholds_summary, cbi_bypassed = self.run()
        return rho_results, snr_results, thresholds_summary, cbi_bypassed



def get_max_chastity(chas_array, major_count):
    for chas_int in range(10,4,-1):
        # convert int to chastity threshold value (7 -> 0.7)
        chas = chas_int/10.
        count = len(chas_array[chas_array >= chas])
        if count >= major_count: return chas_int
    return 0

def get_max_SHI(SHI_array, major_count):
    for SHI_int in range(10, -1, -1):
        SHI_perc = SHI_int/10.
        count = len(SHI_array[SHI_array >= SHI_perc])
        if count >= major_count: return SHI_int
    return -1

def get_max_multicalls(mc_array, major_count):
    for call_count in range(4, -1, -1):
        count = len(mc_array[mc_array >= call_count])
        if count >= major_count: return call_count
    return 0

def main(args):
    slide, lane, fov, cycles, cal_fp, int_fp, norm_paras_fp, background_fp, blocks_fp = args
    cycles = list(map(int, cycles.strip('[]').split(','))) #could use argparse for better parsing
    ca = IntensityAnalysis(slide, lane, fov, cycles, cal_fp, 
                           int_fp, norm_paras_fp, background_fp, blocks_fp)
    rho_results, snr_results, thresholds_summary, cbi_bypassed = ca.run()
    return rho_results, snr_results, thresholds_summary, cbi_bypassed


if __name__ == '__main__':
    main(sys.argv[1:])
    # main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6])
