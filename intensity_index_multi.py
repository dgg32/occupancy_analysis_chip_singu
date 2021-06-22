# -*- coding: utf-8 -*-
"""
Created on Fri Nove 3 1:34 2017

@author: ajorjorian
"""


# extract intensities from bin files for all cycles
import copy
import numpy as np
import extract_intensities_np
import calReader
import scipy as sp
import os
import traceback
import Signal_phasing_from_npy as SignalPhasing
import sys
import pickle
import multiprocessing as mp
import scipy.stats as stats
import glob
from zap_funcs import full_traceback
from zap_funcs import setup_logging
from int2npy import get_posiIndex_Npy
import logging.config
import crosstalk_correction_sklearn_ver_0_18_1 as cc
canonical_subset_order = ['Empty', 'Mixed', 'Poor', 'Low', 'mLow', 'mHigh', 'High', 'Outlier']
insert_subset_order = ['0bp', '0_250bp', '250_350bp', '350_450bp', '450_550bp', '550_inf']


def zmf(val):
    fit = np.ones_like(val)
    case1 = np.where((val > 0) & (val <= 35))
    case2 = np.where((val <= 70) & (val > 35))
    case3 = np.where((val > 70) | (val == 0.0))
    fit[case1] = 1 - 2 * np.power(((val[case1]) / 70), 2)
    fit[case2] = 2 * np.power(((val[case2] - 70) / 70), 2)
    fit[case3] = 0.0
    return fit


class CalQCStats(object):

    def __init__(self, fin_ints, raw_ints, ctc_ints, norm_paras, cal_obj, subset_bool, background, cycle_range=[]):
        self.fin = fin_ints[:, :, :]
        # only used when using Cal dicts because int data should be of cycle_range already
        self.cycle_range = range(1, fin_ints.shape[2] + 1) if len(cycle_range) == 0 else cycle_range
        # self.fin[np.where(self.fin == self.fin.max())] = 0.0
        self.cal = cal_obj
        self.raw = raw_ints
        self.bkg = background
        self.ctc = ctc_ints
        self.subset_bool = subset_bool
        self.fin_max_args = np.argmax(self.fin, axis=1)
        # self.raw = 0
        # self.orig = float()
        self.norm_paras = norm_paras
        # self.norm = float()
        self.origrho = [[[], [], [], []]]*int(self.fin.shape[2])
        self.normphi = [[[], [], [], []]] * int(self.fin.shape[2])
        self.normrho = [[[], [], [], []]]*int(self.fin.shape[2])
        self.rho_mean = {'A': [], 'C': [], 'G': [], 'T': []}

    def get_cycQ30_acgtn(self):
        base_count = self.subset_bool.sum()
        cycQ30 = []
        acgtn = {'A': [], 'C': [], 'G': [], 'T': [], 'N': []}
        # for cycle_index in range(1, self.fin.shape[2]+1):
        for cycle_index in self.cycle_range:
            cycQ30.append((self.cal.qual[cycle_index][self.subset_bool] >= 30).sum()/float(base_count))
            for i, base in enumerate('ACGTN'):
                acgtn[base].append((self.cal.bases[cycle_index][self.subset_bool] == base).sum())
        return cycQ30, acgtn

    def get_signal(self):
        signals = {'A': [], 'C': [], 'G': [], 'T': []}
        signal_vect = []
        for i in range(self.raw.shape[2]):
            for j, base in enumerate('ACGT'):
                current = self.raw[:, j, i]
                sig = current[
                    np.where((current > np.nanpercentile(current, 85)) &
                             (current <= np.nanpercentile(current, 99)))].mean()
                signal_vect.append(sig)
                signals[base].append(sig)
        return signals, signal_vect

    def get_background(self):
        background = {'A': [], 'C': [], 'G': [], 'T': []}
        for i, base, in enumerate('ACGT'):
            background[base] = self.bkg[i, :].tolist()
        return background

    def get_rho_phi(self):
        for i, cycle in enumerate(self.cycle_range):  # range(self.fin.shape[2]):
            cycle_filter = (self.cal.basesDigit[cycle] != 78)[self.subset_bool]
            for j, base in enumerate('ACGT'):
                cycle_base_filter = cycle_filter.copy()
                off_cycle_filter = cycle_filter.copy()
                cycle_base_filter[self.fin_max_args[:, i] != j] = False
                self.normphi[i][j] = np.zeros(cycle_base_filter.sum())
                norm_rho = (self.fin[cycle_base_filter, j, i]*(self.norm_paras[8 + j * 2 + 1, i] -
                                                               self.norm_paras[8 + j * 2, i])) + \
                                                               self.norm_paras[8 + j * 2, i]
                origrho = (norm_rho*(self.norm_paras[j * 2 + 1, i] -
                                     self.norm_paras[j * 2, i])) + self.norm_paras[j * 2, i]

                for k, base2 in enumerate('ACGT'):
                    if k != j:
                        self.normphi[i][j]+=((self.fin[cycle_base_filter, k, i]*(self.norm_paras[8 + k * 2 + 1, i] -
                                                               self.norm_paras[8 + k * 2, i])) +
                                                               self.norm_paras[8 + k * 2, i])**2
                self.normphi[i][j][np.where(np.abs(norm_rho)>0)] = self.normphi[i][j][np.where(np.abs(norm_rho)>0)]/norm_rho[np.where(np.abs(norm_rho)>0)]
                self.normphi[i][j][np.where(np.abs(norm_rho)==0)] = 0.0
                self.normphi[i][j] = np.nan_to_num(sp.arctan(self.normphi[i][j])*(180.0/np.pi))
                self.normrho[i][j] = norm_rho
                self.origrho[i][j] = np.nan_to_num(origrho)
                self.rho_mean[base].append(np.nan_to_num(np.nanmean(origrho)))
        return self.rho_mean

    def bic(self):
        eps = np.float64(0.0000001)
        norm = stats.norm
        ints = self.fin.copy()
        ints.sort(axis=1)
        y1 = ints[:, 3, :]
        y2 = ints[:, 2, :]
        y_max12 = abs(y1/(y2+eps))
        y_max12[np.where(y_max12 <= 1.0)] = 1.0
        y_max12pct = np.nanpercentile(y_max12, 10, axis=0)
        y_max12pct[np.where(y_max12pct > 5.0)] = 5.0
        field_ratio = (-0.1170642) + 2.1515030*y_max12pct + (-0.2127391)*(y_max12pct*y_max12pct)
        y12_log = np.log10(y_max12)
        y_max1_t = 0.85 * norm.cdf((y1 - 0.16) / 0.11) + 0.15 * norm.cdf((y1 - 0.55) / 0.30)
        y_max12_t = 0.97 * norm.cdf((y12_log + 0.01) / 0.40) + 0.03 * norm.cdf((y12_log - 0.95) / 0.70)
        link = (-6.07433297873612) + 3.68687665581271*y_max1_t + 6.61663992294833*y_max12_t + 0.383204046138289 * \
            field_ratio
        score = 1.0/(1.0+np.exp(-link))
        nonp = (1.0-score)/3.0
        ic = 1.0 + score*(np.log2(score)/np.log2(4)) + 3*nonp*(np.log2(nonp)/np.log2(4))
        bic_val = np.nansum(ic, axis=0)/len(ic)
        return bic_val

    def cbi_curve(self):
        cbi = self.fin.max(axis=1).mean(axis=0)
        # self.fin = 0
        return cbi

    def phi_call_vect(self):
        ints = self.fin.copy()
        phi95_array = np.zeros(ints.shape[2])
        for y in range(ints.shape[2]):
            phi95 = np.array([0, 0, 0, 0])
            for i, call_base in enumerate('ACGT'):
                try:
                    phi95[i] = np.nanpercentile(self.normphi[y][i][np.where(self.origrho[y][i] >
                                                                            self.rho_mean[call_base][y])], 95)
                except:
                    phi95[i] = 0
            phi95_array[y] = np.nanmax(phi95)
        fit_score = zmf(phi95_array)
        return fit_score
        # norm_rho = self.ctc.copy()
        # for i in range(4):
        #     orig_rho[:, i, :] = orig_rho[:, i, :]*(self.norm_paras[i][1]-self.norm_paras[i][0])+self.norm_paras[i][0]
        # for i in range(4):
        #     norm_rho[:, i, :] = (norm_rho[:, i, :] -
        #                          self.norm_paras[i][0])/(self.norm_paras[i][1]-self.norm_paras[i][0])
        # # norm_rho =  * self.norm
        # # orig_rho = ints * self.orig
        # self.rho = orig_rho.mean(axis=1)
        # orig_rho = self.ctc.copy()
        # self.rho = orig_rho.mean(axis=1)
        # ints_swapped = ints.swapaxes(1, 2)
        # raw_vector = ints_swapped.reshape(ints_swapped.shape[0]*ints_swapped.shape[1], 4)
        # max_args = self.fin_max_args
        # max_vector = np.ravel(max_args)
        # ints = norm_rho.swapaxes(1, 2)
        # ints_vector = ints_swapped.reshape(ints.shape[0]*ints.shape[1], 4)
        # int_call = ints_vector[np.arange(len(raw_vector)), max_vector[:]].reshape(len(max_vector)/ints.shape[1],
        #                                                                           ints.shape[1])
        # ints_vector[np.arange(len(raw_vector)), max_vector[:]] = np.nan
        # ints = ints_vector.reshape(ints.shape[0], ints.shape[1], ints.shape[2])
        # ints = ints.swapaxes(1, 2)
        # int_rms = np.sqrt(np.nansum((ints**2), axis=1))
        # int_call[np.where(int_call <= 0.0)] = 1.0
        # int_rms[np.where(int_call <= 0.0)] = 0.0
        # int_phi = int_rms/int_call
        # phi = np.arctan(int_phi)
        # raw_call = raw_vector[np.arange(len(raw_vector)), max_vector[:]].reshape(len(max_vector)/phi.shape[1],
        #                                                                          phi.shape[1])
        # phi_95_pos = []
        # for i in range(4):
        #     phi_temp = phi.copy()
        #     raw_temp = np.full_like(raw_call, np.nan)
        #     raw_temp[np.where(max_args == i)] = raw_call[np.where(max_args == i)]
        #     mu = np.ones_like(raw_call[0, :])
        #     for j in range(len(mu)):
        #         raw_vect = raw_temp[:, j]
        #         mu[j] = stats.trim_mean(raw_vect[~np.isnan(raw_vect)], 0.01)
        #     phi_pos = np.where((raw_call <= mu) | (max_args != i))
        #     phi_temp[phi_pos[0], phi_pos[1]] = np.nan
        #     phi_95_pos.append(np.where((phi >= np.nanpercentile(phi_temp, 95, axis=0)) & (max_args == i)))
        # for i in range(4):
        #     phi[phi_95_pos[i][0], phi_95_pos[i][1]] = np.nan
        # phi95 = np.nanmax(phi, axis=0)
        # phi95 = phi95*(180/np.pi)
        # phi95[np.where(phi95 == np.nan)] = 0.0
        # fit_score = zmf(phi95)
        # return fit_score

    def calculate_snr(self):
        # logging.info('calculating SNR ' + str(fov) + ' ' + str(subset) + ' ' + str(split_type))
        snr = {'A': [], 'C': [], 'G': [], 'T': []}
        for y, cycle in enumerate(self.cycle_range):  # range(self.fin_max_args.shape[1]):
            cycle_filter = self.cal.BasesDigit[cycle] != 78
            max_channel = self.fin_max_args[cycle_filter, y]
            ints = self.fin[cycle_filter, :, y]
            mean_val = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
            dev_val = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
            noise_val = [[], [], [], []]
            snr_mean = [0, 0, 0, 0]
            signal_count = [0, 0, 0, 0]
            for i in range(len(ints)):
                for c in range(4):
                    if c == max_channel[i]:
                        mean_val[max_channel[i]][c] += ints[i][c]
                    else:
                        mean_val[max_channel[i]][c] += ints[i][c]
                        noise_val[c].append(ints[i][c])
                signal_count[max_channel[i]] += 1
            for c in range(4):
                for j in range(4):
                    try:
                        mean_val[c][j] /= signal_count[c]
                    except (IndexError, ValueError, ZeroDivisionError):
                        mean_val[c][j] = 0
                    if c == j:
                        pass
                    else:
                        for i in range(signal_count[c]):
                            try:
                                dev_val[c][j] += (noise_val[c][i] - mean_val[c][j]) ** 2
                            except (IndexError, ValueError, ZeroDivisionError):
                                dev_val[c][j] = 0
                        if signal_count[c] == 0:
                            dev_val[c][j] = 0
                        else:
                            dev_val[c][j] = np.sqrt(dev_val[c][j] / signal_count[c])
                        dev_val[c][c] += dev_val[c][j]
                dev_val[c][c] /= 3
                if dev_val[c][c] == 0:
                    snr_mean[c] = 0
                else:
                    snr_mean[c] = mean_val[c][c] / dev_val[c][c]
            snr['A'].append(snr_mean[0])
            snr['C'].append(snr_mean[1])
            snr['G'].append(snr_mean[2])
            snr['T'].append(snr_mean[3])
        return snr

    def calc_snr_fast(self):
        snr = {'A': [], 'C': [], 'G': [], 'T': []}
        ch_masks = ~np.identity(4, bool)
        for y, cycle in enumerate(self.cycle_range):  # range(self.fin_max_args.shape[1]):
            cycle_filter = (self.cal.basesDigit[cycle] != 78)[self.subset_bool]
            cycle_max = self.fin_max_args[cycle_filter, y]
            cycle_ints = self.fin[cycle_filter, :, y]
            mean_val = [[], [], [], []]
            dev_val = [[0, 0, 0, 0],
                       [0, 0, 0, 0],
                       [0, 0, 0, 0],
                       [0, 0, 0, 0]]
            noise_val = [[], [], [], []]
            snr_mean = [0, 0, 0, 0]
            signal_count = [0, 0, 0, 0]
            for c in range(4):
                called_dnbs = cycle_ints[cycle_max == c]
                # Intensity mean of each channel where c is called
                mean_val[c] = np.nan_to_num(np.nanmean(called_dnbs, axis=0)).tolist()
                # Channel not c intensity of DNBs where c is called
                noise_val[c] = called_dnbs[:, ch_masks[c]].ravel()
                signal_count[c] = called_dnbs.shape[0]  # Number of DNBs calling c
            for c in range(4):
                for j in range(4):
                    if c == j:
                        continue
                    else:
                        dev_val[c][j] = np.nansum((noise_val[c] - mean_val[c][j]) ** 2)
                        if signal_count[c] == 0:
                            dev_val[c][j] = 0
                        else:
                            dev_val[c][j] = np.sqrt(dev_val[c][j] / signal_count[c])
                        dev_val[c][c] += dev_val[c][j]
                if dev_val[c][c] == 0:
                    snr_mean[c] = 0
                else:
                    dev_val[c][c] = float(dev_val[c][c]) / 3.0
                    snr_mean[c] = mean_val[c][c] / dev_val[c][c]
            snr['A'].append(snr_mean[0])
            snr['C'].append(snr_mean[1])
            snr['G'].append(snr_mean[2])
            snr['T'].append(snr_mean[3])
        return snr

    def calculate_phasing(self, subset, out_path, split_type, fov, log_dp):
        phasing = SignalPhasing.Zebra_Phase_Calculation(split_type, fov, subset+'_'+split_type, self.ctc, out_path,
                                                        log_dp=log_dp)
        phase_dict = phasing.run()
        lag = {'A': [0], 'C': [0], 'G': [0], 'T': [0]}
        runon = {'A': [], 'C': [], 'G': [], 'T': []}
        for base in lag.keys():
            lag[base].extend(phase_dict['lag'][base]['perc'])
            runon[base].extend(phase_dict['runon'][base]['perc'])
        # lag['AVG'] = []
        # runon['AVG'] = []
        # base = 'A'
        # for i in range(len(lag[base])):
        #     lag_avg = 0
        #     runon_avg = 0
        #     for base in 'ATCG':
        #         lag_avg += lag[base][i]
        #         runon_avg += runon[base][i]
        #     lag['AVG'].append(float(lag_avg) / 4)
        #     runon['AVG'].append(float(runon_avg) / 4)
        return lag, runon


@full_traceback
def output_qc_metrics(subset, out_path, split_type, fov, orderlist, cal_file, log_dp):
    sub_log_fn = os.path.join(log_dp, '%s_%s_info.log' % (fov, subset))
    sub_error_log_fn = os.path.join(log_dp, '%s_%s_errors.log' % (fov, subset))
    override_dict = {'sub.log': sub_log_fn,
                     'sub_errors.log': sub_error_log_fn}
    setup_logging(overrides=override_dict)
    logger = logging.getLogger(__name__)
    orderlist = copy.copy(orderlist)
    orderlist.pop(0)
    qual_stats = {'FIT': {}, 'BIC': {}, 'SNR': {}, 'CBI': {}, 'RHO': {}, 'lag': {}, 'runon': {}, 'SIGNAL': {},
                  'BACKGROUND': {}, 'CycQ30': {}, 'DNBNUM': {}, 'ACGTN': {}}
    logger.info('loading intensities, calFile')
    fin_ints = np.load(os.path.join(out_path, fov + '_finInts.npy'), mmap_mode='r+')
    raw_ints = np.load(os.path.join(out_path, fov + '_rawInts.npy'), mmap_mode='r+')
    try:
        ctc_ints = np.load(os.path.join(out_path, fov + '_ctcInts.npy'), mmap_mode='r+')
        ctc_ints_bool = True
    except:
        ctc_ints = np.zeros_like(fin_ints)
        ctc_ints_bool = False
    cal_obj = calReader.Cal()
    zero_list = np.zeros(raw_ints.shape[2]).tolist()
    if os.path.isfile(os.path.join(out_path, fov + '_Block_Bool.npy')):
        logger.info('Blocks subsetting, loading blocks boolean %s' % os.path.join(out_path, fov + '_Block_Bool.npy'))
        blocks_bool = np.load(os.path.join(out_path, fov + '_Block_Bool.npy'))
        cal_obj.load(cal_file, blocks_bool)
    else:
        cal_obj.load(cal_file)
    logger.info('loading normalization parameters and background')
    background_npy = np.load(os.path.join(out_path, fov + '_background.npy'))
    norm_paras = np.load(os.path.join(out_path, fov + '_normParas.npy'))
    if subset == 'Total':
        label_mask = np.ones(len(fin_ints), dtype=bool)
    else:
        label_vector = np.load(os.path.join(out_path, fov + split_type + '.npy'))
        label_mask = (label_vector == orderlist.index(subset))
    try:
        qc = CalQCStats(fin_ints[label_mask, :, :], raw_ints[label_mask, :, :],
                        ctc_ints[label_mask, :, :], norm_paras, cal_obj, label_mask,
                        background=background_npy)
        try:
            logger.info('Calculating RHO')
            qual_stats['RHO'][subset] = qc.get_rho_phi()
        except Exception as e:
            tb = str(traceback.format_exc())
            logger.error('RHO Calc Failure : ' + tb)
            logger.error('Filling RHO With Zeros')
            qual_stats['RHO'][subset] = {'A': zero_list, 'C': zero_list, 'G': zero_list, 'T': zero_list}
        qual_stats['DNBNUM'][subset] = int(label_mask.sum())
        try:
            logger.info('calculating SNR')
            qual_stats['SNR'][subset] = qc.calc_snr_fast()
        except:
            tb = str(traceback.format_exc())
            logger.error('SNR Calc Failure : ' + tb)
            logger.error('Filling SNR With Zeros')
            qual_stats['SNR'][subset] = {'A': zero_list, 'C': zero_list, 'G': zero_list, 'T': zero_list}
        try:
            logger.info('calculating BIC')
            qual_stats["BIC"][subset] = list(qc.bic())
        except:
            tb = str(traceback.format_exc())
            logger.error('BIC Calc Failure : ' + tb)
            logger.error('Filling BIC With Zeros')
            qual_stats["BIC"][subset] = zero_list
        try:
            logger.info('calculating FIT')
            qual_stats["FIT"][subset] = list(qc.phi_call_vect())
        except:
            tb = str(traceback.format_exc())
            logger.error('FIT Calc Failure : ' + tb)
            logger.error('Filling FIT With Zeros')
            qual_stats["FIT"][subset] = zero_list
        try:
            qual_stats["CBI"][subset] = list(qc.cbi_curve())
        except:
            qual_stats["CBI"][subset] = zero_list
            logger.error('CBI extraction error')
        try:
            logger.info('Calculating Phasing')
            qual_stats['lag'][subset], qual_stats['runon'][subset] = qc.calculate_phasing(subset, out_path, split_type, fov,
                                                                                          log_dp=out_path)
        except Exception as ex:
            tb = str(traceback.format_exc())
            logger.error('Phasing Calc Failure : ' + tb)
            logger.error('Filling Phasing With Zeros')
            qual_stats['lag'][subset], qual_stats['runon'][subset] = {'A': zero_list, 'C': zero_list,
                                                                      'G': zero_list, 'T': zero_list,
                                                                      'AVG': zero_list}, \
                                                                     {'A': zero_list, 'C': zero_list,
                                                                      'G': zero_list, 'T': zero_list,
                                                                      'AVG': zero_list}
        try:
            logger.info('Calculating SIGNAL')
            qual_stats['SIGNAL'][subset], _ = qc.get_signal()
        except:
            tb = str(traceback.format_exc())
            logger.error('SIGNAL Calc Failure : ' + tb)
            logger.error('Filling SIGNAL With Zeros')
            qual_stats['SIGNAL'][subset] = {'A': zero_list, 'C': zero_list, 'G': zero_list, 'T': zero_list}
        try:
            logger.info('Calculating BACKGROUND')
            qual_stats['BACKGROUND'][subset] = qc.get_background()
        except :
            tb = str(traceback.format_exc())
            logger.error('BACKGROUND Calc Failure : ' + tb)
            logger.error('Filling BACKGROUND With Zeros')
            qual_stats['BACKGROUND'][subset] = zero_list
        try:
            logger.info('Calculating CycQ30 and ACGTN counts')
            qual_stats['CycQ30'][subset], qual_stats['ACGTN'][subset] = qc.get_cycQ30_acgtn()
        except:
            tb = str(traceback.format_exc())
            logger.error('CycQ30 and ACGTN counts Calc Failure : ' + tb)
            logger.error('Filling CycQ30 and ACGTN counts With Zeros')
            qual_stats['CycQ30'][subset], qual_stats['ACGTN'][subset] = zero_list, {'A': zero_list, 'C': zero_list,
                                                                                    'G': zero_list, 'T': zero_list,
                                                                                    'N': zero_list}
    except Exception as e:
        tb = str(traceback.format_exc())
        logger.error('Other Exception : ' + tb)
    del fin_ints
    del raw_ints
    del ctc_ints
    return [split_type, fov, subset, qual_stats]


def label_to_subset(label_type, labels, fin_ints_file=None, sigma_val=0.25, outlier=False, label_dir='', fov=''):
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
        'A-nCBI',
        'C-nCBI',
        'G-nCBI',
        'T-nCBI'
    ]
    label_dict = dict((label, c) for c, label in enumerate(label_order))
    na_cbi_file = ''
    signal = 0
    sorted_data = 0
    # if fin_ints_file:
    #     na_cbi_file = glob.glob(os.path.join(label_dir, fov, fov + '*avg_CBI.npy'))[0]
    subset_order = ['Empty', 'Mixed', 'Poor', 'Low',
                    'mLow', 'mHigh', 'High', 'Outlier']
    label_vector = np.ones_like(labels[0, :])*-1
    # label_dict = {'percCBI': 0, 'Chastity': 2, 'children': 3, 'parents': 4, 'MixedSplit': 5,
    #               'percReadSNR': 9, 'SNR': 8, 'normCBI': 12}
    perc_cbi = labels[label_dict['PercCBI'], :]
    # if ('Sigma' in label_type) or ('SNR' in label_type):
    #     fin_ints = np.load(fin_ints_file)
    #     sorted_data = np.sort(fin_ints[:, :, :10], 1)
    #     signal = sorted_data[:, 3, :].mean(axis=1)
    label_vector[np.where(perc_cbi < 0)] = subset_order.index('Empty')
    if 'Mixed' in label_type:
        subset_order.pop(subset_order.index('Mixed'))
    if 'Int' in label_type:
        if outlier:
            label_vector[np.where(perc_cbi > 100)] = subset_order.index('Outlier')
        else:
            subset_order.pop(subset_order.index('Outlier'))
        # if 'Sigma' in label_type:
        #     norm_cbi = ''
        #     label_vector[np.where((norm_cbi <= sigma_val) & (perc_cbi > 0))] = subset_order.index('Poor')
        #     percentile = list()
        #     for i in [0, 1, 2, 3, 4]:
        #         percentile.append(np.percentile(signal[np.where(label_vector == -1)], i*25))
        #         # j += 1
        #     for i, quartile in enumerate(['Low', 'mLow', 'mHigh', 'High']):
        #         label_vector[np.where((signal > percentile[i]) &
        #                               (signal <= percentile[i+1]) &
        #                               (label_vector == -1))] = subset_order.index(quartile)
        #     if not outlier:
        #         label_vector[np.where(perc_cbi > 100)] = subset_order.index('High')
        # else:
        subset_order.pop(subset_order.index('Poor'))
        for i, quartile in enumerate(['Low', 'mLow', 'mHigh', 'High']):
            label_vector[np.where((perc_cbi > i*25) &
                                  (perc_cbi <= (i+1)*25))] = subset_order.index(quartile)
        if not outlier:
            label_vector[np.where(perc_cbi > 100)] = subset_order.index('High')
    # elif 'SNR' in label_type:
    #     noise = sorted_data[:, :3, :].reshape(sorted_data.shape[0], 3 * sorted_data.shape[2]).std(axis=1)
    #     snr = signal / (noise + 0.001)
    #     # snr = labels[label_dict['percReadSNR'], :]
    #     # snr[np.where(snr < 0)] = 0
    #     if 'Sigma' in label_type:
    #         norm_cbi = np.load(na_cbi_file)
    #         label_vector[np.where((norm_cbi <= sigma_val) & (perc_cbi > 0))] = subset_order.index('Poor')
    #     else:
    #         subset_order.pop(subset_order.index('Poor'))
    #     percentile = list()
    #     for j in range(5):
    #         percentile.append(np.percentile(snr[np.where(label_vector == -1)], j * 25))
    #     for i, quartile in enumerate(['Low', 'mLow', 'mHigh', 'High']):
    #         label_vector[np.where((snr > percentile[i]) &
    #                               (snr <= percentile[i + 1]) &
    #                               (label_vector == -1))] = subset_order.index(quartile)
    label_vector[np.where(label_vector == -1)] = 0
    return label_vector


def get_mixed(labels, chastity_thresh=7):
    mix_vect = np.zeros_like(labels[0, :])
    mix_vect[np.where(labels[1] > 1)] = 2
    mix_vect[np.where(labels[2] < chastity_thresh)] = 1
    mix_vect[np.where((labels[2] < chastity_thresh) & (labels[1] > 1))] = 3
    return mix_vect


def get_mirage(labels):
    mirage_vect = np.zeros_like(labels[0, :])
    mirage_vect[np.where(labels[3, :] > 0)] = labels[3, :][np.where(labels[3, :] > 0)]
    mirage_vect[np.where(labels[4, :] > 0)] = -1
    return mirage_vect


def crosstalk(intensities):
    alpha = 0.07
    clipped = np.all(intensities == 0, axis=1)
    init_cycle = 10
    intensities = intensities.astype(np.float32)
    for i in range(3):
        # crosstalk using only matrix calculated from cycle
        # print('Calculate Crosstalk Matrix..............................................')
        # crosstalk matrix calculation for init_cycle
        crosstalk_matrix, _ = cc.crosstalk(intensities[~clipped[:, init_cycle], :, init_cycle],
                                           mode='gmm',
                                           num_clstr=2)
        if not type(crosstalk_matrix) == np.ndarray:
            # print('crosstalk_error')
            init_cycle += 1
        print(crosstalk_matrix)
        if np.abs(crosstalk_matrix)[crosstalk_matrix != 1].max() <= alpha:
            corr_intensities = cc.apply_crosstalk_correction(intensities, crosstalk_matrix, clipped)
            break
        # %%
        # print('Apply Crosstalk Correction..............................................')
        # multiple intensities by inverse crosstalk matrix
        corr_intensities = cc.apply_crosstalk_correction(intensities, crosstalk_matrix, clipped)
    return corr_intensities


def get_inserts(fov, insert_dir, out_path):
    labels = np.load(glob.glob(os.path.join(insert_dir, fov + '-Total_mapping.npy')[0]))
    temp_inserts = labels[-1, :]
    inserts = np.zeros_like(temp_inserts)
    for i, bounds in enumerate([[0, 250], [250, 350], [350, 450], [450, 550], [550, np.inf]]):
        inserts[np.where((temp_inserts > bounds[0]) & (temp_inserts <= bounds[1]))] = i
    np.save(os.path.join(out_path, fov + 'Insert_Size' + '.npy'), inserts.astype(int))


@full_traceback
def single(data_path, out_path, fov, split_types, split_range, label_dir, insert_dir, subset_qc=True, v2=False,
           center=False, log_dp='', log_overrides={}):
    sub_log_fn = os.path.join(log_dp, '%s_info.log' % fov)
    sub_error_log_fn = os.path.join(log_dp, '%s_errors.log' % fov)
    override_dict = {'sub.log': sub_log_fn,
                     'sub_errors.log': sub_error_log_fn}
    override_dict.update(log_overrides)
    setup_logging(overrides=override_dict)
    logger = logging.getLogger(__name__)
    try:
        logger.info('occupancy directory %s' % label_dir)
        label_path = os.path.join(label_dir, fov, '*Labels.npy')
        logger.info('Label path %s' % label_path)
        labels_path = glob.glob(label_path)[0]
    except:
        logger.error('Occupancy Labels %s could not be found, removing fov from subsetting' % label_path)
        return [fov]
    split_len = len((range(split_range[0], split_range[1])))
    logger.info('QC calculation cycles: ' + str(list((range(split_range[0], split_range[1])))))
    cycles = os.listdir(os.path.join(data_path, 'finInts'))
    cycles.sort()
    fname_dnbs = os.path.join(data_path, 'finInts', 'S001', '{0}.QC.txt'.format(fov))
    labels = np.load(labels_path)
    num_dnbs = len(labels[0, :])
    logger.info('creating intensity .npy files')
    if os.path.exists(os.path.join(out_path, fov + '_finInts.npy')):
        logger.info('intensity .npys already exist bypassing extraction')
        pass
    else:
        if os.path.exists(fname_dnbs) & subset_qc & (v2==False):
            logger.info('extracting V1 ints from .bin files')
            with open(fname_dnbs, 'rb') as f:
                nextrow = 0
                for row in f:
                    if 'NUMDNB' in row:
                        nextrow = 1
                    elif nextrow:
                        num_dnbs = int(row)
                        break
            fin_ints = np.empty((num_dnbs, 4, split_len), dtype=np.float16)
            raw_ints = np.empty((num_dnbs, 4, split_len), dtype=np.float16)
            mid_ints = np.empty((num_dnbs, 4, split_len), dtype=np.float16)
            j = 0
            for i in range(split_range[0], split_range[1]):
                if os.path.exists(os.path.join(data_path, 'finInts', cycles[i],
                                               '{}.bin'.format(fov))):
                    fin_ints[:, :, j] = extract_intensities_np.main(
                            os.path.join(data_path, 'finInts', cycles[i],
                                         '{}.bin'.format(fov)))
                    raw_ints[:, :, j] = extract_intensities_np.main(
                            os.path.join(data_path, 'rawInts', cycles[i],
                                         '{}.bin'.format(fov)))
                    mid_ints[:, :, j] = extract_intensities_np.main(
                            os.path.join(data_path, 'midInts', cycles[i],
                                         '{}.bin'.format(fov)))
                    j += 1
                    # sys.stdout.write('\r   Loading {0:0.2f}%'.format(
                    #        100*(i+1)/float(len(cycles))))
            logging.info(fov+' intensities extracted')
            np.save(os.path.join(out_path, fov + '_finInts.npy'), fin_ints)
            np.save(os.path.join(out_path, fov + '_midInts.npy'), mid_ints)
            np.save(os.path.join(out_path, fov + '_rawInts.npy'), raw_ints)
            ctc_ints = crosstalk(raw_ints).astype(np.float16)
            np.save(os.path.join(out_path, fov + '_ctcInts.npy'), ctc_ints.astype(np.float16))
        elif v2:
            logger.info('extracting int information from v2 .int files')
            for int_type in ['_finInts.npy', '_rawInts.npy']:
                if os.path.exists(os.path.join(out_path, fov + int_type)):
                    pass
                else:
                    posinfo_fp, npy_fp, center_block_bool = get_posiIndex_Npy(os.path.join(data_path,
                                                                                           int_type[1:8]),
                                                                              fov, out_path,
                                                                              cycle_range=split_range,
                                                                              center=center)
            if not os.path.exists(os.path.join(out_path, fov + '_ctcInts.npy')):
                raw_ints = np.load(os.path.join(out_path, fov + '_rawInts.npy'))
                logger.info('generating crosstalk corrected raw intensities')
                ctc_ints = crosstalk(raw_ints)
                np.save(os.path.join(out_path, fov + '_ctcInts.npy'), ctc_ints)
                raw_ints = 0
                ctc_ints = 0
    mixed_vect = get_mixed(labels)
    for split_type in split_types:
        logger.info('processing labels for ' + split_type)
        if 'Size' in split_type:

            get_inserts(fov, insert_dir, out_path)
        else:
            fin_ints_file = os.path.join(out_path, fov + '_finInts.npy')
            try:
                label_vector = label_to_subset(split_type, labels, fin_ints_file, label_dir=label_dir, fov=fov)
            except IndexError:
                e = traceback.format_exc()
                logger.error('Failure in label vector generation error: ' + e + '\n removing fov from consideration')
                return [fov]
            if ('Mixed' not in split_type) and ('Size' not in split_type):
                logger.info('caclulating %mixed of each subset')
                label_counts = []
                nonmixed = np.unique(label_vector)
                for i in nonmixed:
                    label_counts.append((label_vector == i).sum())
                label_vector[np.where(mixed_vect > 0)] = 1
                k = 0
                for i in nonmixed:
                    label_counts[k] = 100.0 * (1.0 - float((label_vector == i).sum()) / float(label_counts[k]))
                    k += 1
                perc_mixed = float((label_vector == 1).sum()) / float(len(label_vector))
                label_counts.insert(1, 100.0)
                label_counts.insert(0, perc_mixed * 100)
                label_counts = np.round(np.array(label_counts), 3)
                np.save(os.path.join(out_path, fov + split_type), label_vector.astype(int))
                np.save(os.path.join(out_path, 'counts' + split_type + fov), label_counts)
            else:
                np.save(os.path.join(out_path, fov + split_type), label_vector.astype(int))
    mixed_table = [['Category', 'Mixed', 'Low Chastity', 'Mixed UnChaste', 'Unmixed Low Chaste'],
                   ['Overall'], ['Empty'], ['Low'], ['mLow'], ['mHigh'], ['High']]
    mirage_table = [['Category', 'Child', 'Parent', 'Avg Children'],
                    ['Overall'], ['Empty'], ['Low'], ['mLow'], ['mHigh'], ['High']]
    mixed_dict = {'Mixed': {'Overall': [], 'Empty': [], 'Low': [], 'mLow': [], 'mHigh': [], 'High': []},
                  'Low Chastity': {'Overall': [], 'Empty': [], 'Low': [], 'mLow': [], 'mHigh': [], 'High': []},
                  'Mixed UnChaste': {'Overall': [], 'Empty': [], 'Low': [], 'mLow': [], 'mHigh': [], 'High': []},
                  'Unmixed Low Chaste': {'Overall': [], 'Empty': [], 'Low': [], 'mLow': [], 'mHigh': [], 'High': []}}
    mirage_dict = {'Child': {'Overall': [], 'Empty': [], 'Low': [], 'mLow': [], 'mHigh': [], 'High': []},
                   'Parent': {'Overall': [], 'Empty': [], 'Low': [], 'mLow': [], 'mHigh': [], 'High': []},
                   'Avg Children': {'Overall': [], 'Empty': [], 'Low': [], 'mLow': [], 'mHigh': [], 'High': []}}
    key_dict = {'Empty': 0, 'Low': 1, 'mLow': 2, 'mHigh': 3, 'High': 4}
    mixed_keys = {'Mixed': [2, 3], 'Low Chastity': [1, 3], 'Mixed UnChaste': [2, 2], 'Unmixed Low Chaste': [1, 1]}
    label_vector = label_to_subset('_IntSubsetNoMixed', labels)
    logger.info('creating mixed and mirage count tables')
    for key in mixed_dict.keys():
        for val in mixed_dict[key].keys():
            if val == 'Overall':
                mixed_dict[key][val] = 100.0*((mixed_vect == mixed_keys[key][0]) |
                                              (mixed_vect == mixed_keys[key][1])).sum()/float(len(labels[0, :]))
            else:
                mixed_dict[key][val] = 100.0*(((mixed_vect == mixed_keys[key][0]) |
                                              (mixed_vect == mixed_keys[key][1])) &
                                              (label_vector == key_dict[val])).sum() / float(
                                              (label_vector == key_dict[val]).sum())
    mirage_vect = get_mirage(labels)
    for val in mirage_dict['Child'].keys():
        if val == 'Overall':
            mirage_dict['Child'][val] = 100.0*(labels[2, :] < 0).sum()/float(len(labels[0, :]))
            mirage_dict['Parent'][val] = 100.0*(labels[2, :] > 0).sum()/float(len(labels[0, :]))
            mirage_dict['Avg Children'][val] = labels[2, :][(labels[2, :] > 0)].mean()
        else:
            mirage_dict['Child'][val] = 100.0*((labels[2, :] < 0)
                                               & (label_vector == key_dict[val])).sum() / float(
                                               (label_vector == key_dict[val]).sum())
            mirage_dict['Parent'][val] = 100.0*((labels[2, :] > 0) & (label_vector == key_dict[val])).sum() / float(
                                                (label_vector == key_dict[val]).sum())
            mirage_dict['Avg Children'][val] = mirage_vect[((mirage_vect > 0) & (label_vector == key_dict[val]))].mean()
    for i in range(1, 5):
        for j in range(1, 7):
            cat = mixed_table[0][i]
            subset = mixed_table[j][0]
            mixed_table[j].append(str(mixed_dict[cat][subset]))
    for i in range(1, 4):
        for j in range(1, 7):
            cat = mirage_table[0][i]
            subset = mirage_table[j][0]
            mirage_table[j].append(str(mirage_dict[cat][subset]))
    np.save(os.path.join(out_path, fov + '_mixed_table'), mixed_table)
    np.save(os.path.join(out_path, fov + '_mirage_table'), mirage_table)
    # out_vals.put([fov, fin_ints, raw_ints, mid_ints])
    return


def main(data_path, out_path, fovs, split_types, split_range, label_dir, insert_dir, log_dp='',
         log_overrides=None, subset_qc=True, v2=False, center=False):
    if not log_overrides:
        log_overrides = {}
    sub_log_fn = os.path.join(log_dp, 'info.log')
    sub_error_log_fn = os.path.join(log_dp, 'errors.log')
    override_dict = {'sub.log': sub_log_fn,
                     'sub_errors.log': sub_error_log_fn}
    override_dict.update(log_overrides)
    setup_logging(overrides=override_dict)
    logger = logging.getLogger(__name__)
    logger.info('Initiating intensity classification...')
    logger.info('label_dir: %s' % label_dir)
    int_pool = mp.Pool(processes=len(fovs))
    int_pool_list = [int_pool.apply_async(single,
                     args=(data_path, out_path, field, split_types, split_range, label_dir, insert_dir, subset_qc, v2,
                           center, log_dp, {}))
                     for field in fovs]
    failed_list = [q.get() for q in int_pool_list]
    logger.error('intensity extraction failed for ' + str(failed_list))
    # logger.info('loaded')
    int_pool.close()
    # logger.info('close')
    int_pool.join()
    # logger.info('joined')
    # logger.info('gotten')
    subset_type_dict = {'_IntSubsetNoMixed': ['Total', 'Empty', 'Low', 'mLow', 'mHigh', 'High'],
                        # '_CBI_Sigma': ['Total', 'Empty', 'Poor', 'Low', 'mLow', 'mHigh', 'High'],
                        '_IntSubset': ['Total', 'Empty', 'Mixed', 'Low', 'mLow', 'mHigh', 'High'],
                        # '_CBI_Mixed_Sigma': ['Total', 'Empty', 'Mixed', 'Poor', 'Low', 'mLow', 'mHigh', 'High'],
                        # '_SNR': ['Total', 'Empty', 'Low', 'mLow', 'mHigh', 'High'],
                        # '_SNR_Sigma': ['Total', 'Empty', 'Poor', 'Low', 'mLow', 'mHigh', 'High'],
                        # '_SNR_Mixed': ['Total', 'Empty', 'Mixed', 'Low', 'mLow', 'mHigh', 'High'],
                        # '_SNR_Mixed_Sigma': ['Total', 'Empty', 'Mixed', 'Poor', 'Low', 'mLow', 'mHigh', 'High'],
                        '_SizeSubset': ['Total', '0bp', '<250bp', '250-350bp', '350-450bp', '450-550bp', '>550bp']
                        }
    p_count = 0
    for entry in failed_list:
        # print(entry)
        if type(entry) is list:
            fovs.pop(fovs.index(entry[0]))
    for split_type in split_types:
        p_count += len(subset_type_dict[split_type])
    qs_pool = mp.Pool(processes=p_count)
    qs_pool_list = []
    logger.info('populating multiprocessing queue for all fov-subsets')
    for fov in fovs:
        calFile = os.path.join(data_path, 'calFile', fov + '.cal')
        # print(calFile)
        for split_type in split_types:
            if not os.path.isfile(os.path.join(out_path, fov + split_type + '_QC_Data.p')):
                for subset in subset_type_dict[split_type]:
                    q = qs_pool.apply_async(output_qc_metrics, args=(subset, out_path, split_type, fov,
                                                                     subset_type_dict[split_type],
                                                                     calFile, log_dp))
                    qs_pool_list.append(q)
    qs_pool.close()
    qs_pool.join()
    if not qs_pool_list:
        return fovs
    qs = [q.get() for q in qs_pool_list]
    vals = {}
    logger.info('writing out QC_Data.p')
    for split_type in split_types:
        vals[split_type] = {}
    for key in vals.keys():
        for fov in fovs:
            vals[key][fov] = {}
    for value in qs:
        vals[value[0]][value[1]][value[2]] = value[3]
    for key in vals.keys():
        for fov in vals[key].keys():
            qc_out = os.path.join(out_path, fov + key + '_QC_Data.p')
            with open(qc_out, 'wb') as fo:
                pickle.dump(vals[key][fov], fo)
    # print (fovs)
    return fovs


if __name__ == '__main__':
    main(str(sys.argv[1]), str(sys.argv[2]), ['C002R008', 'C002R023', 'C002R038', 'C007R008'],
         ['_mixed', '_qrtle'], [0, 100], str(sys.argv[3]), str(sys.argv[3]))
