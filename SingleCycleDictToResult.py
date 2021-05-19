import os
import sys

base_list = ['Total', 'A', 'A_Singular', 'AC', 'AG', 'AT', 'C', 'C_Singular', 'CG', 'CT', 'AC', 'G',
             'G_Singular', 'AG', 'CG', 'GT', 'T_All', 'T', 'AT', 'CT', 'GT', 'T', 'M', 'E']
def dict_summary_list(excess_out, real_mean, sim_means, perc_out, perc_valid):
    out = []
    for b in base_list:
        if b in perc_out.keys():
            out.append(['%% Called %s of Total' % b, perc_out[b]])
    for b in base_list:
        if (b+'_Valid') in perc_out.keys():
            out.append(['%% Called %s of Valid' % (b), perc_out[b+'_Valid']])
    for b in base_list:
        for key in sorted(excess_out.keys()):
            if 'Excluding_Mixed_' in key:
                if 'Raw' in key:
                    if b in key:
                        out.append(['%% Excess Concordant Neighbors %s Excluding mixed match All DNBs' % (key.split('_')[0] + " " + key.split('_')[-1]), excess_out[key]])
    for b in base_list:
        for key in sorted(excess_out.keys()):
            if 'Including_Mixed_' in key:
                if 'Raw' in key:
                    if b in key:
                        out.append(['%% Excess Concordant Neighbors %s Including mixed match All DNBs' % (key.split('_')[0] + " " + key.split('_')[-1]) , excess_out[key]])
    out.append(['% valid Called', perc_valid['real']])
    out.append(['% valid Simulated', perc_valid['sim']])

    for b in base_list:
        for key in sorted(excess_out.keys()):
            if 'Excluding_Mixed_' in key:
                if 'Valid' in key:
                    if b in key:
                        out.append(['%% Excess Concordant Neighbors %s Excluding mixed match Valid DNBs' % (key.split('_')[0] + " " + key.split('_')[-1]), excess_out[key]])
    for b in base_list:
        for key in sorted(excess_out.keys()):
            if 'Including_Mixed_' in key:
                if 'Valid' in key:
                    if b in key:
                        out.append(['%% Excess Concordant Neighbors %s Including mixed match Valid DNBs'% (key.split('_')[0] + " " + key.split('_')[-1]), excess_out[key]])
    for b in base_list:
        for key in sorted(sim_means.keys()):
            if 'Excluding_Mixed_' in key:
                if 'Raw' in key:
                    if b in key:
                        out.append(['Mean Real Concordant Neighbors %s Excluding mixed match All DNBs' % (
                                key.split('_')[0] + " " + key.split('_')[-1]), real_mean[key]])
                        out.append(['Mean Sim Concordant Neighbors %s Excluding mixed match All DNBs' % (
                                    key.split('_')[0] + " " + key.split('_')[-1]), sim_means[key]])
    for b in base_list:
        for key in sorted(excess_out.keys()):
            if 'Including_Mixed_' in key:
                if 'Raw' in key:
                    if b in key:
                        out.append(['Mean Real Concordant Neighbors %s Including mixed match All DNBs' % (
                                key.split('_')[0] + " " + key.split('_')[-1]), real_mean[key]])
                        out.append(['Mean Sim Concordant Neighbors %s Including mixed match All DNBs' % (
                                key.split('_')[0] + " " + key.split('_')[-1]), sim_means[key]])
    for b in base_list:
        for key in sorted(real_mean.keys()):
            if 'Excluding_Mixed_' in key:
                if 'Valid' in key:
                    if b in key:
                        out.append(['Mean Real Concordant Neighbors %s Excluding mixed match Valid DNBs' % (
                                key.split('_')[0] + " " + key.split('_')[-1]), real_mean[key]])
                        out.append(['Mean Sim Concordant Neighbors %s Excluding mixed match Valid DNBs' % (
                                key.split('_')[0] + " " + key.split('_')[-1]), sim_means[key]])
    for b in base_list:
        for key in sorted(sim_means.keys()):
            if 'Including_Mixed_' in key:
                if 'Valid' in key:
                    if b in key:
                        out.append(['Mean Real Concordant Neighbors %s Including mixed match Valid DNBs' % (
                                key.split('_')[0] + " " + key.split('_')[-1]), real_mean[key]])
                        out.append(['Mean Sim Concordant Neighbors %s Including mixed match Valid DNBs' % (
                                key.split('_')[0] + " " + key.split('_')[-1]), sim_means[key]])
    return out