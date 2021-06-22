# -*- coding: utf-8 -*-
"""
Created on Tue Mar 07 10:56:43 2017

@author: aau
"""

# crosstalk correction 

import numpy as np
import itertools as itt
from sklearn.linear_model import LinearRegression, RANSACRegressor
from sklearn.mixture import GaussianMixture
from scipy.signal import savgol_filter
#from scipy.signal import argrelextrema
#import matplotlib.pyplot as plt
#from datetime import datetime

def find_threshold(intensities,nbins):
    smooth_win = int(nbins / 30) #arbitrary number for now but it works
    if (smooth_win%2)==0: #make sure smoothing window is odd
        smooth_win += 1
    maxima     = np.zeros((4))
    thresh     = np.zeros((4))
    for i in range(4):#number of channels (4)   
        x = intensities[:,i] #dnbs per channel
        #remove outliers for better resolution of histogram bins
        # filt_outliers_low, filt_outliers_high = np.nanpercentile(x,[,99.5])
        filt_outliers_low = np.nanpercentile(x, 50)
        filt_outliers_high = np.nanpercentile(x, 99)
        if (filt_outliers_low == np.nan) | (filt_outliers_low == np.nan):
            return False, False
        no_out = x[ (x > filt_outliers_low) & (x < filt_outliers_high) ]

        #find threshold to filter out null cluster
        hist, bin_edges = np.histogram(no_out, bins=nbins)
        std  = no_out.std()
    
        bin_centers = (bin_edges[:-1] + bin_edges[1:])/2
        
        smoothed1   = savgol_filter(hist, smooth_win, 1) #initial smoothing
        smoothed2   = savgol_filter(smoothed1, smooth_win, 2) #second smoothing
        
        maxima[i]   = bin_centers[ smoothed2.argmax() ]
        thresh[i]   = maxima[i] + std

    return thresh, maxima    

def normalize_intensities(intensities):
    # normalize each ch in each cycle by the 87.5th percentile of dnb intensities
    # assumption based off of uniform 25% base distribution for E.Coli
    return np.divide( intensities, np.percentile(intensities,87.5,axis=0),
                        out=np.zeros(intensities.shape),
                        where= np.any(intensities.sum(axis=1)!=0,axis=0))

# def normalize_size_subset(intensities,raw,lim):
#     avg_cbi = 

### tried to vectorize, but can't figure out way to vectorize the most inner loop and 3d array indexing is slow
# def subpixel_normalize(data):
#     x_coord_col = [3,6,9,12]
#     y_coord_col = [4,7,10,13]
#     int_col     = [5,8,11,14]
#     nbins = 50
#     xbins = np.linspace(0,1,nbins+1)
#     ybins = np.linspace(0,1,nbins+1)

#     cbi = data[:,int_col,:].max(axis=1)
#     bad_dnbs = cbi==cbi.max() #not clipped readsbin
    
#     low  = np.percentile(data[:,int_col,:],75,axis=0)
#     pct_filter = (data[:,int_col,:] >= low)
#     good_dnbs = (~bad_dnbs[:,np.newaxis,:]) & pct_filter
#     for cycle in xrange(data.shape[2]):
#         for i in xrange(4):
#             xbin = np.digitize(data[:,x_coord_col[i],cycle]%1,xbins)-1
#             ybin = np.digitize(data[:,y_coord_col[i],cycle]%1,ybins)-1
#             dnbs = ((xbin[:,None,None]==x[None,:,:]) & 
#                     (ybin[:,None,None]==y[None,:,:]) & 
#                     good_dnbs[:,i,cycle][:,None,None])
#             for x,y in itt.product(np.arange(len(xbins)-1),np.arange(len(ybins)-1)):
#                 data[ dnbs[:,x,y],int_col[i],cycle ] /= data[ dnbs[:,x,y],int_col[i],cycle ].mean()
#     return data[:,int_col,:]
def subpixel_normalize(data):
    x_coord_col = [3,6,9,12]
    y_coord_col = [4,7,10,13]
    int_col     = [5,8,11,14]
    nbins = 50
    xbins = np.linspace(0,1,nbins+1)
    ybins = np.linspace(0,1,nbins+1)
    x,y = np.meshgrid((xbins[:-1]+xbins[1:])/2, (ybins[:-1]+ybins[1:])/2)
    mag = np.sqrt( (x.reshape(-1,1,order='f')-0.5)**2 + (y.reshape(-1,1,order='f')-0.5)**2)

    cbi = data[:,int_col,:].max(axis=1)
    bad_dnbs = cbi==cbi.max() #not clipped readsbin
    # norm = np.zeros(data[:,int_col,:].shape,dtype=float)
    low  = np.percentile(data[:,int_col,:],75,axis=0)
    pct_filter = (data[:,int_col,:] >= low)
    times_called = pct_filter.sum(axis=1)==1
    for cycle in range(data.shape[2]):
        for i in range(len(int_col)):
            norm = np.zeros(len(mag),dtype=float)
            xbin = np.digitize(data[:,x_coord_col[i],cycle]%1,xbins)-1
            ybin = np.digitize(data[:,y_coord_col[i],cycle]%1,ybins)-1
            good_dnbs = (~bad_dnbs[:,cycle])&pct_filter[:,i,cycle]&times_called[:,cycle]
            for j,xy in enumerate(itt.product(np.arange(len(xbins)-1),np.arange(len(ybins)-1))):
                norm[j] = data[ (xbin==xy[0])&(ybin==xy[1])&good_dnbs,int_col[i],cycle ].mean()
            model = RANSACRegressor()
            model.fit(mag,norm.reshape(-1,1))
            mag_all = np.sqrt(((data[~bad_dnbs[:,cycle],x_coord_col[i],cycle]%1)-0.5)**2 +
                                ((data[~bad_dnbs[:,cycle],y_coord_col[i],cycle]%1)-0.5)**2)
            norm_const = model.predict(mag_all.reshape(-1,1))
            data[~bad_dnbs[:,cycle],int_col[i],cycle] /= norm_const.ravel()

    return data[:,int_col,:]

def crosstalk(intensities, mode = 'gmm', num_clstr=3, fit_intercept=True,
              plt_slopes=0,lim=[-1000,8000]):
    
    bases             = ['A','C','G','T']
    base_combinations = list(itt.combinations(bases,2))
    base_dict         = {base:idx for idx,base in enumerate(bases)}
    nbins             = int(round(np.sqrt(intensities.shape[0])))
    
    crosstalk_matrix  = np.eye(4,dtype=float)
    bg = np.zeros((4,4),dtype=float)

    thresh, _          = find_threshold(intensities, nbins)
    if type(thresh) != np.ndarray:
        return False, False
    if mode == 'cbi':
        ch_max = normalize_intensities(intensities).argmax(axis=1)
    
    for idx,pair in enumerate(base_combinations):
        base1 = base_dict[pair[0]]
        base2 = base_dict[pair[1]]
        
        x = intensities[:,base1]
        y = intensities[:,base2]
        upper_lim = 99.9
        x_filtered = (x > thresh[base1]) & (x < np.percentile(x,upper_lim)) & (y < np.percentile(x,upper_lim))
        y_filtered = (y > thresh[base2]) & (y < np.percentile(y,upper_lim)) & (x < np.percentile(x,upper_lim))

        for i in range(5):
            try:
                if mode == 'gmm':
                    x1 = x[x_filtered | y_filtered]
                    y1 = y[x_filtered | y_filtered]
                    # print x1.min(),x1.max(),y1.min(),y1.max()

            #        sub_sample = np.random.choice(np.arange(x1.shape[0]),x1.shape[0]/4,replace='False')
            #        x1 = x1[sub_sample]
            #        y1 = y1[sub_sample]        
                    
                    # initialize cluster centers with the "noise" cluster initialized faraway 
                    # to prevent getting stuck in local minima
                    if num_clstr == 3:
                        init = np.array([ [x1.mean()           , y1.mean()+y1.std()  ], 
                                        [x1.mean()+x1.std()  , y1.mean()           ],
                                        [x1.mean()+2.5*x1.std(), y1.mean()+2.5*y1.std()] ])
                    elif num_clstr == 2:
                        init = np.array([ [x1.mean()           , y1.mean()+y1.std()  ], 
                                        [x1.mean()+x1.std()  , y1.mean()           ] ])
                    
                    gmm  = GaussianMixture(n_components=num_clstr, 
                                    covariance_type = 'full', 
                                    n_init = 1, 
                                    means_init = init).fit( np.column_stack((x1,y1)) )
            #                        weights_init = intensities.shape[0]/3
            #                        init_params = 'wmc',
            #                        params='wmc') #initialize mixture model
                    clusters = gmm.predict( np.column_stack( (x1,y1) ) )
                    # print gmm.means_

                    # find clusters corresponding to upper and lower arms
                    # noise cluster corresponds to largest of the minimum variances in eigenvector direction
                    # (largest out of (the smallest eigenvalues for each cluster) )
                    if num_clstr ==3:
                        # noise_idx     = np.argmax( np.min( np.linalg.eigvals(gmm.covariances_), axis=1 ) )# assume noise cluster has largest variance (not always true)
                        # arms_idx      = np.delete(np.arange(num_clstr),noise_idx)
                        # print noise_idx
                        clstr_angle     = np.arctan(gmm.means_[:,1]/gmm.means_[:,0])
                        upper_arm_idx = np.argmax(clstr_angle)
                        lower_arm_idx = np.argmin(clstr_angle)
                        noise_idx      = np.delete(np.arange(num_clstr),[upper_arm_idx,lower_arm_idx])[0]
                    else:
                        arms_idx  = np.arange(num_clstr)

                        clstr_angle   = np.absolute(gmm.means_[arms_idx,1] / gmm.means_[arms_idx,0])
                        upper_arm_idx = arms_idx[ np.argmax( clstr_angle )]
                        lower_arm_idx = arms_idx[ np.argmin( clstr_angle )]
            
                    # assign upper and lower arms indicies and reshapes into a column vector
                    x_upper = x1[clusters==upper_arm_idx].reshape(-1,1)
                    y_upper = y1[clusters==upper_arm_idx].reshape(-1,1)
                    x_lower = x1[clusters==lower_arm_idx].reshape(-1,1)
                    y_lower = y1[clusters==lower_arm_idx].reshape(-1,1)
                
                if mode == 'cbi':
                    out_filt   = x_filtered | y_filtered
                    x_lower = x[out_filt & (ch_max==base1)].reshape(-1,1)
                    y_lower = y[out_filt & (ch_max==base1)].reshape(-1,1)
                    x_upper = x[out_filt & (ch_max==base2)].reshape(-1,1)
                    y_upper = y[out_filt & (ch_max==base2)].reshape(-1,1)

                regr_upper = LinearRegression(fit_intercept=fit_intercept,n_jobs=1)
                regr_lower = LinearRegression(fit_intercept=fit_intercept,n_jobs=1)
                regr_upper.fit(y_upper,x_upper)
                regr_lower.fit(x_lower,y_lower)
                # print regr_upper.score(y_upper,x_upper)
                # print regr_lower.score(x_lower,y_lower)
                
                #RANSACregressor to help with inaccurate arm assignment
                # regr_upper = RANSACRegressor(LinearRegression(fit_intercept=fit_intercept,n_jobs=1))
                # regr_lower = RANSACRegressor(LinearRegression(fit_intercept=fit_intercept,n_jobs=1))
                # regr_upper.fit(y_upper,x_upper)
                # regr_lower.fit(x_lower,y_lower)
            
            except ValueError:
                if i==4:
                    print('    {0}:Failed'.format(''.join(pair)))
                    raise
                print('    {0}:Retry'.format(''.join(pair)))
                continue
            else:
                print('    {0}:Passed'.format(''.join(pair)))
                break
        
        crosstalk_matrix[base1,base2] = regr_upper.coef_[0]
        crosstalk_matrix[base2,base1] = regr_lower.coef_[0]
        # crosstalk_matrix[base1,base2] = regr_upper.estimator_.coef_
        # crosstalk_matrix[base2,base1] = regr_lower.estimator_.coef_

        # print regr_lower.intercept_[0],regr_upper.intercept_[0],regr_lower.coef_[0],regr_upper.coef_[0]
        
        # y = mx+b
        m1 = regr_lower.coef_[0]
        m2 = 1/regr_upper.coef_[0]
        b1 = regr_lower.intercept_[0]
        b2 = -(m2*regr_upper.intercept_[0])
        bg[base1,base2] = (b2-b1) / (m1-m2)
        bg[base2,base1] = m1 * ( (b2-b1)/(m1-m2) ) + b1

        # if plt_slopes:
        #     if idx == 0:
        #         fig = plt.figure()
        #         colors = ['b','r','g','k','y','m']
        #
        #     ax = fig.add_subplot(2,3,idx+1,aspect='equal')
        #     ax.scatter(x_lower, y_lower,
        #                 marker = '.',
        #                 s=0.5,
        #                 alpha = 0.5,
        #                 color = colors[0])
        #     ax.scatter(x_upper, y_upper,
        #                 marker = '.',
        #                 s=0.5,
        #                 alpha = 0.5,
        #                 color = colors[1])
        #     if (num_clstr==3) and (mode=='gmm'):
        #          ax.scatter(x1[clusters==noise_idx], y1[clusters==noise_idx],
        #                 marker = '.',
        #                 s=0.5,
        #                 alpha = 0.5,
        #                 color = colors[2])
        #     if fit_intercept:
        #         ax.plot(np.linspace(lim[0],lim[1],2) + regr_upper.intercept_[0],
        #                 (1/regr_upper.coef_[0]) * np.linspace(lim[0],lim[1],2), color = 'k')
        #         ax.plot(np.linspace(lim[0],lim[1],2),
        #                 regr_lower.coef_[0] * np.linspace(lim[0],lim[1],2) + regr_lower.intercept_[0],
        #                 color = 'k')
        #     else:
        #         ax.plot(np.linspace(lim[0],lim[1],2),
        #                 (1/regr_upper.coef_[0]) * np.linspace(lim[0],lim[1],2), color = 'k')
        #         ax.plot(np.linspace(lim[0],lim[1],2),
        #                 regr_lower.coef_[0] * np.linspace(lim[0],lim[1],2),
        #                 color = 'k')
        #     ax.axis([lim[0],lim[1],lim[0],lim[1]])
        #     ax.set_title(''.join(pair))
            
    return crosstalk_matrix, bg

##apply crosstalk matrix to all 
def apply_crosstalk_correction(intensities,crosstalk_matrix,clipped):    
    corr_intensities      = np.empty((intensities.shape),dtype = np.float)
    inv_crosstalk_matrix  = np.linalg.inv(crosstalk_matrix )
    for cycle in range(intensities.shape[2]):
        corr_intensities[~clipped[:,cycle],:,cycle] = np.dot( inv_crosstalk_matrix, intensities[~clipped[:,cycle],:,cycle].T ).T
    return corr_intensities

def crosstalk_correction_cascade(intensities,crosstalk_matrix):
    # create crosstalk matrix with each ch filtered
    cross_mat_first  = np.tile( np.linalg.inv(crosstalk_matrix)[:,:,np.newaxis],
                               (1,1,4))
    cross_mat_second = np.tile( np.linalg.inv(crosstalk_matrix)[:,:,np.newaxis],
                               (1,1,4))
    ch_filt = np.eye(4)
    for ch in range(4):
        cross_mat_first[ch,:,ch] = ch_filt[ch,:]
        cross_mat_second[np.arange(4)!=ch,:,ch] = ch_filt[np.arange(4)!=ch,:]
        
    ch_max     = normalize_intensities(intensities).argmax(axis=1)
    corr_intensities      = np.empty((intensities.shape),dtype = np.float)
    for cycle in range(intensities.shape[2]):
        for ch in range(4):
            corr_intensities[ch_max[:,cycle]==ch,:,cycle] = np.dot( 
                                 cross_mat_first[:,:,ch], 
                                 intensities[ch_max[:,cycle]==ch,:,cycle].T ).T
            corr_intensities[ch_max[:,cycle]==ch,:,cycle] = np.dot(
                                 cross_mat_second[:,:,ch],
                                 corr_intensities[ch_max[:,cycle]==ch,:,cycle].T ).T
    return corr_intensities
            

  