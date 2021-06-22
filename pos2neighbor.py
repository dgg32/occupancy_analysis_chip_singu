# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 09:30:26 2017

@author: aau
"""
import logging.config
logger = logging.getLogger(__name__)
from sap_funcs import setup_logging

import numpy as np
import sys

def find_neighbors(pos_idx, block, diagonals, v1):
    row = pos_idx[:,1] == block #filters dnbs in a block
    block_idx = pos_idx[row,0].astype(int)

    if v1:
        x = pos_idx[block_idx, 2].astype(int)
        y = pos_idx[block_idx, 3].astype(int)
    else:
        x = pos_idx[block_idx, 3].astype(int)
        y = pos_idx[block_idx, 2].astype(int)

    min_x  = x.min()
    max_x  = x.max()
    min_y  = y.min()
    max_y  = y.max()
    diff_x = max_x-min_x
#    diff_y = max_y-min_y
    #edges  = (y!=min_y) & (y!=max_y) & (x!=min_y) & (x!=max_y)
    
    left   = (x == min_x) 
    right  = (x == max_x)
    bottom = (y == min_y)
    top    = (y == max_y)
    bl     =  bottom & left
    br     = bottom & right
    tl     = top & left
    tr     = top & right
    middle = ~(left | right | bottom | top)
    
    #remove corners from edges
    left_edge   = np.where((left != top) & (left != bottom))[0]
    right_edge  = np.where((right != top) & (right != bottom))[0]
    bottom_edge = np.where((bottom != left) & (bottom != right))[0]
    top_edge    = np.where((top != left) & (top != right))[0]
    if diagonals:
        # all neighbors
        neighbors = np.full((block_idx.shape[0],9),-1,dtype=np.int32)
        #indices of surrounding dnbs
        neighbors_idx = np.array([-(diff_x+2), -(diff_x+1), -diff_x, -1, 1, diff_x, diff_x+1, diff_x+2]) 
        left_idx      = np.array([1,2,4,6,7])
        right_idx     = np.array([0,1,3,5,6])
        bottom_idx    = np.array([3,4,5,6,7])
        top_idx       = np.array([0,1,2,3,4])
        bl_idx        = np.array([4,6,7])
        br_idx        = np.array([3,5,6])
        tl_idx        = np.array([1,2,4])
        tr_idx        = np.array([0,1,3])
#        neighbors_idx_left   = np.array([-(diff_x+1), -diff_x, 1 , diff_x+1 , diff_x+2])
#        neighbors_idx_right  = np.array([-(diff_x+2), -(diff_x+1) , -1 , diff_x , diff_x+1])
#        neighbors_idx_bottom = np.array([-1 , 1 , diff_x , diff_x+1 , diff_x+2])
#        neighbors_idx_top    = np.array([-(diff_x+2) , -(diff_x+1) , -diff_x , -1 , 1])
#        neighbors_idx_bl     = np.array([-(diff_x+2), -(diff_x+1), -diff_x, -1, 1, diff_x, diff_x+1, diff_x+2])
#        neighbors_idx_br     = np.array([-(diff_x+2), -(diff_x+1), -diff_x, -1, 1, diff_x, diff_x+1, diff_x+2])
#        neighbors_idx_tl     = np.array([-(diff_x+2), -(diff_x+1), -diff_x, -1, 1, diff_x, diff_x+1, diff_x+2])
#        neighbors_idx_tr     = np.array([-(diff_x+2), -(diff_x+1), -diff_x, -1, 1, diff_x, diff_x+1, diff_x+2])
    else:
        #only adjacent neighbors (no diagonals)
        neighbors = np.full((block_idx.shape[0],5),-1,dtype=np.int32)
        neighbors_idx    = np.array([-(diff_x+1) , -1 , 1 , diff_x+1])
        left_idx      = np.array([1,2,4,6,7])
        right_idx     = np.array([0,1,3,5,6])
        bottom_idx    = np.array([3,4,5,6,7])
        top_idx       = np.array([0,1,2,3,4])
        bl_idx        = np.array([4,6,7])
        br_idx        = np.array([3,5,6])
        tl_idx        = np.array([1,2,4])
        tr_idx        = np.array([0,1,3])

    neighbors[:,0] = block_idx[:]
     #finds neighbors of dnbs in current block  
    neighbors[middle,1:]                        = block_idx[middle,None] + neighbors_idx[None,:]    
    neighbors[left_edge[:,None],left_idx+1]     = block_idx[left_edge,None] + neighbors_idx[None,left_idx]
    neighbors[right_edge[:,None],right_idx+1]   = block_idx[right_edge,None] + neighbors_idx[None,right_idx]
    neighbors[bottom_edge[:,None],bottom_idx+1] = block_idx[bottom_edge,None] + neighbors_idx[None,bottom_idx]
    neighbors[top_edge[:,None],top_idx+1]       = block_idx[top_edge,None] + neighbors_idx[None,top_idx]
    neighbors[bl,bl_idx+1]              = block_idx[bl,None] + neighbors_idx[None,bl_idx]
    neighbors[br,br_idx+1]              = block_idx[br,None] + neighbors_idx[None,br_idx]
    neighbors[tl,tl_idx+1]              = block_idx[tl,None] + neighbors_idx[None,tl_idx]
    neighbors[tr,tr_idx+1]              = block_idx[tr,None] + neighbors_idx[None,tr_idx]
             
    return neighbors


def main(pos_idx_path, coords_fp, neighbors_fp, blocks_fp, diagonals, block_list, v1=True):
    pos_idx = np.loadtxt(pos_idx_path,delimiter = '\t')
    logger.debug('pos_idx.shape: %s' % str(pos_idx.shape))

    if block_list:
        block_bool = np.in1d(pos_idx[:, 1], block_list)
        try:
            np.save(blocks_fp, block_bool)
        except:
            np.save('\\\\?\\' + blocks_fp, block_bool)

        logger.debug('block_bool.shape: %s' % str(block_bool.shape))

        pos_idx = pos_idx[block_bool]
        pos_idx[:, 0] = np.arange(len(pos_idx))

        logger.debug('pos_idx.shape: %s' % str(pos_idx.shape))
    else:
        block_list = range(len(np.unique(pos_idx[:, 1])))

    try:
        np.save(coords_fp, pos_idx[:,4:6].T)
    except:
        np.save('\\\\?\\' + coords_fp, pos_idx[:, 4:].T)

    if diagonals:
        neighbor_dnbs = np.full((pos_idx.shape[0],9),np.nan,dtype=np.int32)
    else:
        neighbor_dnbs = np.full((pos_idx.shape[0],5),np.nan,dtype=np.int32)
        
    for block in block_list:
        row = pos_idx[:, 1] == block #filters dnbs in a block
        block_idx = pos_idx[row, 0].astype(int)
        neighbor_dnbs[block_idx, :] = find_neighbors(pos_idx, block, diagonals, v1)
    try:
        np.save(neighbors_fp, neighbor_dnbs)
    except:
        np.save('\\\\?\\' + neighbors_fp, neighbor_dnbs)
    return

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3])