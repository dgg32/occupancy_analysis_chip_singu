# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 18:05:54 2017

@author: aau
"""
import numpy as np
import struct
import sys
import os
#import datetime

def HalfToFloat_np(h):
    s = ( np.right_shift(h,15) & 0x00000001 ).astype(int)    # sign
    e = ( np.right_shift(h,10) & 0x0000001f ).astype(int)    # exponent
    f = ( h                    & 0x000003ff ).astype(int)    # fraction
    x = np.zeros(h.shape,dtype=int) #initialize 
    
    # boolean filters
    case1 = (e==0)&(f==0) 
    case2 = (e==31)&(f==0) 
    case3 = (e==31)&(f!=0)     
    case5 = (e==0)&(f!=0)
    case4 = case5 & ((f & 0x00000400) == 0)
    case6 = (e!=31) & ~case1  
            
    x[case1] =   np.left_shift(s[case1],31).astype(int)
    x[case2] = ( np.left_shift(s[case2],31) | 0x7f800000 ).astype(int)
    x[case3] = ( np.left_shift( s[case3],31 ) | 0x7f800000  | np.left_shift(f[case3],13) ).astype(int)
    
    
    while np.any( case4):
        f[case4]  = np.left_shift(f[case4],1)
        e[case4] -= 1
        case4 = case5 & ((f & 0x00000400) == 0) #recalculate after changes to f
                              
    e[case5] += 1
    f[case5] &= ~0x00000400
        
    e[case6] += 112
    f[case6] = np.left_shift(f[case6],13)
    x[case6] = ( np.left_shift( s[case6],31 ) | np.left_shift( e[case6],23 ) | f[case6] ).astype(int)
    return x

def main(input_path):
    
#    step_start = datetime.datetime.now()
    platform   = sys.platform
    dnb_number = np.fromfile(input_path,count=1,dtype='i')[0]
    #print dnb_number

    with open(input_path,'rb') as f:
        f.seek(4,os.SEEK_SET)
        data = np.fromfile(f,dtype='H')

    x = HalfToFloat_np(data)
    
    # windows
    if 'win' in platform:
        int_sample = np.array(struct.unpack('<{}f'.format(len(x)), x))

    # #linux (twice the bits but why?)
    elif 'linux' in platform:
        int_sample = np.array(struct.unpack_from('<{}f'.format(len(x)*2), x))[::2]
    
#    step_stop = datetime.datetime.now()
#    ela_time = step_stop - step_start
#    print 'Write Intensities Sample Time:', (ela_time)
    
#    return int_sample
    return int_sample.reshape((dnb_number,4), order = 'f')


if __name__ == '__main__':
    main(sys.argv)
    
    
    
    
    
    
    