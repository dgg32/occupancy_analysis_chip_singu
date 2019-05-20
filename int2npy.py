from intReader import IntensityBin
import os
import numpy as np
import sys
import csv
#%%
#path = '/home/aau/zebra_data/CL100055516/L01/FS2000/L01/rawInts/'


def get_coord(path):
    elem = path.split(os.path.sep)
    fov = elem[-1].split('.')[0]
    #use (os.path.sep).join() to get leading backslash
    cycles = [cycle for cycle in os.listdir((os.path.sep).join(elem[:-2]))
               if cycle.startswith('S')]
    cycles.sort()

    # spot_number = []
    # for i,fov in enumerate(fovs):
    # 	ib = IntensityBin(os.path.join(path,cycles[0],fov))
    # 	spot_number.append(ib.metrics.SpotNumber)
    ib = IntensityBin(path)
    data = np.zeros((ib.metrics.SpotNumber, 15, len(cycles)), dtype=np.float)

    #extract subpixel coords
    metList=["BlockId", "DnbRow", "DnbCol", "Coords", "Ints"]
    
    for j,cycle in enumerate(cycles):
#        print i,cycle
        int_path = os.path.join( (os.path.sep).join(elem[:-2]), cycle, '{0}.int'.format(fov) )
        if not os.path.exists(int_path):
            continue
        ib = IntensityBin(int_path)
        blockList = ib.metrics.ExtractOrder
        ct = 0
        for b in blockList:
            blk = ib.metrics.Blocks[b]
            blkData = ib.getMetricsByBlock(blk, metList)

            data[ct:ct+blkData.shape[0], :, j] = blkData
            ct += blkData.shape[0]
    return data


def get_coord_single_cycle(int_path, num_colors=4):
    elem       = int_path.split(os.path.sep)
    fov        = elem[-1].split('.')[0]
    #use (os.path.sep).join() to get leading backslash
    # cycles  = [cycle for cycle in os.listdir( (os.path.sep).join(elem[:-2])) 
    #                 if cycle.startswith('S')]
    # cycles.sort()

    # spot_number = []
    # for i,fov in enumerate(fovs):
    # 	ib = IntensityBin(os.path.join(path,cycles[0],fov))
    # 	spot_number.append(ib.metrics.SpotNumber)

    ib = IntensityBin(int_path)
    data = np.zeros((ib.metrics.SpotNumber, (num_colors+1)*3), dtype=np.float)

    #extract subpixel coords
    metList=["BlockId", "DnbRow", "DnbCol", "Coords", "Ints"]
    
    # int_path = os.path.join( (os.path.sep).join(elem[:-2]), cycle, '{0}.int'.format(fov) )
    if os.path.exists(int_path):
        ib = IntensityBin(int_path)
        blockList = ib.metrics.ExtractOrder
        ct = 0
        for b in blockList:
            blk = ib.metrics.Blocks[b]
            blkData = ib.getMetricsByBlock(blk, metList)
            
            data[ct:ct+blkData.shape[0], :] = blkData
            ct += blkData.shape[0]

    return data


def get_posiIndex_Npy(ints_path, fov, data_path, center=False, cycle_range=[0, None]):
    ints_type = ints_path.split(os.path.sep)[-1]
    occu_cycle = str(cycle_range[0]+3)
    int_path = os.path.join(ints_path, 'S000'[:-len(occu_cycle)]+occu_cycle, str(fov) + '.int')
    posIndex = os.path.join(data_path, str(fov) + '.posiIndex.txt')
    numpyArray = os.path.join(data_path, str(fov) + '_' + str(ints_type) + '.npy')
    if not os.path.isfile(posIndex):
        pos_data = get_coord_single_cycle(int_path)
        pos_data = pos_data[:, :5]
        dnb_ind = np.arange(len(pos_data)).reshape(len(pos_data), 1)
        pos_data = np.concatenate((dnb_ind, pos_data), axis=1)
        if center:
            block_bool_file = os.path.join(data_path, fov + '_Block_Bool.npy')
            if not os.path.isfile(block_bool_file):
                blocks = pos_data[:, 1].astype(int)
                center_block_bool = np.zeros(len(pos_data), dtype=bool)
                center_blocks = center
                for center_block in center_blocks:
                    center_block_bool[blocks == center_block] = True
                # center_block_bool = ((blocks == 44) | (blocks == 45) | (blocks == 54) | (blocks == 55))
                np.save(block_bool_file, center_block_bool)
            else:
                center_block_bool = np.load(block_bool_file)
            pos_data = pos_data[center_block_bool, :]
            pos_data[:, 0] = np.arange(len(pos_data))
        elif center == '4x4':
            block_bool_file = os.path.join(data_path, fov + '_Block_Bool.npy')
            if not os.path.isfile(block_bool_file):
                blocks = pos_data[:, 1].astype(int)
                center_block_bool = np.zeros(len(pos_data), dtype=bool)
                center_blocks = [33, 34, 35, 36, 43, 44, 45, 46,
                                 53, 54, 55, 56, 63, 64, 65, 66]
                for center_block in center_blocks:
                    center_block_bool[blocks == center_block] = True
                # center_block_bool = ((blocks == 44) | (blocks == 45) | (blocks == 54) | (blocks == 55))
                np.save(block_bool_file, center_block_bool)
            else:
                center_block_bool = np.load(block_bool_file)
            pos_data = pos_data[center_block_bool, :]
            pos_data[:, 0] = np.arange(len(pos_data))
        else:
            center_block_bool = False
        pos_list = pos_data.tolist()
        with open(posIndex, 'w') as posfile:
            writer = csv.writer(posfile, delimiter='\t', lineterminator='\n')
            writer.writerows(pos_list)
    else:
        if bool(center):
            block_bool_file = os.path.join(data_path, fov + '_Block_Bool.npy')
            if os.path.isfile(block_bool_file):
                center_block_bool = np.load(block_bool_file)
            else:
                pos_data = np.loadtxt(posIndex, delimiter='\t')
                blocks = pos_data[:, 1].astype(int)
                center_block_bool = np.zeros(len(pos_data), dtype=bool)
                center_blocks = center
                for center_block in center_blocks:
                    center_block_bool[blocks == center_block] = True
                # center_block_bool = ((blocks == 44) | (blocks == 45) | (blocks == 54) | (blocks == 55))
                np.save(block_bool_file, center_block_bool)
        elif center == '4x4':
            block_bool_file = os.path.join(data_path, fov + '_Block_Bool.npy')
            if os.path.isfile(block_bool_file):
                center_block_bool = np.load(block_bool_file)
            else:
                pos_data = np.loadtxt(posIndex, delimiter='\t')
                blocks = pos_data[:, 1].astype(int)
                center_block_bool = np.zeros(len(pos_data), dtype=bool)
                center_blocks = [33, 34, 35, 36, 43, 44, 45, 46,
                                 53, 54, 55, 56, 63, 64, 65, 66]
                for center_block in center_blocks:
                    center_block_bool[blocks == center_block] = True
                # center_block_bool = ((blocks == 44) | (blocks == 45) | (blocks == 54) | (blocks == 55))
                np.save(block_bool_file, center_block_bool)
        else:
            center_block_bool = False
    if not os.path.isfile(numpyArray):
        int_data, norm_paras, background = get_Ints(int_path, cycle_range=cycle_range)
        # sort_data = np.sort(int_data, axis=1)
        dnb, channel, cycle = np.where(int_data == int_data.max())
        # single_sat = (int_data[dnb, :, cycle] == int_data.max()).sum(axis=1) == 1
        int_data[dnb, :, cycle] = 0
        # int_data[dnb[~single_sat], :, cycle[~single_sat]] = 0
        # int_data[dnb[single_sat], channel[single_sat], cycle[single_sat]] = sort_data[dnb[single_sat],
        #                                                                               np.ones(int(single_sat.sum()),
        #                                                                                       dtype=int)*2,
        #                                                                               cycle[single_sat]]*4
        if center:
            int_data = int_data[center_block_bool, :, :]
        np.save(numpyArray, int_data.astype(np.float32))
        if 'fin' in ints_type:
            np.save(os.path.join(data_path, fov + '_normParas.npy'), norm_paras)
            if bool(center):
                background = background[np.array(center_blocks, dtype=int), :, :]
            background = background.mean(axis=0)
            np.save(os.path.join(data_path, fov + '_background.npy'), background)
    return posIndex, numpyArray, center_block_bool


def get_Ints(path, cycle_range=[0, None]):
    elem = path.split(os.path.sep)
    fov = elem[-1]
    ###use (os.path.sep).join() to get leading backslash
    cycles = [cycle for cycle in os.listdir((os.path.sep).join(elem[:-2]))
              if cycle.startswith('S')]
    cycles.sort()
    cycles = cycles[cycle_range[0]:cycle_range[1]]

    for cycle in cycles:
        try:
            ib = IntensityBin(os.path.join((os.path.sep).join(elem[:-2]), cycle, fov))
        except IOError:
            continue
        else:
            break

    ## return if unable to read any cycle (no data)
    try:
        ib
    except NameError:
        return
    else:
        data = np.zeros((ib.metrics.SpotNumber, ib.metrics.ChannelNumber, len(cycles)), dtype=np.float32)
        norm_paras = np.zeros((16, len(cycles)))
        background = np.zeros((100, 4, len(cycles)))

        for j, cycle in enumerate(cycles):
            # sys.stdout.write('\r   Loading {0:0.2f}%'.format( 100*(j+1)/float(len(cycles))) )

            int_path = os.path.join((os.path.sep).join(elem[:-2]), cycle, fov)
            if not os.path.exists(int_path):
                # print '{} failed\n'.format(cycle)
                continue
            ib = IntensityBin(int_path)
            data[:, :, j] = ib.metrics.IntsData.reshape(ib.metrics.ChannelNumber, -1).T
            norm_paras[:, j] = ib.metrics.NormalizationValue
            background[:, :, j] = ib.dump_block_backgrounds()
        return data, norm_paras, background

def main():
    return

if __name__=="__main__":
    main()