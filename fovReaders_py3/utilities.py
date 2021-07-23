import numpy as np

class _bundle(dict):
    """ A class like dict which can be accessed
        use . operater.
        Usage:
        newDict = _bundle({})
        newDict["day"] = 1 ## set value as dict
        print newDict.day ## got 1, access as class

        newDict.month = 2 ## set value as class

        del newDict.month ## delete pair as class
    """
    def __init__(self, *args, **kw):
        super(_bundle, self).__init__(*args, **kw)
        return

    __getattr__ = dict.__getitem__
    __delattr__ = dict.__delitem__
    __setattr__ = dict.__setitem__

def calculateBlockDnbs(dnb_cols, dnb_rows):
    cols = np.repeat(dnb_cols, len(dnb_rows)).reshape(len(dnb_cols), len(dnb_rows)).T
    rows = np.repeat(dnb_rows, len(dnb_cols)).reshape(len(dnb_cols), len(dnb_rows))
    total = np.cumsum((cols * rows).ravel())
    block_ids = np.zeros(total[-1], dtype=np.uint8)
    b_start = 0
    for i, b_end in enumerate(total):
        block_ids[b_start:b_end] = i
        b_start = b_end
    return block_ids

def calculateDnbRowCol(dnb_cols, dnb_rows):
    # cols = np.concatenate([np.arange(col) for col in dnb_cols])
    # rows = np.concatenate([np.arange(row) for row in dnb_rows])
    # grid = np.array(np.meshgrid(cols, rows))
    # blocks = []
    # for row in dnb_rows:
    #     for col in dnb_cols:
    #         blocks.append(grid[:, :row, :col].reshape(2, -1).T)
    # dnb_idx = np.concatenate(blocks)

    blockX,blockY = np.meshgrid(dnb_cols,dnb_rows)
    blocks = (blockX*blockY).ravel()
    block_dnb_ct = np.cumsum(blocks)

    ##block rows and columns
    cols,rows = np.array([]),np.array([])
    dnb_col_row = np.zeros( (blocks.sum(),3), dtype=np.int )
    for b,(x,y) in enumerate(zip(blockX.ravel(),blockY.ravel())):
        col,row = np.meshgrid(np.arange(x),np.arange(y))
        dnb_col_row[block_dnb_ct[b]-blocks[b]:block_dnb_ct[b],0] = b
        dnb_col_row[block_dnb_ct[b]-blocks[b]:block_dnb_ct[b],1] = col.ravel()
        dnb_col_row[block_dnb_ct[b]-blocks[b]:block_dnb_ct[b],2] = row.ravel()

    return dnb_col_row

def sequencerVersion(version,track_width=3):
    sd = _bundle(
        #T20_DIPSEQT20
        FP21= _bundle( 
            dnb_cols = np.array([315, 405, 495, 630, 630, 495, 405, 315, 450]),
            dnb_rows = np.array([315, 405, 495, 630 ,630, 495, 405, 315, 450]),
            block_rows = 10,
            block_cols = 10,
            ),
        #T7_MIGSEQT7
        DP40 = _bundle(
            dnb_cols = np.array([130, 234, 286, 312, 312, 286, 234, 130, 208]),
            dnb_rows = np.array([130, 234, 286, 312, 312, 286, 234, 130, 208]),
            block_rows = 10,
            block_cols = 10,
            ),
        #T7_MGISEQT7
        DP84= _bundle(
            dnb_cols = np.array([112, 144, 208, 224, 224, 208, 144, 112, 160]),
            dnb_rows = np.array([112, 144, 208, 224, 224, 208, 144, 112, 160]),
            block_rows = 10,
            block_cols = 10,
            ),
        #T5_DIPSEQT5
        FP1= _bundle(
            dnb_cols = np.array([192, 240, 288, 312, 312, 288, 240, 192, 384]),
            dnb_rows = np.array([192, 240, 288, 312, 312, 288, 240, 192, 384]),
            block_rows = 10,
            block_cols = 10,
            ),
        #T10_DIPSEQT10
        FP2= _bundle(
            dnb_cols = np.array([240, 300, 330, 390, 390, 330, 300, 240, 420]),
            dnb_rows = np.array([240, 300, 330, 390, 390, 330, 300, 240, 420]),
            block_rows = 10,
            block_cols = 10,
            ),
        #V8_DIPSEQT1
        DP8= _bundle(
            dnb_cols = np.array([76, 133, 152, 228, 228, 152, 133, 76, 114]),
            dnb_rows = np.array([76, 133, 152, 228, 228, 152, 133, 76, 114]),
            block_rows = 10,
            block_cols = 10,
            ),
        #V2_MGISEQ2000
        V1= _bundle(
            dnb_cols = np.array([70, 112, 168, 196, 196, 168, 112, 70, 84]),
            dnb_rows = np.array([48, 64, 128, 176, 176, 128, 64, 48, 160]),
            block_rows = 10,
            block_cols = 10,
            ),
        #V2_MGISEQ2000
        V3= _bundle(
            dnb_cols = np.array([66, 88, 198, 220, 220, 198, 88, 66, 176]),
            dnb_rows = np.array([54, 72, 144, 198, 198, 144, 72, 54, 180]),
            block_rows = 10,
            block_cols = 10,
            ),
        #V2_MGISEQ2000
        P1= _bundle(
            dnb_cols = np.array([100, 140, 180, 280, 280, 180, 140, 100, 200]),
            dnb_rows = np.array([100, 140, 220, 280, 280, 220, 140, 100, 200]),
            block_rows = 10,
            block_cols = 10,
            ),
        #V02_MGISEQ200
        S1= _bundle(
            dnb_cols = np.array([60, 105, 135, 195, 195, 135, 105, 60, 150]),
            dnb_rows = np.array([48, 84, 108, 156, 156, 108, 84, 48, 120]),
            block_rows = 10,
            block_cols = 10,
            ),
        #V02_MGISEQ200
        S2= _bundle(
            dnb_cols = np.array([66, 88, 198, 220, 220, 198, 88, 66, 176]),
            dnb_rows = np.array([60, 105, 135, 195, 195, 135, 105, 60, 150]),
            block_rows = 10,
            block_cols = 10,
            ),
        #V01_BGISEQ50
        N1= _bundle(
            dnb_cols = np.array([76, 96, 110, 120, 120, 110, 96, 76, 176]),
            dnb_rows = np.array([76, 96, 110, 120, 120, 110, 96, 76, 176]),
            block_rows = 10,
            block_cols = 10,
            ),
        #V1_BGISEQ500
        CL1= _bundle(
            dnb_cols = np.array([65, 82, 95, 105, 105, 95, 82, 65, 160]),
            dnb_rows = np.array([55, 63, 77, 90, 90, 77, 63, 55, 150]),
            block_rows = 10,
            block_cols = 10,
            ),
        )
    
    #subtract track dnbs from block dimensions and separate edge block into 2
    sd[version].dnb_cols[:-1] -= track_width
    sd[version].dnb_rows[:-1] -= track_width
    sd[version].dnb_cols = np.insert(sd[version].dnb_cols,0,(sd[version].dnb_cols[-1]/2.)-np.floor(track_width/2.))
    sd[version].dnb_rows = np.insert(sd[version].dnb_rows,0,(sd[version].dnb_rows[-1]/2.)-np.floor(track_width/2.))
    sd[version].dnb_cols[-1] = (sd[version].dnb_cols[-1]/2.)-np.ceil(track_width/2.)
    sd[version].dnb_rows[-1] = (sd[version].dnb_rows[-1]/2.)-np.ceil(track_width/2.)

    # get blocks and rows/cols
    # sd[version].block_ids = calculateBlockDnbs(sd[version].dnb_cols, sd[version].dnb_rows)
    idx = calculateDnbRowCol(sd[version].dnb_cols, sd[version].dnb_rows)
    sd[version].block_ids = idx[:,0]
    sd[version].row_ids   = idx[:, 2]
    sd[version].col_ids   = idx[:, 1]
    sd[version].dnb_num   = idx.shape[0]
    return sd[version]
