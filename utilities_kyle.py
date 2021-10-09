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
    cols = np.concatenate([np.arange(col) for col in dnb_cols])
    rows = np.concatenate([np.arange(row) for row in dnb_rows])
    grid = np.array(np.meshgrid(cols, rows))
    blocks = []
    for row in dnb_rows:
        for col in dnb_cols:
            blocks.append(grid[:, :row, :col].reshape(2, -1).T)
    dnb_idx = np.concatenate(blocks)
    return dnb_idx

def sequencerVersion(version):
    sd = _bundle(
        V2=_bundle(
            dnb_cols = np.array([87, 63, 85, 195, 217, 217, 195, 85, 63, 86]),
            dnb_rows = np.array([89, 51, 69, 141, 195, 195, 141, 69, 51, 88]),
            block_rows = 10,
            block_cols = 10,
            ),
        V40=_bundle(
            ),
        S=_bundle(
            dnb_cols = np.array([87, 63, 85, 195, 217, 217, 195, 85, 63, 86]),
            dnb_rows = np.array([74, 57, 132, 193, 192, 132, 102, 57, 73]),
            block_rows = 10,
            block_cols = 10,
            ),
        S2=_bundle(
            dnb_cols = np.array([72, 57, 102, 132, 192, 192, 132, 102, 57, 71]),
            dnb_rows = np.array([57, 45, 81, 105, 153, 153, 105, 81, 45, 56]),
            block_rows = 10,
            block_cols = 10,
            ),
        )
    sd[version].block_ids = calculateBlockDnbs(sd[version].dnb_cols, sd[version].dnb_rows)
    idx = calculateDnbRowCol(sd[version].dnb_cols, sd[version].dnb_rows)
    sd[version].row_ids = idx[:, 1]
    sd[version].col_ids = idx[:, 0]
    sd[version].dnb_num = idx.shape[0]
    return sd[version]
