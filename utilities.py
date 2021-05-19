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
    block_ids = np.zeros(total[-1] - 1, dtype=np.uint8)
    b_start = 0
    for i, b_end in enumerate(total):
        block_ids[b_start:b_end] = i
        b_start = b_end
    return block_ids

def sequencerVersion(version):
    sd = _bundle(
        V2=_bundle(
            dnb_cols = np.array([87, 63, 85, 195, 217, 217, 195, 85, 63, 86]),
            dnb_rows = np.array([89, 51, 69, 141, 195, 195, 141, 69, 51, 88]),
            ),
        V40=_bundle(
            ),
        )
    sd[version].block_ids = calculateBlockDnbs(sd[version].dnb_cols, sd[version].dnb_rows)
    return sd[version]
