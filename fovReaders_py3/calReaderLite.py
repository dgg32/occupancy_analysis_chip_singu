from .fovReaderLite import FovReaderLite
from collections import OrderedDict
from . import utilities as util
import itertools as itt
import numpy as np
import os, sys, csv
import gzip
# import ipdb

'''
    Goals:
        Add PE cal file reading,
        Add PE cycle cal file reading

        Add PE fq file writing
'''

class CalReaderLite(FovReaderLite):
    """
        API for reading cal files generated by the Basecaller Lite

        Attributes:
            calPaths:
            calCount:
            writeMode:
            bases:
            BaseIdx:

            qThreshold: Q score threshold for filter [r1, r2]
            esrCycleLimit: Maximum number of cycles below qThreshold before being filtered
            esrCycleNbr: Number of cycles used for ESR filter

        Public Methods:
            dumpFq: Write fq file to designated file object (default stdout)
            writeFqDnbs: Write fq file for the designated subset of DNBs (default all Dnbs)
            writeFqBlocks: Write fq for DNBs in the designated blocks (default all blocks)
            writeFqNoEdge: Write fq for only Non Edge blocks
            writeFq: Write fq using the designated filters
            write_fqsat: Write the fqstat file (allows for subsetting using writeFq filters)
    """

    def __init__(self, filePaths=None, bases="ACGTN"):
        if isinstance(filePaths, (list, tuple, np.ndarray)) and len(filePaths) > 1:
            super(CalReaderLite, self).__init__(filePaths[0])
            self.calCount = len(filePaths)
        else:
            super(CalReaderLite, self).__init__(filePaths)
            self.calCount = 1
        self.calPaths = filePaths
        self.writeMode = 'w'

        self._encodeB2S6 = lambda base, score: (base << 6 | score) & 0xFF
        self.bases = list(bases)
        self.max_score = 42
        self.BaseIdx = dict([(base, idx) for idx, base in enumerate(self.bases)])

        #ESR Filter Parameters
        self.qThreshold = [20, 15]
        self.esrCycleLimit = 2
        self.esrCycleNbr = 20
        return

    def _cycles_is_pe(self, cycles):
        '''
            Check if the cal file is PE based on the cycles parameter
        '''
        if isinstance(cycles, (list, tuple, np.ndarray)):
            cycles = np.array(cycles)
            is_pe = ((cycles.ndim == 2) and (cycles.shape[1] == 2))
            return is_pe
        else:
            return False

    ###########################################################################
    #Base/Score data type conversion                                          #
    ###########################################################################
    def int_to_ascii(self, base_i, score_i):
        '''
            Convert numpy array of base and score values to the ascii string equivalents
        '''
        base_a = np.array(self.bases)[base_i]
        score_a = (score_i + ord('!')).view('S1').astype('U1')
        return base_a, score_a


    def ascii_to_int(self, base_a, score_a):
        '''
            Convert numpy array of base and score strings to the equivalent int value
        '''
        base_a = base_a.view('S1').reshape(base_a.shape[0], -1)
        base_i = np.zeros(base_a.shape, dtype=np.uint8)
        for ch_i, ch in enumerate(self.bases):
            base_i[base_a == ch] = ch_i
        score_i = score_a.view(np.uint8).reshape(score_a.shape[0], -1) - ord('!')
        return base_i, score_i

    ###########################################################################
    #Encoding and Decoding Cal Bytes                                          #
    ###########################################################################
    def _decodeB2S6(self, cal):
        parts = util._bundle(score = 0x3F & cal)
        base = cal >> 6
        # base[parts.score == 0] = 4
        parts.base = base
        return parts

    def _decode(self, cal):
        parts = self._decodeB2S6(cal)
        #Transpose parts to (dnb x cycle shape)
        return parts.score.T, parts.base.T

    def _encode(self, base, score):
        cal = self._encodeB2S6(base, score)
        return cal

    def _chunkError(self, data, chunk_id):
        data[:] = self._encode(self.BaseIdx["N"], 0)
        return

    ###########################################################################
    #Read and Interpret Cal Data from Cal File                                #
    ###########################################################################
    def _decodeCal(self, cycles=None, ascii_fmt=False):
        cal = self.readCal(cycles)
        score, base = self._decode(cal)
        #print ("(((((((((((((((((((((((((((((((((((((((((((((( ", base, score)
        #base[np.where(score == "!")] = "N"
        if ascii_fmt:
            base, score = self.int_to_ascii(base, score)
        
        #print ("(((((((((((((((((((((((((((((((((((((((((((((( ", base, score)
            
        return base, score

    def readCal(self, cycles=None, cal_idx=0):
        '''
            Read raw byte data from cal file (bytes are encoded with the base and score value)

            Parameters:

                cycles: Argument types:
                        None: All cycles
                  [int, int]: Start and end of cycle range (1-indexed)

            Return:
                data: Raw encoded cal data
        '''
        data = self._readData(cycles, np.uint8)[:, 0, :]
        return data

    def _generateEsrMask(self, score, strand, q_threshold=20, cycle_limit=2, cycle_nbr=20):
        dropped_cycles = np.all(score == 0, axis=0)
        bad_cycle_count = (score[:, :cycle_nbr] < q_threshold).sum(axis=1)
        esr_mask = (bad_cycle_count <= cycle_limit)
        return esr_mask

    def _generateDnbFilter(self, score_i, strand, dnb_subset, esr_filter, qscore_filter):
        #Generate ESR Filter Mask
        if esr_filter:
            strand_idx = int(str(strand) not in ["None", "1"])
            esr_mask = self._generateEsrMask(score_i, strand, self.qThreshold[strand_idx],
                                            self.esrCycleLimit, self.esrCycleNbr)
        else:
            esr_mask = np.ones(score_i.shape[0], dtype=bool)

        #Generate QScore Filter mask
        if qscore_filter is not None:
            esr_mask &= (~np.any(score_i <= qscore_filter ,axis=1))

        #Incorporate DNB Subset
        if type(dnb_subset) is np.ndarray:
            dnbs = np.where(dnb_subset & esr_mask)[0]
        elif type(dnb_subset) is list:
            dnbs = np.array(dnb_subset) - 1
            dnbs = [dnb for dnb in dnbs if esr_mask[dnb]]
        else: #DNB subset is a boolean mask
            dnbs = np.arange(len(esr_mask))[esr_mask]
        return set(dnbs)

    def memFq(self, cycles=None, strand=None, dnb_subset=False, esr_filter=False, qscore_filter=None):
        base, score = self._decodeCal(cycles, False)
        
        
        dnbs = self._generateDnbFilter(score, strand, dnb_subset, esr_filter, qscore_filter)

        base, score = self.int_to_ascii(base, score)
        #print ("!!!!!!!!!!!!!!!!!base", base, type(base))
        #print ("score", score, type(score))
        base[np.where(score == "!")] = "N"

        base  = np.ascontiguousarray(base).view("U{}".format(base.shape[1]))
        score = np.ascontiguousarray(score).view("U{}".format(score.shape[1]))

        return base, score, dnbs

    ###########################################################################
    #Cal to Fastq IO                                                          #
    ###########################################################################
    def _handleFq(self, fq_path, mode, function, f_args=(), f_kwargs={}):
        if fq_path.endswith(".gz"):
            file_handeler = gzip.open
            mode += 'b'
        else:
            file_handeler = open
        with file_handeler(fq_path, mode=mode) as fp:
            ret = function(fp, *f_args, **f_kwargs)
        return ret
        
    def _writeFq(self, fq_path, base, score, strand, dnbs, mode='w'):
        self._handleFq(fq_path, mode, self._dumpFq, f_args=(base, score, strand, dnbs))
        return

    def _dumpFq(self, fp, base, score, strand, dnbs):
        strand = "" if strand is None else "/{}".format(strand)
        fqfmtstr = "\n".join(["@{SlideId}L{Lane:d}C{Col:03d}R{Row:03d}_{DNB:06d}{strand}",
                              "{Bases}",
                              "+",
                              "{Scores}\n"])
        #Force dnbs to iterate in order
        dnb_list = list(dnbs)
        dnb_limit = max(dnb_list) + 1
        dnbs = np.zeros((dnb_limit), dtype=bool)
        dnbs[dnb_list] = True
        dnbs = np.where(dnbs)[0]

        fqfmtgen = (util._bundle(DNB=dnb_i + 1, strand=strand,
                            Bases=base[dnb_i,0], Scores=score[dnb_i,0],
                            **self.header.fileTag) for dnb_i in dnbs)
        self._dumpData(fp, fqfmtstr, fqfmtgen)
        return

    def dumpFq(self, fp=sys.stdout, cycles=None, strand=None, dnb_subset=False,
                    esr_filter=False, qscore_filter=None):
        base, score, dnbs = self.memFq(cycles, strand, dnb_subset, esr_filter, qscore_filter)
        self._dumpFq(fp, base, score, strand, dnbs)
        return 

    def writeFqSE(self, fq_path, cycles=None, strand=None, dnb_subset=False,
                    esr_filter=False, qscore_filter=None):
        base, score, dnbs = self.memFq(cycles, strand, dnb_subset, esr_filter, qscore_filter)
        self._writeFq(fq_path, base, score, strand, dnbs, mode=self.writeMode)
        return

    def writeFqPE(self, fq_path, cycles=None, dnb_subset=False, esr_filter=False, qscore_filter=None):
        if (self.calCount == 2):
            calPaths = self.calPaths
            #Set up cycles to be the same for each run
            if not self._cycles_is_pe(cycles):
                cycles = [cycles, cycles]
        else:
            calPaths = [self.calPaths] * 2
            
        bases = []
        scores = []
        dnbs = []
        fns = []
        #Load data for each read
        for strand, cal_fn in enumerate(calPaths, 1):
            self._open(cal_fn)
            base, score, dnb_set = self.memFq(cycles[strand - 1], strand, dnb_subset, esr_filter, qscore_filter)
            bases.append(base)
            scores.append(score)
            dnbs.append(dnb_set)

        #Write fq file using combined filter of each read
        dnb_set = dnbs[0].intersection(dnbs[1])
        for strand, cal_fn in enumerate(calPaths, 1):
            idx = strand - 1
            self._open(cal_fn)
            pe_name = fq_path.replace('.fq', "_{}.fq".format(strand))
            self._writeFq(pe_name, bases[idx], scores[idx], strand, dnb_set, mode=self.writeMode)
        return

    ###########################################################################
    #Main Fq writing functions (allows for different subsetting)              #
    ###########################################################################
    def writeFqDnbs(self, fq_fn, cycles=None, strand=None, dnb_subset=False,
                            esr_filter=False, qscore_filter=None):
        '''
            Write fastq for specified DNBs

            Parameters:
                fq_fn: Save path for the fq file (supports gz compression)
                 cycles:
                        int: Number of cycles
               [start, end]: Cycle range written (inclusive for both start and end)
         [[s1,e1], [s2,e2]]: Two cycle ranges corresponding to read1 and read2 of a PE run

                 strand: Which strand is being written (None:SE, 1:PE Read1, 2:PE Read2)
             dnb_subset: list or bool mask of DNBs that will be written to fq file.
             esr_filter: Apply esr filter (using instance filter parameters)
          qscore_filter: int: Filter out DNBs that have a cycle with a lower qscore
        '''
        if (self.calCount == 2) or self._cycles_is_pe(cycles):
            self.writeFqPE(fq_fn, cycles, dnb_subset, esr_filter, qscore_filter)
        else:
            self.writeFqSE(fq_fn, cycles, strand, dnb_subset, esr_filter, qscore_filter)
        return

    def writeFqBlocks(self, fq_fn, cycles=None, strand=None, blocks=None,
                        esr_filter=False, qscore_filter=None):
        '''
            Write Fq files for specific blocks

            Parameters:
                fq_fn: Save path for the fq file (supports gz compression)
                 cycles:
                        int: Number of cycles
               [start, end]: Cycle range written (inclusive for both start and end)
         [[s1,e1], [s2,e2]]: Two cycle ranges corresponding to read1 and read2 of a PE run

                 strand: Which strand is being written (None:SE, 1:PE Read1, 2:PE Read2)
                 blocks: list of blocks that will be written for.
             esr_filter: Apply esr filter (using instance filter parameters)
          qscore_filter: int: Filter out DNBs that have a cycle with a lower qscore
        '''
        dnb_subset = self._generateBlockFilter(blocks)
        self.writeFqDnbs(fq_fn, cycles, strand, dnb_subset, esr_filter, qscore_filter)
        return

    def writeFqNoEdge(self, fq_fn, cycles=None, strand=None,
                        esr_filter=False, qscore_filter=None):
        '''
            Write Fq file with the edge blocks filtered out

            Parameters:
                fq_fn: Save path for the fq file (supports gz compression)
                 cycles:
                        int: Number of cycles
               [start, end]: Cycle range written (inclusive for both start and end)
         [[s1,e1], [s2,e2]]: Two cycle ranges corresponding to read1 and read2 of a PE run

                 strand: Which strand is being written (None:SE, 1:PE Read1, 2:PE Read2)
             esr_filter: Apply esr filter (using instance filter parameters)
          qscore_filter: int: Filter out DNBs that have a cycle with a lower qscore
        '''
        inner_blocks = self._generateInnerBlockList()
        self.writeFqBlocks(fq_fn, cycles, strand, inner_blocks, esr_filter, qscore_filter)
        return

    def writeFq(self, fq_fn, cycles=None, strand=None, esr_filter=False,
                    eb_filter=False, qscore_filter=None):
        '''
            Write Fq file using the designated filters

            Parameters:
                fq_fn: Save path for the fq file (supports gz compression)
                 cycles:
                        int: Number of cycles
               [start, end]: Cycle range written (inclusive for both start and end)
         [[s1,e1], [s2,e2]]: Two cycle ranges corresponding to read1 and read2 of a PE run

                 strand: Which strand is being written (None:SE, 1:PE Read1, 2:PE Read2)
             esr_filter: Apply esr filter (using instance filter parameters)
              eb_filter: Apply edge block filter
          qscore_filter: int: Filter out DNBs that have a cycle with a lower qscore
        '''
        if eb_filter:
            self.writeFqNoEdge(fq_fn, cycles, strand, esr_filter, qscore_filter)
        else:
            self.writeFqDnbs(fq_fn, cycles, strand, False, esr_filter, qscore_filter)
        return

    ###########################################################################
    #Main Fq reading functions (allows for different subsetting)              #
    ###########################################################################
    def _read_fq(self, fp, line_per_seq):
        def grouper(iterable, n, fillvalue=None):
            args = [iter(iterable)] * n
            return (itt.izip_longest(*args, fillvalue=fillvalue))

        base_a = []
        score_a = []
        has_scores = line_per_seq > 2
        for i, lines in enumerate(grouper(fp, line_per_seq)):
            base_a.append(lines[1].strip())
            if has_scores:
                score_a.append(lines[3].rstrip())
        return np.array(base_a), np.array(score_a)

    def read_fq(self, fq_path, int_fmt=True):
        base_a, score_a = self._handleFq(fq_path, 'r', self._read_fq, f_args=(4,))
        if int_fmt:
            base_a, score_a = self.ascii_to_int(base_a, score_a)
        return base_a, score_a

    ###########################################################################
    #Fq Stat writing functions (allows for different subsetting)              #
    ###########################################################################
    def _count_fq(self, base_i, score_i):
        def count_categories(arr, category_count):
            _, cycles = arr.shape
            hist = np.zeros((cycles, category_count))
            for cycle in range(cycles):
                idx, count = np.unique(arr[:, cycle], return_counts=True)
                if hasattr(idx, "mask"):
                    count = count[~idx.mask]
                    idx = idx[~idx.mask]
                hist[cycle, idx] = count
            return hist

        base_hist = count_categories(base_i, len(self.bases))
        score_hist = count_categories(score_i, self.max_score)
        return base_hist, score_hist

    def _calculate_estimated_error(self, score_hist, dnb_count):
        est_err_count = ((1.0 / 10**(np.arange(self.max_score) * 0.1)) * score_hist).sum(axis=1)
        est_err_p = np.divide(100. * est_err_count, dnb_count, where=(dnb_count != 0))
        est_err_p = np.around(est_err_p, 2)
        est_err_p[(est_err_p > 100.) | (est_err_p < 0.)] = 0
        print(type(score_hist), type(dnb_count))
        if (np.any(est_err_p > 100.) or np.any(est_err_p < 0.)):
            print(est_err_p.dtype, est_err_p)
            # ipdb.set_trace()
        return est_err_p

    def _write_fqstat(self, base_i, score_i):
        dnb_count, cycle_count = base_i.shape
        base_count = base_i.size
        base_hist, score_hist = self._count_fq(base_i, score_i)
        est_err_p = self._calculate_estimated_error(score_hist, dnb_count)

        m_bases = [np.ma.array(base_i, mask=(base_i != ch)) for ch in range(len(self.bases))]
        ch_breakdowns = [self._count_fq(mb, np.ma.array(score_i, mask=mb.mask)) for mb in m_bases]
        base_est_err = np.array([self._calculate_estimated_error(cb[1], cb[0][:, i])
                            for i, cb in enumerate(ch_breakdowns)]).T

        if (np.any(base_est_err > 100.) or np.any(base_est_err < 0.)):
            #This seems to be an error that pops up sometimes? need to repeat
            print(base_est_err[:, 4], base_est_err.dtype)
            # ipdb.set_trace()
        n_count = base_hist[:, -1].sum().astype(int)

        fq_stat = OrderedDict()
        fq_stat["#PhredQual"] = ord('!')
        fq_stat["#ReadNum"] = dnb_count
        fq_stat["#row_readLen"] = cycle_count
        fq_stat["#col"] = 48 #Not sure what this is supposed to stand for, referebce calReader.py
        fq_stat["#BaseNum"] = base_count
        fq_stat["#N_Count"] = (n_count, np.around(100. * n_count / max(base_count, 1), 2))
        fq_stat["#GC%"] = np.around(100. * base_hist[:, [1, 2]].sum() /
                                       max(base_count - n_count, 1), 2)
        fq_stat["#>Q10%"] = np.around(100. * score_hist[:, 10:].sum() / max(base_count, 1), 2)
        fq_stat["#>Q20%"] = np.around(100. * score_hist[:, 20:].sum() / max(base_count, 1), 2)
        fq_stat["#>Q30%"] = np.around(100. * score_hist[:, 30:].sum() / max(base_count, 1), 2)
        fq_stat["#>Q40%"] = np.around(100. * score_hist[:, 40:].sum() / max(base_count, 1), 2)
        fq_stat["#EstErr%"] = np.around(est_err_p.mean(), 2)
        return fq_stat, base_hist, score_hist, est_err_p, base_est_err

    def write_fqstat(self, fqstat_fn, fq_path=None, cycles=None,
                        strand=None, dnb_subset=False, esr_filter=False,
                        qscore_filter=None, delimiter='\t'):
        if fq_path is None:
            name = self.calPaths
            base_i, score_i = self._decodeCal()
        else:
            name = fq_path
            base_i, score_i = self.read_fq(fq_path, int_fmt=True)
        sc, ec = self._parse_cycle_parameter(cycles)
        cycle_slice = slice(sc - 1, min(ec, base_i.shape[1]) - 1)
        dnbs = list(self._generateDnbFilter(score_i[:, cycle_slice], None,
                                dnb_subset, esr_filter, qscore_filter))
        base_i = base_i[dnbs, cycle_slice]
        score_i = score_i[dnbs, cycle_slice]
        fq_stat, base_hist, score_hist, est_err_p, base_est_err = self._write_fqstat(base_i, score_i)
        with open(fqstat_fn, 'w') as fp:
            stat_csv = csv.writer(fp, delimiter=delimiter, lineterminator='\n')
            stat_csv.writerow(["#Name", name])
            for k, v in fq_stat.iteritems():
                stat_csv.writerow([k] + list(v if isinstance(v, tuple) else [v]))
            stat_csv.writerow(['#Pos'] + self.bases + list(range(self.max_score)) +
                              [b + "_Err%" for b in self.bases] + ["Err%"])
            for i, cycle in enumerate(range(cycle_slice.start, cycle_slice.stop)):
                stat_csv.writerow([cycle + 1] + list(base_hist[i].astype(int)) +
                        list(score_hist[i].astype(int)) +
                        list(np.around(base_est_err[i], 2)) +
                        [np.around(est_err_p[i], 2)])
        return

    ###########################################################################
    #TODO:Addiontal functions being worked on                                 #
    ###########################################################################

    def writeCal(self):
        #TODO: Low Priority
        return

    def fq_to_cal(self, fq_path):
        #TODO: Low Priority
        return

if __name__ == "__main__":
    if len(sys.argv) == 3:
        cal = CalReader(sys.argv[1])
        cal.writeFq(sys.argv[2])
