import utilities as util
import numpy as np
import os, sys
import ipdb


class FileHeader(object):
    '''

        FileHeader:
            FileTag:
                char SlideId[32];				//Slide ID
                unsigned char Type;				//Binary file type
                unsigned char Lane;				//Lane index start at 1
                unsigned char Row;				//Row ID
                unsigned char Col;				//Col ID
                uint32_t SpotNum;				//Spot number per fov
                unsigned char ChannelNum;		//Channel number, for cal file, this is 1
                unsigned char ElementSize;		//For F16 is 2
                unsigned short CycleNum;		//Cycle numnber
                unsigned char Recycle;			//1: open recycle, cycle number will be limitted
                unsigned char Reserves[19];
                std::string ToString() const
                {
                    std::string str;
                    CUtils::StringFormat(str, "%s.L%.3d.C%.3dR%.3d", SlideId, Lane, Col, Row);
                    return str;
                }


            ChunkEntry: Entries[EntryCount]:
                char Flag;					//Indicate current chunk data status, 1: has data, 0: unknown;
                float NorLow;				//for NotInt, here stores the normalization coefficient
                float NorUp;
                unsigned char Reserves[7];
                
    '''
    def __init__(self, filePath=None):
        self._clearFileTag()
        self._clearChunkEntries()
        self._AllocGranulatiry = 64 * 1024 #All chunk sizes are 64 kB for current version
        self.size = self._AllocGranulatiry
        self.debug = False
        if filePath is not None:
            if self.debug:
                print(filePath)
            self.load(filePath)
        return

    def _clearFileTag(self):
        self.tagNames = ["SlideId", "Type", "Lane", "Row", "Col", "SpotNum",
                         "ChannelNum", "ElementSize", "CycleNum", "Recycle"]
        self.fileTag = util._bundle(
            SlideId = "",
            Type = 0,
            Lane = 0,
            Row = 0,
            Col = 0,
            SpotNum = 0,
            ChannelNum = 0,
            ElementSize = 0,
            CycleNum = 0,
            Recycle = 0,
            fmtStr = "{SlideId}.L{Lane:03d}.C{Col:03d}.R{Row:03d}"
            )
        self.fileTagDtype = util._bundle(
            #SlideId = util._bundle(count=32, dtype=np.uint8),
            SlideId = util._bundle(count=512, dtype=np.uint8),
            Type    = util._bundle(count= 1, dtype=np.uint8),
            Lane    = util._bundle(count= 1, dtype=np.uint8),
            Row     = util._bundle(count= 1, dtype=np.uint8),
            Col     = util._bundle(count= 1, dtype=np.uint8),
            SpotNum = util._bundle(count= 1, dtype=np.uint32),
            ChannelNum = util._bundle(count=1, dtype=np.uint8),
            ElementSize = util._bundle(count=1, dtype=np.uint8),
            CycleNum = util._bundle(count=1, dtype=np.uint16),
            Recycle = util._bundle(count=1, dtype=np.bool)
            )
        return

    def _guessVersion(self):
        ft = self.fileTag
        #Slide ID requires two 00's between prefix and slide number
        prefix = ft.SlideId.split("00")[0]
        prefix = prefix.split("-")[0]
        prefix_v_dict = { "V3" : "V2", "V2" : "S2", "S2" : "S2"}
        self.version = prefix_v_dict.get(prefix, prefix)
        return

    def _clearChunkEntries(self):
        self.chunkEntries = util._bundle(chunk_size=0, num=0)
        return

    def _addChunkEntry(self, chunkId, flag, norUp=0, norLow=0):
        '''
            Load Meta Data information for specified chunk
        '''
        chunk = util._bundle(
                            Flag = bool(flag),
                            NorUp = norUp,
                            NorLow = norLow
                            )
        self.chunkEntries[chunkId] = chunk
        return

    def createAndViewHeader(self, path, size=0, exists=False):
        ft = self.fileTag
        if exists:
            with open(path, 'rb', self.size) as fh:
                for tag in self.tagNames:
                    data = np.fromfile(fh, **self.fileTagDtype[tag])
                    if tag == "ChannelNum" and self.debug:
                        print(tag, data)
                    if tag == "SlideId":
                        end = np.where(data == 0)[0][0]
                        ft.SlideId = "".join(data[:end].view('S1'))
                    else:
                        ft[tag] = data[0]
                self.calculateChunkMetrics()

                #fh.seek(64)#End of FileTag Header region)
                fh.seek(544)#End of FileTag Header region)
                #fh.seek(576)#End of FileTag Header region)
                for chunk in range(self.chunkEntries.num):
                    flag = np.fromfile(fh, count=1, dtype=np.bool)[0]
                    nor = np.fromfile(fh, count=2, dtype=np.float32)
                    #There is 7 leftover bytes delimiting each Chunk Entry
                    leftovers = np.fromfile(fh, count=7, dtype=np.uint8)
                    self._addChunkEntry(chunk, flag, nor[1], nor[0])
                
        else:
            #TODO Create a custom header
            pass
        self._guessVersion()
        sequencer_md = util.sequencerVersion(self.version)
        return

    def calculateChunkMetrics(self):
        ft = self.fileTag
        rawChunkSize = ft.SpotNum * ft.ElementSize
        appendSize =  (self._AllocGranulatiry - (rawChunkSize % self._AllocGranulatiry))
        self.chunkEntries.chunk_size = rawChunkSize + appendSize
        self.chunkEntries.num = ft.CycleNum * ft.ChannelNum.astype(np.uint16)
        return

    def load(self, filePath):
        self.createAndViewHeader(filePath, 0, True)
        self.calculateChunkMetrics()
        self.totalSize = self.size + self.chunkEntries.chunk_size * self.chunkEntries.num
        self._guessVersion()
        return

class FovReaderLite(object):

    def __init__(self, filePath=None, fovFileTag=None, bufferLimit=10000000):
        if filePath is None:
            #self.Create(FovFileTag)
            pass
        else:
            self._open(filePath)
        self.buffSize = min(bufferLimit, self.header.chunkEntries.chunk_size)
        return

    def Create(self, fovFileTag=None):
        #TODO:Add ability to add custom File Tags
        self.header = FileHeader()
        #self.filePath = #Create FilePath
        return

    def _open(self, filePath):
        self.header = FileHeader(filePath)
        self.filePath = filePath
        return

    ###########################################################################
    #Chunk Handling                                                           #
    ###########################################################################
    def _getChunkId(self, cycle, channel=0):
        chunkId = 0
        if self.header.fileTag.Recycle:
            chunkId = (cycle - 1) % self.cycleNumLimit * self.header.fileTag.ChannelNum + channel
        else:
            chunkId = (cycle - 1) * self.header.fileTag.ChannelNum + channel
        return chunkId

    def _setChunkFlag(self, chunkId, flag, norUp=0, norLow=0):
        self.header._addChunkEntry(chunkId, flag, norUp, norLow)
        return

    def _getChunkFlag(self, chunkId):
        '''
            Chunk Flags:
                Flag 		//Indicate current chunk data status, 1: has data, 0: unknown;
                NorLow		//for NotInt, here stores the normalization coefficient
                NorUp
        '''
        flags = self.header.chunkEntries[chunkId]
        return flags

    def _getChunkOffset(self, chunkId):
        '''
            Get the byte offset of the chunk in the file
        '''
        offset = (chunkId * self.header.chunkEntries.chunk_size) + self.header.size
        return offset

    def _chunkError(self, data, chunk_id):
        ft = self.header.fileTag
        raise ValueError("{} Invalid Chunk:{}".format(ft.fmtStr.format(**ft), chunk_id))
        return

    ###########################################################################
    #Kwarg Handling                                                           #
    ###########################################################################
    def _parse_cycle_parameter(self, cycles):
        start_cycle = 1
        if isinstance(cycles, (list, tuple, np.ndarray)):
            start_cycle, end_cycle = cycles[:2]
            end_cycle += 1
        elif not cycles:
            end_cycle = self.header.fileTag.CycleNum + 1
        else:
            end_cycle = cycles + 1
        return  start_cycle, end_cycle

    ###########################################################################
    #File Reading                                                             #
    ###########################################################################
    def _readData(self, cycles, dtype):
        '''
            Read Data from Data File
            Data has shape:(cycle x channel x dnbs)
        '''
        ft = self.header.fileTag
        start_cycle, end_cycle = self._parse_cycle_parameter(cycles)

        data = np.zeros((end_cycle - start_cycle, ft.ChannelNum, ft.SpotNum), dtype=dtype)
        with open(self.filePath, 'rb', self.buffSize) as self.fileObj:
            for cycle_i, cycle in enumerate(range(start_cycle, end_cycle)):
                self._readDataCycle(cycle, data[cycle_i], dtype=dtype)
        return data

    def _readDataCycle(self, cycle, data=None, dtype=np.uint8):
        ft = self.header.fileTag #ShortCut
        if data is None: data = np.zeros((ft.ChannelNum, ft.SpotNum), dtype=dtype)

        for channel in range(ft.ChannelNum):
            self._readDataChannel(cycle, channel, data[channel], dtype)
        return data

    def _readDataChannel(self, cycle, channel, data=None, dtype=np.uint8):

        if data is None: data = np.zeros(self.header.fileTag.SpotNum, dtype=dtype)

        idx = self._getChunkId(cycle, channel)
        chunk_flg = self._getChunkFlag(idx)
        #if chunk_flg.Flag:
        if True:
            offset = self._getChunkOffset(idx)
            self.fileObj.seek(offset, 0)
            data[:] = np.fromfile(self.fileObj, dtype=data.dtype, count=data.size)
        else:
            ipdb.set_trace()
            self._chunkError(data, idx)
        return data

    ###########################################################################
    #File Writing                                                             #
    ###########################################################################
    def _dumpData(self, fp, frmt_str, frmt_generator):
        for d_frmt in frmt_generator:
            fp.write(frmt_str.format(**d_frmt))
        return

    ###########################################################################
    #Data subsetting                                                          #
    ###########################################################################
    def _generateInnerBlockList(self):
        sd = util.sequencerVersion(self.header.version)
        blocks = np.arange(sd.block_rows * sd.block_cols).reshape(sd.block_rows, sd.block_cols)
        inner_mask = np.zeros((sd.block_rows, sd.block_cols), dtype=bool)
        inner_mask[1:-1, 1:-1] = True
        inner_blocks = blocks[inner_mask].ravel()
        return inner_blocks

    def _generateBlockFilter(self, blocks=[]):
        sequencer_md = util.sequencerVersion(self.header.version)
        block_ids = sequencer_md.block_ids
        block_filter = np.zeros(len(block_ids), dtype=bool)
        if len(blocks) == 0:
            block_filter[:] = True
        else:
            for block in blocks:
                block_filter[block_ids == block] = True
        return block_filter

    def _generate_dnb_pos(self):
        sequencer_md = util.sequencerVersion(self.header.version)
        return np.stack([sequencer_md.block_ids, sequencer_md.row_ids, sequencer_md.col_ids], axis=-1)

    def load_sequencer_version(self):
        sequencer_md = util.sequencerVersion(self.header.version)
        return sequencer_md