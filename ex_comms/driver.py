# main driver code for inference via FPGA

import argparse

from pynq import Overlay
import numpy as np
from pynq import allocate
import time
from finn.util.data_packing import (
    finnpy_to_packed_bytearray,
    packed_bytearray_to_finnpy
)
from finn.core.datatype import DataType

class FINNAccelDriver():
    def __init__(self, N, bitfile):
        """Instantiate the FINN accelerator driver.
        Gets batchsize (N) as integer and path to bitfile as string."""
        self.N = N
        
        # input FINN DataType
        self.idt = DataType.UINT8
        
        # output FINN DataType
        self.odt = DataType.UINT32
        
        # input and output shapes
        self.ishape_normal = (N, 8, 8, 1)
        self.oshape_normal = (N, 3)
        self.ishape_folded = (N, 8, 8, 1, 1)
        self.oshape_folded = (N, 1, 3)
        self.ishape_packed = (N, 8, 8, 1, 1)   # datatype np.uint8
        self.oshape_packed = (N, 1, 12)  # datatype np.uint8
        # load bitfile and set up accelerator
        self.ol = Overlay(bitfile)

        self.dma = self.ol.axi_dma_0
        self.ctrl_regs = self.ol.resize_accel_0
        
        # neuron folding factor of output = iterations per sample
        self.itersPerSample = self.oshape_packed[-2]
        
        # AXI lite register offset for number of iterations
        # used by TLastMarker to signal end of transmission for AXI CDMA
        self.REG_OFFSET_NUM_ITERS = 0x10
        # set up TLastMarker with correct num. samples
        self.ctrl_regs.write(self.REG_OFFSET_NUM_ITERS, self.N*self.itersPerSample)

        # allocate a PYNQ buffer for the packed input and buffer
        self.ibuf_packed_device = allocate(shape=self.ishape_packed, dtype=np.uint8)
        self.obuf_packed_device = allocate(shape=self.oshape_packed, dtype=np.uint8)

    def fold_input(self, ibuf_normal):
        """Reshapes input in desired shape.
        Gets input data (ibuf_normal), checks if data is in expected normal shape.
        Returns folded input."""
        # ensure that shape is as expected
        assert ibuf_normal.shape == self.ishape_normal
        # convert to folded form
        ibuf_folded = ibuf_normal.reshape(self.ishape_folded)
        return ibuf_folded

    def pack_input(self, ibuf_folded):
        """Packs folded input and reverses both SIMD dim and endianness.
        Gets input data in folded shape and returns packed input data."""
        ibuf_packed = finnpy_to_packed_bytearray(
            ibuf_folded, self.idt, reverse_endian=True, reverse_inner=True
        )
        return ibuf_packed

    def unpack_output(self, obuf_packed):
        """Unpacks the packed output buffer from accelerator.
        Gets packed output and returns output data in folded shape."""
        obuf_folded = packed_bytearray_to_finnpy(
            obuf_packed, self.odt, self.oshape_folded, reverse_endian=True, reverse_inner=True
        )
        return obuf_folded

    def unfold_output(self, obuf_folded):
        """Unfolds output data to normal shape.
        Gets folded output data and returns output data in normal shape."""
        obuf_normal = obuf_folded.reshape(self.oshape_normal)
        return obuf_normal

    def copy_input_data_to_device(self, data):
        """Copies given input data to PYNQ buffer."""
        np.copyto(self.ibuf_packed_device, data)

    def execute(self):
        """Executes accelerator by setting up the DMA and
        waiting until all transfers complete. Uses only member variables and
        returns nothing."""
        dma = self.dma
        dma.sendchannel.transfer(self.ibuf_packed_device)
        dma.recvchannel.transfer(self.obuf_packed_device)
        dma.sendchannel.wait()
        dma.recvchannel.wait()
    
# compute the softmax values for each set of scores
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def print_output(output):
    logits = output.flatten()
    prob = softmax(logits)
    print(prob)

# expects an 8x8 input
def load_to_ibuf(input):
	return np.array(((((input[0][0], ), (input[0][1], ), (input[0][2], ), (input[0][3], ), (input[0][4], ), (input[0][5], ), (input[0][6], ), (input[0][7], )),
                         ((input[1][0], ), (input[1][1], ), (input[1][2], ), (input[1][3], ), (input[1][4], ), (input[1][5], ), (input[1][6], ), (input[1][7], )),
                         ((input[2][0], ), (input[2][1], ), (input[2][2], ), (input[2][3], ), (input[2][4], ), (input[2][5], ), (input[2][6], ), (input[2][7], )),
                         ((input[3][0], ), (input[3][1], ), (input[3][2], ), (input[3][3], ), (input[3][4], ), (input[3][5], ), (input[3][6], ), (input[3][7], )),
                         ((input[4][0], ), (input[4][1], ), (input[4][2], ), (input[4][3], ), (input[4][4], ), (input[4][5], ), (input[4][6], ), (input[4][7], )),
                         ((input[5][0], ), (input[5][1], ), (input[5][2], ), (input[5][3], ), (input[5][4], ), (input[5][5], ), (input[5][6], ), (input[5][7], )),
                         ((input[6][0], ), (input[6][1], ), (input[6][2], ), (input[6][3], ), (input[6][4], ), (input[6][5], ), (input[6][6], ), (input[6][7], )),
                         ((input[7][0], ), (input[7][1], ), (input[7][2], ), (input[7][3], ), (input[7][4], ), (input[7][5], ), (input[7][6], ), (input[7][7], )) ), ))

def parse_output_to_class(output):
    print("Received output of size", output.shape)
    max_value = -999
    predict_class = 0
    for i in range(len(output[0])):
        if output[0][i] > max_value:
            max_value = output[0][i]
            predict_class = i

    return predict_class

# test with running 2 bitfiles
def dual_inference(input):
    bitfile1 = "resizer.bit"
    bitfile2 = "resizer1.bit"
    print("making driver1")
    finnDriver1 = FINNAccelDriver(N, bitfile1)
    print("making driver2")
    finnDriver2 = FINNAccelDriver(N, bitfile2)
    print("executing")

    ibuf_normal = load_to_ibuf(input)
    ibuf_folded = finnDriver1.fold_input(ibuf_normal)
    ibuf_packed = finnDriver1.pack_input(ibuf_folded)

    finnDriver1.copy_input_data_to_device(ibuf_packed)
    finnDriver1.execute()
    print("executed 1")

    obuf_folded = finnDriver1.unpack_output(finnDriver1.obuf_packed_device)
    obuf_normal = finnDriver1.unfold_output(obuf_folded)
    print(obuf_normal)

    finnDriver2.copy_input_data_to_device(ibuf_packed)
    finnDriver2.execute()
    print("executed 2")

    obuf_folded = finnDriver2.unpack_output(finnDriver2.obuf_packed_device)
    obuf_normal = finnDriver2.unfold_output(obuf_folded)
    print(obuf_normal)

# expects a 8x8 numpy array, outputs a single integer (predicted class)
# you can test this by calling with --exec_mode test_dummy
def inference(input):
    #exec_mode = "execute"
    N = 1                    # batchsize
    bitfile = "resizer.bit"

    start = time.time()
    finnDriver = FINNAccelDriver(N, bitfile)
    print("Loaded bitfile in %.3f s" % (time.time() - start))

    ibuf_normal = load_to_ibuf(input)
    # pack and copy to the PYNQ buffer
    start = time.time() # keep track of timing

    #ibuf_normal = np.load(inputfile)
    ibuf_folded = finnDriver.fold_input(ibuf_normal)
    ibuf_packed = finnDriver.pack_input(ibuf_folded)
    finnDriver.copy_input_data_to_device(ibuf_packed)

    print("Loaded file into buffer in %.3f s." % (time.time() - start))

    # execute accelerator
    start = time.time()

    finnDriver.execute()

    print("Inference took %.8f s" % (time.time() - start))

    # if execution is selected unpack, unfold and save output to output npy file
    obuf_folded = finnDriver.unpack_output(finnDriver.obuf_packed_device)
    obuf_normal = finnDriver.unfold_output(obuf_folded)
    #np.save(outputfile, obuf_normal)

    print_output(obuf_normal)

    print("Prediction is ", parse_output_to_class(obuf_normal))
    return parse_output_to_class(obuf_normal)
    #print(obuf_normal)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Set exec mode, batchsize N, bitfile name, inputfile name and outputfile name')
    parser.add_argument('--exec_mode', help='Please select functional verification ("execute") or throughput test ("throughput_test")', default="execute")
    parser.add_argument('--batchsize', help='number of samples for inference', type=int, default=1)
    parser.add_argument('--bitfile', help='name of bitfile (i.e. "resizer.bit")', default="resizer.bit")
    parser.add_argument('--inputfile', help='name of input npy file (i.e. "input.npy")', default="input.npy")
    parser.add_argument('--outputfile', help='name of output npy file (i.e. "output.npy")', default="output.npy")
    
    # parse arguments
    args = parser.parse_args()
    exec_mode = args.exec_mode
    N = args.batchsize
    bitfile = args.bitfile
    inputfile = args.inputfile
    outputfile = args.outputfile

    # test with dummy input of 8x8
    if exec_mode == "test_dummy":

        input = [[1, 2, 3, 4, 5, 6, 7, 8],
        [1, 2, 3, 4, 5, 6, 7, 8],
        [1, 2, 3, 4, 5, 6, 7, 8],
        [1, 2, 3, 4, 5, 6, 7, 8],
        [1, 2, 3, 4, 5, 6, 7, 8],
        [1, 2, 3, 4, 5, 6, 7, 8],
        [1, 2, 3, 4, 5, 6, 7, 8],
        [1, 2, 3, 4, 5, 6, 7, 8]]

        dual_inference(input)

    # instantiate FINN accelerator driver and pass batchsize and bitfile
    finnDriver = FINNAccelDriver(N, bitfile)
      
    # for the remote execution the data from the input npy file has to be loaded,
    # packed and copied to the PYNQ buffer
    if exec_mode == "execute":
        # load desired input .npy file
        start = time.time() # keep track of timing
        
        ibuf_normal = np.load(inputfile)
        ibuf_folded = finnDriver.fold_input(ibuf_normal)
        ibuf_packed = finnDriver.pack_input(ibuf_folded)
        finnDriver.copy_input_data_to_device(ibuf_packed)
        
        print("Loaded file into buffer in %.3f s." % (time.time() - start))
        
    elif exec_mode != "throughput_test":
        raise Exception("Exec mode has to be set to remote_pynq or throughput_test")

    # for the throughput test the runtime of the network has to be measured
    if exec_mode == "throughput_test":
        # measure runtime of network
        start = time.time()
        # dictionary for results of throughput test
        res={}

    # execute accelerator
    start = time.time()
    finnDriver.execute()
    print("Inference took %.8f s" % (time.time() - start))

    # measure run time and fill dictionary with results of the throughput test
    if exec_mode == "throughput_test":
        end = time.time()
        runtime = end - start
        res["runtime[ms]"] = runtime*1000
        res["throughput[images/s]"] = N / runtime
        res["DRAM_in_bandwidth[Mb/s]"] = np.prod(finnDriver.ishape_packed)*0.000001 / runtime
        res["DRAM_out_bandwidth[Mb/s]"] = np.prod(finnDriver.oshape_packed)*0.000001 / runtime
        file = open("nw_metrics.txt", "w")
        file.write(str(res))
        file.close()

    # if execution is selected unpack, unfold and save output to output npy file
    elif exec_mode == "execute":
        obuf_folded = finnDriver.unpack_output(finnDriver.obuf_packed_device)
        obuf_normal = finnDriver.unfold_output(obuf_folded)
        np.save(outputfile, obuf_normal)
        print_output(obuf_normal)
        #print(obuf_normal)

