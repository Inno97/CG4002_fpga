# custom driver.py for 1D CNNs
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
from finn.core.modelwrapper import ModelWrapper
import os

import model as cnv
import dataset as ds

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def get_prediction(output_tensor):
    prediction_list = softmax(output_tensor.tolist())
    max_val = -1
    prediction = 0
    for i in range(len(prediction_list)):
        if prediction_list[i] > max_val:
            max_val = prediction_list[i]
            prediction = i

    return prediction

def get_prediction_numpy(output_np):
    prediction_list = softmax(output_np.tolist())
    max_val = -1
    prediction = 0
    for i in range(len(prediction_list)):
        if prediction_list[i] > max_val:
            max_val = prediction_list[i]
            prediction = i

    return prediction

def load_parent_model(path):
    parent_model = ModelWrapper(path + "_dataflow_parent_with_remote_bitfile_exec.onnx")
    return parent_model

# the 1D CNN, containing the brevitas software layers and the FINN hardware layers
class Cnv_Model():
    def __init__(self, bitfile):
        self.software_model = cnv.cnv_software_auto()
        self.hardware_model = cnv.cnv_hardware_auto()
        self.dataset = ds.Dataset()

        self.fpga_driver = FINNAccelDriver(1, bitfile)

        self.parent_model = load_parent_model(os.path.dirname(os.path.realpath(__file__)) + '/onnx/cnn_1d_3_classes_sample_dataset')
        # for hardware inference
        self.iname = "global_in"
        self.oname = "global_out"
        self.ishape = [1, 256]

        print("loaded model")

        #hardware acceleration will be done later, make sure that the model works for now
        #self.hardware_model = FINNAccelDriver(1, bitfile)

    # test on local inference from dataset without hardware acceleration
    def test_inference_software(self):
        test_input, test_output = self.dataset.get_next_train_data()

        software_output = self.software_model(test_input)
        print(software_output)

        hardware_output = self.hardware_model(software_output)
        print(hardware_output)

        for i in range(len(hardware_output)):
            print(softmax(hardware_output[i].tolist()), "prediction", get_prediction(hardware_output[i]), "target", test_output)

    # test on local inference from dataset with and without hardware acceleration
    def test_inference(self):
        test_input, test_output = self.dataset.get_next_train_data()

        software_output = self.software_model(test_input)
        hardware_output = self.hardware_model(software_output)

        for i in range(len(software_output)):
            ibuf_normal = software_output[i].reshape(self.ishape).detach().numpy()
            ibuf_normal = self.normalize_data(ibuf_normal)

            ibuf_folded = self.fpga_driver.fold_input(ibuf_normal)
            ibuf_packed = self.fpga_driver.pack_input(ibuf_folded)

            self.fpga_driver.copy_input_data_to_device(ibuf_packed)
            self.fpga_driver.execute()

            obuf_folded = self.fpga_driver.unpack_output(self.fpga_driver.obuf_packed_device)
            obuf_normal = self.fpga_driver.unfold_output(obuf_folded)

            accel_output = obuf_normal.astype(np.float32)
            accel_output /= 255
            print("raw output")
            print("hardware accelerated", accel_output[0])
            print("regular", hardware_output[i])

            print("softmax output, target:", test_output)
            print("hardware accelerated", softmax(accel_output[0].tolist()), "prediction", get_prediction_numpy(accel_output[0]))
            print("regular", softmax(hardware_output[i].tolist()), "prediction", get_prediction(hardware_output[i]))


    # normalize data for hardware inference
    # single bit width output, hence hard-coded normalization
    def normalize_data(self, input):
        norm_input = np.zeros((1, 256), dtype=np.float32)
        for i in range(len(input[0])):
            if input[0][i] == -1:
                norm_input[0][i] = 0
            else:
                norm_input[0][i] = 254
        return norm_input

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
        self.ishape_normal = (N, 256)
        self.oshape_normal = (N, 3)
        self.ishape_folded = (N, 8, 32)
        self.oshape_folded = (N, 1, 3)
        self.ishape_packed = (N, 8, 32)   # datatype np.uint8
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

# test inference locally from dataset
if __name__ == "__main__":
    bitfile = "resizer.bit"
    model = Cnv_Model(bitfile)
    model.test_inference()
    
    """
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

    # instantiate FINN accelerator driver and pass batchsize and bitfile
    finnDriver = FINNAccelDriver(N, bitfile)

    # for the remote execution the data from the input npy file has to be loaded,
    # packed and copied to the PYNQ buffer
    if exec_mode == "execute":
        # load desired input .npy file
        ibuf_normal = np.load(inputfile)
        ibuf_folded = finnDriver.fold_input(ibuf_normal)
        ibuf_packed = finnDriver.pack_input(ibuf_folded)
        finnDriver.copy_input_data_to_device(ibuf_packed)
    elif exec_mode != "throughput_test":
        raise Exception("Exec mode has to be set to remote_pynq or throughput_test")

    # for the throughput test the runtime of the network has to be measured
    if exec_mode == "throughput_test":
        # measure runtime of network
        start = time.time()
        # dictionary for results of throughput test
        res={}

    # execute accelerator
    finnDriver.execute()

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
    else:
        obuf_folded = finnDriver.unpack_output(finnDriver.obuf_packed_device)
        obuf_normal = finnDriver.unfold_output(obuf_folded)
        np.save(outputfile, obuf_normal)
    """
    
