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
import time
import torch

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

        #self.parent_model = load_parent_model(os.path.dirname(os.path.realpath(__file__)) + '/onnx/cnn_1d_3_classes_sample_dataset')
        # for hardware inference
        self.iname = "global_in"
        self.oname = "global_out"
        self.ishape = [1, 256]

        print("loaded model")

    # expects numpy ndarray, in the format (1, num_channels, input_size)
    def inference(self, data):
        input_tensor = torch.from_numpy(data)

        # software inference (Conv1D)
        time_taken_software_output = time.time()
        software_output = self.software_model(input_tensor)
        time_taken_software_output = time.time() - time_taken_software_output

        # hardware accelerated (FC)
        ibuf_normal = software_output.reshape(self.ishape).detach().numpy()
        ibuf_normal = self.normalize_data(ibuf_normal)

        ibuf_folded = self.fpga_driver.fold_input(ibuf_normal)
        ibuf_packed = self.fpga_driver.pack_input(ibuf_folded)

        time_taken_accel_output = time.time()
        self.fpga_driver.copy_input_data_to_device(ibuf_packed)
        self.fpga_driver.execute()

        obuf_folded = self.fpga_driver.unpack_output(self.fpga_driver.obuf_packed_device)
        time_taken_accel_output = time.time() - time_taken_accel_output
        obuf_normal = self.fpga_driver.unfold_output(obuf_folded)

        accel_output = obuf_normal.astype(np.float32)
        accel_output /= 255
        print("hardware accelerated: {}, prediction: {}, accelerated inference took {}".format(softmax(accel_output[0].tolist()), get_prediction_numpy(accel_output[0]),
		                                                                                  time_taken_software_output + time_taken_accel_output))
        print("software layers took {}s, hardware layers took {}s".format(time_taken_software_output, time_taken_accel_output))
        return softmax(accel_output[0].tolist()), get_prediction_numpy(accel_output[0])

    # test on local inference from dataset without hardware acceleration
    def test_inference_software(self):
        test_input, test_output = self.dataset.get_next_train_data()

        software_output = self.software_model(test_input)
        print(software_output)

        hardware_output = self.hardware_model(software_output)
        print(hardware_output)

        for i in range(len(hardware_output)):
            print(softmax(hardware_output[i].tolist()), "prediction", get_prediction(hardware_output[i]), "target", test_output[i])

    # test on local inference from dataset with and without hardware acceleration
    def test_inference(self):
        test_input, test_output = self.dataset.get_next_train_data()

        time_taken_software_output = time.time()
        software_output = self.software_model(test_input)
        time_taken_software_output = time.time() - time_taken_software_output

        time_taken_hardware_output = time.time()
        hardware_output = self.hardware_model(software_output)
        time_taken_hardware_output = time.time() - time_taken_hardware_output

        print(software_output)
        print(type(software_output[0][0].item()))

        for i in range(len(software_output)):
            ibuf_normal = software_output[i].reshape(self.ishape).detach().numpy()
            ibuf_normal = self.normalize_data(ibuf_normal)

            ibuf_folded = self.fpga_driver.fold_input(ibuf_normal)
            ibuf_packed = self.fpga_driver.pack_input(ibuf_folded)

            time_taken_accel_output = time.time()
            self.fpga_driver.copy_input_data_to_device(ibuf_packed)
            self.fpga_driver.execute()

            obuf_folded = self.fpga_driver.unpack_output(self.fpga_driver.obuf_packed_device)
            time_taken_accel_output = time.time() - time_taken_accel_output
            obuf_normal = self.fpga_driver.unfold_output(obuf_folded)

            accel_output = obuf_normal.astype(np.float32)
            accel_output /= 255
            print("raw output")
            print("hardware accelerated", accel_output[0])
            print("regular", hardware_output[i])

            print("softmax output, target:", test_output[i])
            print("hardware accelerated", softmax(accel_output[0].tolist()), "prediction", get_prediction_numpy(accel_output[0]))
            print("regular", softmax(hardware_output[i].tolist()), "prediction", get_prediction(hardware_output[i]))
            print("regular inference took %.3f, accelerated inference took %.3f" % (((time_taken_software_output / 2) + (time_taken_hardware_output / 2)),
                                                                                    ((time_taken_software_output / 2) + (time_taken_accel_output))))

    # test and benchmark hardware acceleration
    def benchmark_inference(self, verbose = False):
        num_inputs = 0
        num_outputs_correct_software = 0
        num_outputs_correct_hardware = 0
        num_outputs_matched = 0

        for i, data in enumerate(self.dataset.get_train_loader()):
            (test_input, test_output) = data

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

                if verbose:
                    print("target: {}, hardware: {} / {}, regular: {} / {}".format(int(test_output[i]), softmax(accel_output[0].tolist()), get_prediction_numpy(accel_output[0]),
                                                                               softmax(hardware_output[i].tolist()), get_prediction(hardware_output[i])))

                num_inputs += 1
                if get_prediction_numpy(accel_output[0]) == get_prediction(hardware_output[i]):
                    num_outputs_matched += 1

                if int(test_output[i]) == get_prediction_numpy(accel_output[0]):
                    num_outputs_correct_hardware += 1

                if int(test_output[i]) == get_prediction_numpy(hardware_output[i]):
                    num_outputs_correct_software += 1

                if num_inputs % 50 == 0:
                    print("total inputs: %d | accuracy - hardware: %.3f regular %.3f | num outputs matched in both models: %.3f" %
                                                        (num_inputs, (num_outputs_correct_hardware * 100 / num_inputs),
                                                        (num_outputs_correct_software * 100 / num_inputs), (num_outputs_matched * 100 / num_inputs)))

        for i, data in enumerate(self.dataset.get_test_loader()):
            (test_input, test_output) = data

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

                if verbose:
                    print("target: {}, hardware: {} / {}, regular: {} / {}".format(int(test_output[i]), softmax(accel_output[0].tolist()), get_prediction_numpy(accel_output[0]),
                                                                               softmax(hardware_output[i].tolist()), get_prediction(hardware_output[i])))

                num_inputs += 1
                if get_prediction_numpy(accel_output[0]) == get_prediction(hardware_output[i]):
                    num_outputs_matched += 1

                if int(test_output[i]) == get_prediction_numpy(accel_output[0]):
                    num_outputs_correct_hardware += 1

                if int(test_output[i]) == get_prediction_numpy(hardware_output[i]):
                    num_outputs_correct_software += 1

                if num_inputs % 50 == 0:
                    print("total inputs: %d | accuracy - hardware: %.3f regular %.3f | num outputs matched in both models: %.3f" %
                                                        (num_inputs, (num_outputs_correct_hardware * 100 / num_inputs),
                                                        (num_outputs_correct_software * 100 / num_inputs), (num_outputs_matched * 100 / num_inputs)))

        print("total inputs:", num_inputs)
        print("accuracy - hardware: %.3f regular %.3f" % ((num_outputs_correct_hardware * 100 / num_inputs),
                                                        (num_outputs_correct_software * 100 / num_inputs)))
        print("num of outputs matched in both models: %.3f", ((num_outputs_matched * 100 / num_inputs)))

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
        self.oshape_normal = (N, 9)
        self.ishape_folded = (N, 8, 32)
        self.oshape_folded = (N, 1, 9)
        self.ishape_packed = (N, 8, 32)   # datatype np.uint8
        self.oshape_packed = (N, 1, 36)  # datatype np.uint8
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

    # run inference on random input from dataset
    #model.test_inference()
    print(sum(p.numel() for p in model.software_model.parameters()))
    # test inference function call with empty np array
    data = np.zeros((1, 5, 56), dtype=np.float32)
    values, prediction = model.inference(data)
    values, prediction = model.inference(data)
    values, prediction = model.inference(data)
    #print("values: {}, prediction: {}".format(values, prediction))

    # benchmark
    #model.benchmark_inference(verbose=True)
