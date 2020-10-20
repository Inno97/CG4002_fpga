import re
import driver
import os
import sys
import random
import time
from sender import *
import subprocess

import socket
import threading

import base64
import numpy as np
from tkinter import Label, Tk
import pandas as pd
from Crypto.Cipher import AES
from Crypto import Random

import struct #converting bytes

MESSAGE_SIZE = 3 #timestamp|RTT|message
#added by alexis oct10
input = [[1, 2, 3, 4, 5, 6, 7, 8],
        [1, 2, 3, 4, 5, 6, 7, 8],
        [1, 2, 3, 4, 5, 6, 7, 8],
        [1, 2, 3, 4, 5, 6, 7, 8],
        [1, 2, 3, 4, 5, 6, 7, 8],
        [1, 2, 3, 4, 5, 6, 7, 8],
        [1, 2, 3, 4, 5, 6, 7, 8],
        [1, 2, 3, 4, 5, 6, 7, 8]]

actions = ['zigzag', 'rocket', 'hair']

class Server(threading.Thread):
    def __init__(self, ip_addr, port_num, dancer_num):
        super(Server, self).__init__()

        self.timeout = 600
        self.has_no_response = False
        self.connection = None
        self.timer = None
        self.logout = False
        self.RTT = 0
        self.clock_offset = 0

        self.dancer = dancer_num

        # Create a TCP/IP socket and bind to port
        self.shutdown = threading.Event()
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_address = (ip_addr, port_num)

        #print('Starting up on %s port %s' % server_address, file=sys.stderr)
        self.socket.bind(server_address)

        # Listen for incoming connections
        self.socket.listen(1)
        self.client_address, self.secret_key = self.setup_connection(self.dancer) 

    def decrypt_message(self, cipher_text):
        decoded_message = base64.b64decode(cipher_text)
        iv = decoded_message[:16]
        secret_key = bytes(str(self.secret_key), encoding="utf8") 
        cipher = AES.new(secret_key, AES.MODE_CBC, iv)
        decrypted_message = cipher.decrypt(decoded_message[16:]).strip()
        decrypted_message = decrypted_message.decode('utf8')
        messages = decrypted_message.split('|')
        clock_offset, rtt, message = messages[:MESSAGE_SIZE]
        clock_offset = self.recv_time - float(clock_offset) - (self.RTT*0.5)

        return {
            'clock_offset': clock_offset, 'rtt': float(rtt), 'message': message
        }

    def sendReply(self):
        #send timestamp as t3-t2
        timestamped_msg = str(float(time.time() - self.recv_time)) + '|reply'
        msg_bytes = str.encode(timestamped_msg)
        length = 16 - (len(msg_bytes) % 16)
        for i in range(length):
            msg_bytes += b' '
        iv = Random.new().read(AES.block_size)
        cipher = AES.new(self.secret_key.encode("utf8"),AES.MODE_CBC,iv)
        encoded = base64.b64encode(iv + cipher.encrypt(msg_bytes))
        self.connection.sendall(encoded)

    # fills the buffer based on the current input, by appending to numpy arrays
    # tbh idk how to handle the aggregating of data together cause i suck at manipulating numpy arrays
    def fill_buffer(self, input, buffer):
        #thsi function crashes when the list is not exactly 8? 
        while (len(input) != 8):
            input.append(0)

        # parse inputs to uint8, proper conversion will be done later on
        temp_input = np.empty([0], dtype=np.uint8)

        for i in range(len(input)):
            temp_input = np.append(temp_input, np.array([int(round(input[i] * 100))], dtype=np.uint8), axis=0) # perform rough conversion for now
        # will need to clean up this part in the future
        temp_input_2 = np.array([[temp_input[0], temp_input[1], temp_input[2], temp_input[3],
                                  temp_input[4], temp_input[5], temp_input[6], temp_input[7]], ] )
        buffer = np.concatenate((buffer, temp_input_2))
        return buffer

    # splits message based on the given format, with specified delimiter, outputs the input, start_flag
    # yaw / pitch / roll / time / start_flag
    def parse_message(self, message):
        data_as_strings = re.split(r'\t+',message)
        rpy_floats = [float(data_as_strings[0]), float(data_as_strings[1]), float(data_as_strings[2])]
        flags_int = int(data_as_strings[4])
        is_dancing = False
        is_moving_L = False
        is_moving_R = False
        if flags_int == 1:
            is_dancing = True
        elif flags_int == 2:
            is_moving_R = True
        elif flags_int == 4:
            is_moving_L = True

        return rpy_floats, is_dancing, is_moving_L, is_moving_R

    def run(self):
        collect_data = False
        flag_prev = False
        collect_flag = 0
        data = []
        count = 0 # the number of packets in the buffer
        count_secondary = 0

        buffer = np.empty([1, 8], dtype=np.uint8)

        while not self.shutdown.is_set():
            data = self.connection.recv(1024)

            if data:
                try:
                    self.recv_time = float(time.time())
                    msg = data.decode("utf8")
                    decrypted_message = self.decrypt_message(msg)
                    self.sendReply()
                    self.RTT = decrypted_message['rtt']
                    self.clock_offset = decrypted_message['clock_offset']
                    #print("Dancer " + str(self.dancer) + ": " + str(decrypted_message['message']))
                    #print("Dancer " + str(self.dancer) + " RTT: " + str(self.RTT) + " Offset: " + str(self.clock_offset))
                    #print("Currently received " + str(count) + " inputs in the buffer")

                    rpy_data, is_dancing, is_moving_L, is_moving_R = self.parse_message(decrypted_message['message']) # please input the proper string and remove this comment
                    #if is_moving_L:
                    #    print("<<< User moving left")
                    #if is_moving_R:
                    #    print(">>> User moving right")

                    if is_dancing and not flag_prev: # set flag to start collecting if flag == 1
                        collect_data = True
                    flag_prev = is_dancing

                    if collect_data:
                        #buffer = self.fill_buffer(rpy_data, buffer)
                        count_secondary += 1

                    if count_secondary == 8:
                        buffer = self.fill_buffer(rpy_data, buffer)
                        count_secondary = 0
                        count += 1

                    if count >= 8: # buffer filled, perform inference
                        buffer = np.delete(buffer, 0, 0) # remove the dummy row that was initialized
                        input = buffer # set input so that we can clear buffer in the event of pl server failure
                        buffer = np.empty([1, 8], dtype=np.uint8) # reset
                        count = 0
                        collect_data = False

                        try:
                            prediction = driver.inference(input) # returns a prediction based on class
                            print(str(prediction))
                        except ConnectionError:
                            print("attempting to restart pl_server")
                            subprocess.call("sudo service pl_server restart")
                        sendToEvalServer('#1 2 3|' + actions[int(prediction)] + '|0.00|')
                        sendToDashboard('#1 2 3|' + actions[int(prediction)] + '|0.00|')

                except Exception as e:
                    print(e)
            else:
                print('no more data from', self.client_address, file=sys.stderr)
                self.stop()

    def setup_connection(self, dancer_num):
        # Wait for a connection
        print('Waiting for dancer ' + str(dancer_num) + ' to connect.')
        self.connection, client_address = self.socket.accept()

        secret_key = "BF7EE4A12FA30C12826CD8A4DC1546A6"

        print('connection from ', client_address, file=sys.stderr)
        return client_address, secret_key

    def stop(self):
        self.connection.close()
        self.shutdown.set()
        self.timer.cancel()

def main():
    startSender() # ultra96 -> eval_server
    startDashboardSender() # ultra06 -> dashboard

    ip_addr = '0.0.0.0'
    #Port numbers of dancer 1, 2 and 3

    #For running on real server
    port_num = [3001, 3002, 3003]

    #For running locally
    #port_num = [2601, 2602, 2603]

    server1 = Server(ip_addr, port_num[0], 1)
    #server2 = Server(ip_addr, port_num[1], 2)
    #server3 = Server(ip_addr, port_num[2], 3)
    server1.start()
    #server2.start()
    #server3.start()

if __name__ == '__main__':
    main()


