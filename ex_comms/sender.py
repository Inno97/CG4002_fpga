import os
import sys
import time
import socket
from Crypto import Random
from Crypto.Cipher import AES
import base64

key = "BF7EE4A12FA30C12826CD8A4DC1546A6"
hostname = "137.132.92.63"
port = 2500
s = None

db_hostname = "202.166.37.137"
db_port = 5555
sd = None


def startSender():
    global s
    try: 
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
        print("Socket successfully created")
    except socket.error as err: 
        print("socket creation failed with error " + str(err))
    
    
    try: 
        host_ip = socket.gethostbyname(hostname) 
    except socket.gaierror: 
        print("Host not found")
        sys.exit() 

    s.connect((host_ip, port)) 

    print("the socket has successfully connected")

def startDashboardSender():
    global sd
    try: 
        sd = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
        print("Socket for dashboard created.")
    except socket.error as err: 
        print("socket creation for dashboard failed with error " + str(err))
    
    
    try: 
        host_ip = socket.gethostbyname(db_hostname) 
    except socket.gaierror: 
        print("Dashboard host not found")
        sys.exit() 

    sd.connect((host_ip, db_port)) 

    print("Dashboard successfully connected")

def sendToEvalServer(message):
    msg = str.encode(message)
    length = 16 - (len(msg) % 16)
    for i in range(length):
        msg += b' '
    iv = Random.new().read(AES.block_size)
    cipher = AES.new(key.encode("utf8"),AES.MODE_CBC,iv)
    encoded = base64.b64encode(iv + cipher.encrypt(msg))
    s.sendall(encoded)
    print("Sent: " + str(msg))
    
def sendToDashboard(message):
    msg = str.encode(message)
    length = 16 - (len(msg) % 16)
    for i in range(length):
        msg += b' '
    iv = Random.new().read(AES.block_size)
    cipher = AES.new(key.encode("utf8"),AES.MODE_CBC,iv)
    encoded = base64.b64encode(iv + cipher.encrypt(msg))
    sd.sendall(encoded)
    print("Sent to dashboard: " + str(msg))

def closeSender():
    global s, sd
    s.close()
    sd.close()

