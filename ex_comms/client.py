import os
import sys
import time
import socket
from Crypto import Random
from Crypto.Cipher import AES
import base64

key = "BF7EE4A12FA30C12826CD8A4DC1546A6"
hostname = "youmusichub.places.sg"
port = 2500

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
while True:
    time.sleep(3)
    msg = b'#2 1 3|muscle|1.87|'
    length = 16 - (len(msg) % 16)
    for i in range(length):
        msg += b' '
    iv = Random.new().read(AES.block_size)
    cipher = AES.new(key.encode("utf8"),AES.MODE_CBC,iv)
    encoded = base64.b64encode(iv + cipher.encrypt(msg))
    s.sendall(encoded)
    print("Sent: " + str(msg))
    
s.close()

