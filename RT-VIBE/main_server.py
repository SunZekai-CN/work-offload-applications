import cv2
import numpy as np
import pyrender
from vibe.rt.rt_vibe import RtVibe
import time
import pickle
import zmq
import sys

port = sys.argv[1]

# create zmq server
context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind(f"tcp://*:{port}")

rt_vibe = RtVibe()
rt_vibe.render = False
print('listening...')
while True:

    msg = socket.recv()

    times = dict()

    tbegin = time.time()
    frame = pickle.loads(msg)
    tend = time.time()
    print(f'unpickle {tend-tbegin:.3f}')

    tbegin = time.time()
    frame = cv2.imdecode(np.frombuffer(frame, np.uint8), 1)
    tend = time.time()
    times['decompress'] = tend - tbegin
    print(f'decompress {tend-tbegin:.3f}')

    tbegin = time.time()
    result = rt_vibe(frame)
    tend = time.time()
    times['compute'] = tend - tbegin
    print(f'compute {tend-tbegin:.3f}')

    tbegin = time.time()
    socket.send(pickle.dumps((result, times)))
    tend = time.time()
    print(f'pickle+send {tend-tbegin:.3f}')

    print(f'--------')

