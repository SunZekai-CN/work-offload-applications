import cv2
import time
import pickle
import zmq
import sys
from collections import OrderedDict

addr = sys.argv[1]

# connect zmq server
context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect(f"tcp://{addr}")
time.sleep(1)

cap = cv2.VideoCapture('sample_video.mp4')
# cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print(f'end of stream')
        break
    frame = cv2.resize(frame, (1280, 800))
    times = dict()

    total_tbegin = time.time()

    tbegin = time.time()
    frame = cv2.imencode('.jpg', frame)[1].tobytes()
    tend = time.time()
    times['compress'] = tend - tbegin

    tbegin = time.time()
    socket.send(pickle.dumps(frame))
    result = pickle.loads(socket.recv())
    tend = time.time()
    times['total'] = tend - total_tbegin
    times['request'] = tend - tbegin

    times.update(result[1])

    times['send+recv'] = times['request'] - times['decompress'] - times['compute']

    show_times = OrderedDict()
    for k in ['compute', 'compress', 'decompress', 'send+recv', 'total']:
        show_times[k] = f'{times[k]:.3f}'
    print(show_times)

    time.sleep(1)
