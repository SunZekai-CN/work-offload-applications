import pyrender
import cv2
from vibe.rt.rt_vibe import RtVibe
import time

rt_vibe = RtVibe()
rt_vibe.render = False
cap = cv2.VideoCapture('sample_video.mp4')
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print('end of stream')
        break
    frame = cv2.resize(frame, (1280, 800))

    tbegin = time.time()
    result = rt_vibe(frame)
    tend = time.time()
    print(f'{tend-tbegin:.3f}')
