import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time

import cv2

from live_mocap.capturer import Capturer


video_captures = []
for i in range(10):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        video_captures.append(cap)
    else:
        cap.release()
        break

print(f"Number of video captures: {len(video_captures)}")

capturer = Capturer(video_captures)

capturer.start()

recent_timestamps = []
while True:
    capturer.next_frame()
    result = capturer.get_result()
    recent_timestamps.append(time.time())
    if len(recent_timestamps) > 30:
        recent_timestamps.pop(0)
    
    if len(recent_timestamps) > 3:
        fps = (len(recent_timestamps) - 1) / (recent_timestamps[-1] - recent_timestamps[0])
        print(f"FPS: {fps:.2f}")
