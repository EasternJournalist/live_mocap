import cv2
import numpy as np

print("Importing live_mocap")

# Overall pipeline
# 1. Capture frames and detect landmarks from multiple cameras in parallel
# 2. Track 3D landmarks from multiview 2D landmarks
# 3. Solve IK to get joint angles
