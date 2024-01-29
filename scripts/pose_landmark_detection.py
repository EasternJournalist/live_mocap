import time
import threading
import cv2
import numpy as np
from typing import *

import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2


def draw_landmarks_on_image(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected poses to visualize.
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]

        # Draw the pose landmarks.
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
        ])
        mp.solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            mp.solutions.pose.POSE_CONNECTIONS,
            mp.solutions.drawing_styles.get_default_pose_landmarks_style()
        )
    return annotated_image


def caputure_detect_thread_fn(
    idx: int, 
    frames: List[np.ndarray], 
    results: List[Any], 
    cap: cv2.VideoCapture, 
    detector: mp.tasks.vision.FaceLandmarker, 
    barrier_grab: threading.Barrier, 
    event_proceed: threading.Event, 
    event_ready: threading.Event, 
    event_exit: threading.Event
):
    while True:
        event_proceed.wait()
        event_proceed.clear()
        frame_time = time.time()
        if event_exit.is_set():
            break
        
        barrier_grab.wait()
        cap.grab()
        barrier_grab.wait()
        
        _, frame = cap.retrieve()
        frames[idx] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results[idx] = detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=frames[idx]), int((frame_time - start_time) * 1000))
        # frames[idx] = draw_landmarks_on_image(frames[idx], results[idx])
        event_ready.set()


if __name__ == '__main__':
    base_options = mp.tasks.BaseOptions(model_asset_path='./mediapipe-assets/pose_landmarker_lite.task')

    # Open video capture devices
    MAX_CAMERAS = 2
    cameras = []
    for i in range(MAX_CAMERAS):
        cam = cv2.VideoCapture(i)
        if cam.isOpened():
            cameras.append(cam)
        else:
            cam.release()
            break
    n_cameras = len(cameras)
    print(f'Found {len(cameras)} cameras')
    
    # Initialize landmark detectors for each camera
    detectors = []
    for i in range(n_cameras):
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=False,
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
        )
        detectors.append(mp.tasks.vision.PoseLandmarker.create_from_options(options))

    start_time = time.time()
    
    cnt_frames = 0

    threads = []
    frames, results = [None] * n_cameras, [None] * n_cameras
    events_proceed = [threading.Event() for _ in range(n_cameras)]
    events_ready = [threading.Event() for _ in range(n_cameras)]
    event_exit = threading.Event()
    barrier_capture = threading.Barrier(n_cameras)
    for i in range(n_cameras):
        thread = threading.Thread(
            target=caputure_detect_thread_fn, 
            args=(i, frames, results, cameras[i], detectors[i], barrier_capture, events_proceed[i], events_ready[i], event_exit)
        )
        thread.start()
        threads.append(thread)

    # A hack to synchronize the start of the cameras
    for _ in range(10):
        for cam in cameras:
            cam.grab()

    while(True): 
        # Capture video frames
        current_time = time.time()
        cnt_frames += 1

        # Proceed next frame
        for i in range(n_cameras):
            events_proceed[i].set()
        # Wait for all frames to be ready
        for i in range(n_cameras):
            events_ready[i].wait()
            events_ready[i].clear()

        # Display the resulting frame
        cv2.imshow('frame', cv2.cvtColor(np.concatenate(frames, axis=1), cv2.COLOR_RGB2BGR))
        print(f'FPS: {cnt_frames / (time.time() - start_time)}')

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    event_exit.set() 
    for i in range(n_cameras):
        events_proceed[i].set()

    for i in range(n_cameras):
        threads[i].join()
        cameras[i].release()
        detectors[i].close()
    cv2.destroyAllWindows() 