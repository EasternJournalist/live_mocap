import time
import threading
from queue import Queue
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


class Capturer:
    """
    A multithread pipeline for capturing and detecting landmarks from multiple cameras.
    """
    def __init__(
        self, 
        video_captures: List[cv2.VideoCapture],
        capture_face: bool = True,
        capture_pose: bool = True,
        capture_hands: bool = False,
    ):
        # VideoCaptures (cameras)
        self.video_captures = video_captures
        self.n_captures = len(video_captures)
        self.queue_capture = [Queue() for _ in range(self.n_captures)]

        # Detectors
        self.capture_face = capture_face
        if capture_face:
            self._init_face_detectors()
            self.queue_detect_face = [Queue() for _ in range(self.n_captures)]
            self.queue_gather_result_face = [Queue() for _ in range(self.n_captures)]
        
        self.capture_pose = capture_pose
        if capture_pose:
            self._init_pose_detectors()
            self.queue_detect_pose = [Queue() for _ in range(self.n_captures)]
            self.queue_gather_result_pose = [Queue() for _ in range(self.n_captures)]
        
        self.capture_hands = capture_hands
        if capture_hands:
            self._init_hand_detectors()
            self.queue_detect_hands = [Queue() for _ in range(self.n_captures)]
            self.queue_gather_result_hands = [Queue() for _ in range(self.n_captures)]

        self.queue_gather_result_output = Queue()
        self.barrier_capture = threading.Barrier(self.n_captures)
    
    def _init_face_detectors(self):
        self.face_detectors = []
        for i in range(self.n_captures):
            options = mp.tasks.vision.FaceLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path='./mediapipe-assets/face_landmarker_v2_with_blendshapes.task'),
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=True,
                running_mode=mp.tasks.vision.RunningMode.VIDEO,
                num_faces=1
            )
            detector = mp.tasks.vision.FaceLandmarker.create_from_options(options)
            self.face_detectors.append(detector)
        
    def _init_pose_detectors(self):
        self.pose_detectors = []
        for i in range(self.n_captures):
            options = mp.tasks.vision.PoseLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path='./mediapipe-assets/pose_landmarker_lite.task'),
                output_segmentation_masks=False,
                running_mode=mp.tasks.vision.RunningMode.VIDEO,
            )
            detector = mp.tasks.vision.PoseLandmarker.create_from_options(options)
            self.pose_detectors.append(detector)
    
    def _init_hand_detectors(self):
        raise NotImplementedError()
            
    def _caputure_thread_fn(self, idx: int):
        while True:
            flag = self.queue_capture[idx].get(block=True)
            
            # Synchronize cameras
            self.barrier_capture.wait()
            frame_time = time.time()
            self.video_captures[idx].grab()
            
            _, frame = self.video_captures[idx].retrieve()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            frame_timestamp_ms = int((frame_time - self.start_time) * 1000)

            if self.capture_face:
                self.queue_detect_face[idx].put((mp_image, frame_timestamp_ms))
            if self.capture_pose:
                self.queue_detect_pose[idx].put((mp_image, frame_timestamp_ms))
            if self.capture_hands:
                self.queue_detect_hands[idx].put((mp_image, frame_timestamp_ms))
    
    def _detect_face_thread_fn(self, idx: int):
        while True:
            mp_image, frame_timestamp_ms = self.queue_detect_face[idx].get(block=True)
            face_landmarker_result = self.face_detectors[idx].detect_for_video(mp_image, frame_timestamp_ms)
            self.queue_gather_result_face[idx].put(face_landmarker_result)

    def _detect_pose_thread_fn(self, idx: int):
        while True:
            mp_image, frame_timestamp_ms = self.queue_detect_pose[idx].get(block=True)
            pose_landmarker_result = self.pose_detectors[idx].detect_for_video(mp_image, frame_timestamp_ms)
            self.queue_gather_result_pose[idx].put(pose_landmarker_result)
    
    def _detect_hands_thread_fn(self, idx: int):
        raise NotImplementedError()
    
    def _gather_result_thread_fn(self):
        while True:
            result = {}
            if self.capture_face:
                result['face'] = [self.queue_gather_result_face[i].get(block=True) for i in range(self.n_captures)]
            if self.capture_pose:
                result['pose'] = [self.queue_gather_result_pose[i].get(block=True) for i in range(self.n_captures)]
            if self.capture_hands:
                result['hands'] = [self.queue_gather_result_hands[i].get(block=True) for i in range(self.n_captures)]
            self.queue_gather_result_output.put(result)

    def start(self):
        self.start_time = time.time()
        self.cnt_frames = 0

        # A hack to synchronize the start of the cameras
        for _ in range(10):
            for cam in self.video_captures:
                cam.grab()

        self.thread_capture = []
        for i in range(self.n_captures):
            thread = threading.Thread(target=self._caputure_thread_fn, args=(i,))
            thread.start()
            self.thread_capture.append(thread)
        
        if self.capture_face:
            self.thread_detect_face = []
            for i in range(self.n_captures):
                thread = threading.Thread(target=self._detect_face_thread_fn, args=(i,))
                thread.start()
                self.thread_detect_face.append(thread)
        
        if self.capture_pose:
            self.thread_detect_pose = []
            for i in range(self.n_captures):
                thread = threading.Thread(target=self._detect_pose_thread_fn, args=(i,))
                thread.start()
                self.thread_detect_pose.append(thread)
        
        if self.capture_hands:
            self.thread_detect_hands = []
            for i in range(self.n_captures):
                thread = threading.Thread(target=self._detect_hands_thread_fn, args=(i,))
                thread.start()
                self.thread_detect_hands.append(thread)
        
        self.thread_gather_result = threading.Thread(target=self._gather_result_thread_fn)
        self.thread_gather_result.start()

    def next_frame(self):
        self.cnt_frames += 1
        for i in range(self.n_captures):
            self.queue_capture[i].put(True)

    def get_result(self):
        return self.queue_gather_result_output.get(block=True)


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