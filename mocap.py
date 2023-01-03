#####################################################################################
# Single View Human Motion Capture, Based on Mediapipe & OpenCV & PyTorch
# 
# Author: Ruicheng Wang
# License: Apache License 2.0
#####################################################################################
import os
import shutil
import argparse
import pickle
import subprocess

import numpy as np
import cv2
import torch
from tqdm import tqdm

from body_keypoint_track import BodyKeypointTrack, show_annotation
from skeleton_ik_solver import SkeletonIKSolver


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--blend', type=str, help='Path to rigged model Blender file. eg. c:\\tmp\\model.blend')
    parser.add_argument('--video', type=str, help='Path to video file. eg. c:\\tmp\\video.mp4')
    parser.add_argument('--track_hands', action='store_true', help='Enable hand tracking')

    args = parser.parse_args()
    FOV = np.pi / 3

    # Call blender to export skeleton
    os.makedirs('tmp', exist_ok=True)
    print("Export skeleton...")
    if os.path.exists('tmp/skeleton'):
        shutil.rmtree('tmp/skeleton')
    os.system(f"blender {args.blend} --background --python export_skeleton.py")
    if not os.path.exists('tmp/skeleton'):
        raise Exception("Skeleton export failed")

    # Open the video capture
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise Exception("Video capture failed")
    frame_width, frame_height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize the body keypoint tracker
    body_keypoint_track = BodyKeypointTrack(
        im_width=frame_width,
        im_height=frame_height,
        fov=FOV,
        frame_rate=frame_rate,
        track_hands=args.track_hands,
        smooth_range=10 * (1 / frame_rate),
        smooth_range_barycenter=30 * (1 / frame_rate),
    )

    # Initialize the skeleton IK solver
    skeleton_ik_solver = SkeletonIKSolver(
        model_path='tmp/skeleton',
        track_hands=args.track_hands,
        smooth_range=15 * (1 / frame_rate),
    )

    bone_euler_sequence, scale_sequence, location_sequence = [], [], []

    frame_t = 0.0
    frame_i = 0
    bar = tqdm(total=total_frames, desc='Running...')
    while cap.isOpened():
        # Get the frame image
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Get the body keypoints
        body_keypoint_track.track(frame, frame_t)
        kpts3d, valid = body_keypoint_track.get_smoothed_3d_keypoints(frame_t)

        # Solve the skeleton IK
        skeleton_ik_solver.fit(torch.from_numpy(kpts3d).float(), torch.from_numpy(valid).bool(), frame_t)

        # Get the skeleton pose
        bone_euler = skeleton_ik_solver.get_smoothed_bone_euler(frame_t)
        location = skeleton_ik_solver.get_smoothed_location(frame_t)
        scale = skeleton_ik_solver.get_scale()

        bone_euler_sequence.append(bone_euler)
        location_sequence.append(location)
        scale_sequence.append(skeleton_ik_solver.get_scale())

        # Show keypoints tracking result
        show_annotation(frame, kpts3d, valid, body_keypoint_track.K)
        if cv2.waitKey(1) == 27:
            print('Cancelled by user. Exit.')
            exit()

        frame_i += 1
        frame_t += 1.0 / frame_rate
        bar.update(1)

    # Save animation result
    print("Save animation result...")
    with open('tmp/bone_animation_data.pkl', 'wb') as fp:
        pickle.dump({
            'fov': FOV,
            'frame_rate': frame_rate,
            'bone_names': skeleton_ik_solver.optimizable_bones,
            'bone_euler_sequence': bone_euler_sequence,
            'location_sequence': location_sequence,
            'scale': np.mean(scale_sequence),
            'all_bone_names': skeleton_ik_solver.all_bone_names
        }, fp)

    # Open blender and apply the animation
    print("Open blender and apply animation...")
    
    proc = subprocess.Popen(f"blender {args.blend} --python apply_animation.py")
    proc.wait()


if __name__ == '__main__':
    main()