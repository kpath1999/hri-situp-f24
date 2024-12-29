# data-stream.py
import os
import math
import argparse

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np

import cv2
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


parser = argparse.ArgumentParser(description='Runs mediapipe and publishes data')
parser.add_argument('-v', '--verbose', action='store_true', 
                    help='Flag to turn on printing')
parser.add_argument('-c', '--cam_idx', type=int, default=0,
                    help='index of camera to stream from (Default: 0)')


def draw_landmarks_on_image(rgb_image, detection_result, level, lean):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected poses to visualize.
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]

        # Draw the pose landmarks.
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        for i, landmark in enumerate(pose_landmarks):
            pose_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) 
            ])
        solutions.drawing_utils.draw_landmarks(
        annotated_image,
        pose_landmarks_proto,
        solutions.pose.POSE_CONNECTIONS,
        solutions.drawing_styles.get_default_pose_landmarks_style())


        cv2.putText(annotated_image, f"Level: {level:.4f}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, .5, 255)
        cv2.putText(annotated_image, f"Lean: {lean:.4f}", (50, 75), cv2.FONT_HERSHEY_SIMPLEX, .5, 255)

        # fine tune these somehow probably
        level_min = 0.02
        level_max = 0.07

        lean_min = 0.06
        lean_max = 0.09
        level = np.clip((level - level_min) / (level_max - level_min), 0, 1)
        lean = np.clip((lean - lean_min) / (lean_max - lean_min), 0, 1)

        score = 50 * (1 - level) + 50 * (1 - lean)


        cv2.putText(annotated_image, f"Score: {score:.1f}%", (50, 25), cv2.FONT_HERSHEY_SIMPLEX, .5, 255)
    return annotated_image


def main(args):
    verbose = args.verbose
    cam_idx = args.cam_idx

    # load model
    base_options = python.BaseOptions(model_asset_path='models/pose_landmarker_full.task')
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=True)
    detector = vision.PoseLandmarker.create_from_options(options)

    # #########################
    # #### START MAIN LOOP ####
    # #########################

    # Initialize video capture object
    cap = cv2.VideoCapture(cam_idx)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_size = (width, height)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    # Define the codec and create VideoWriter object
    out = cv2.VideoWriter('videos/test.mp4', fourcc, fps, frame_size)

    print('Processing..')
    while cap.isOpened():
        # Capture frames.
        success, image = cap.read()
        stick_image = np.zeros_like(image)
        if not success:
            print("Null.Frames")
            break

        # Convert the BGR image to RGB.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # STEP 3: Load the input image.
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

        # STEP 4: Detect pose landmarks from the input image.
        detection_result = detector.detect(image)

        if detection_result.pose_world_landmarks == []:
            continue
        
        # grab the world coordinates we care about
        # need to calculate for level shoulders and ears inline with shoulders
        l_shldr = detection_result.pose_world_landmarks[0][11]
        r_shldr = detection_result.pose_world_landmarks[0][12]
        l_ear = detection_result.pose_world_landmarks[0][7]
        r_ear = detection_result.pose_world_landmarks[0][8]

        # breakpoint()
        level = abs(l_shldr.y - r_shldr.y)  # difference in y values
        lean = ((l_shldr.z + r_shldr.z) / 2) - ((l_ear.z + r_ear.z) / 2) # difference in mean z values

        # Process the detection result. In this case, visualize it.
        stick_image = draw_landmarks_on_image(stick_image, detection_result, level, lean)
        cv2.imshow("image", cv2.cvtColor(image.numpy_view(), cv2.COLOR_RGB2BGR))
        cv2.imshow("stick", cv2.cvtColor(stick_image, cv2.COLOR_RGB2BGR))
        # out.write(cv2.cvtColor(stick_image, cv2.COLOR_RGB2BGR))

        # display image, q to quit
        # cv2.imshow("Stream", annotated_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    print('Finished.')
    cap.release()
    out.release()

if __name__ == "__main__":
    main(parser.parse_args())