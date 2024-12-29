# data-stream.py
import os
import math
import argparse

import cv2
import time
import mediapipe as mp
import numpy as np

parser = argparse.ArgumentParser(description='Runs mediapipe and publishes data')
parser.add_argument('-v', '--verbose', action='store_true', 
                    help='Flag to turn on printing')
parser.add_argument('-c', '--cam_idx', type=int, default=0,
                    help='index of camera to stream from (Default: 0)')


# Font type
FONT = cv2.FONT_HERSHEY_SIMPLEX

# Colors
BLUE = (255, 127, 0)
RED = (50, 50, 255)
GREEN = (127, 255, 0)
DARK_BLUE = (127, 20, 0)
LIGHT_GREEN = (127, 233, 100)
YELLOW = (0, 255, 255)
PINK = (255, 0, 255)


def findDistance(x1, y1, x2, y2):
    dist = math.sqrt((x2-x1)**2+(y2-y1)**2)
    return dist


def findAngle(x1, y1, x2, y2):
    theta = math.acos((y2-y1)*(-y1) / (math.sqrt((x2-x1)**2 + (y2-y1)**2)*y1))
    degree = int(180/math.pi)*theta
    return degree

def write_to_file(file, data):
    with open(file, 'a') as f:
        f.write(data + '\n')

def draw_landmarks(image, offset, neck_inclination, torso_inclination, pose_time, 
                   l_shldr_x, l_shldr_y, r_shldr_x, r_shldr_y, 
                   l_ear_x, l_ear_y, l_hip_x, l_hip_y):


    h, w = image.shape[0:2]
    # Assist to align the camera to point at the side view of the person.
    # Offset threshold 30 is based on results obtained from analysis over 100 samples.
    if offset < 100:
        cv2.putText(image, str(int(offset)) + ' Aligned', (w - 150, 30), FONT, 0.9, GREEN, 2)
    else:
        cv2.putText(image, str(int(offset)) + ' Not Aligned', (w - 150, 30), FONT, 0.9, RED, 2)

    # Draw landmarks.
    cv2.circle(image, (l_shldr_x, l_shldr_y), 7, YELLOW, -1)
    cv2.circle(image, (l_ear_x, l_ear_y), 7, YELLOW, -1)

    # Let's take y - coordinate of P3 100px above x1,  for display elegance.
    # Although we are taking y = 0 while calculating angle between P1,P2,P3.
    cv2.circle(image, (l_shldr_x, l_shldr_y - 100), 7, YELLOW, -1)
    cv2.circle(image, (r_shldr_x, r_shldr_y), 7, PINK, -1)
    cv2.circle(image, (l_hip_x, l_hip_y), 7, YELLOW, -1)

    # Similarly, here we are taking y - coordinate 100px above x1. Note that
    # you can take any value for y, not necessarily 100 or 200 pixels.
    cv2.circle(image, (l_hip_x, l_hip_y - 100), 7, YELLOW, -1)

    # Put text, Posture and angle inclination.
    # Text string for display.
    angle_text_string = 'Neck : ' + str(int(neck_inclination)) + '  Torso : ' + str(int(torso_inclination))

    # Determine whether good posture or bad posture.
    # The threshold angles have been set based on intuition.
    if neck_inclination < 40 and torso_inclination < 10:
        # place text
        cv2.putText(image, angle_text_string, (10, 30), FONT, 0.9, LIGHT_GREEN, 2)
        cv2.putText(image, str(int(neck_inclination)), (l_shldr_x + 10, l_shldr_y), FONT, 0.9, LIGHT_GREEN, 2)
        cv2.putText(image, str(int(torso_inclination)), (l_hip_x + 10, l_hip_y), FONT, 0.9, LIGHT_GREEN, 2)

        # Join landmarks.
        cv2.line(image, (l_shldr_x, l_shldr_y), (l_ear_x, l_ear_y), GREEN, 4)
        cv2.line(image, (l_shldr_x, l_shldr_y), (l_shldr_x, l_shldr_y - 100), GREEN, 4)
        cv2.line(image, (l_hip_x, l_hip_y), (l_shldr_x, l_shldr_y), GREEN, 4)
        cv2.line(image, (l_hip_x, l_hip_y), (l_hip_x, l_hip_y - 100), GREEN, 4)

        # show time in good
        time_string_good = 'Good Posture Time : ' + str(round(pose_time, 1)) + 's'
        cv2.putText(image, time_string_good, (10, h - 20), FONT, 0.9, GREEN, 2)

    else:
        # place text
        cv2.putText(image, angle_text_string, (10, 30), FONT, 0.9, RED, 2)
        cv2.putText(image, str(int(neck_inclination)), (l_shldr_x + 10, l_shldr_y), FONT, 0.9, RED, 2)
        cv2.putText(image, str(int(torso_inclination)), (l_hip_x + 10, l_hip_y), FONT, 0.9, RED, 2)

        # Join landmarks.
        cv2.line(image, (l_shldr_x, l_shldr_y), (l_ear_x, l_ear_y), RED, 4)
        cv2.line(image, (l_shldr_x, l_shldr_y), (l_shldr_x, l_shldr_y - 100), RED, 4)
        cv2.line(image, (l_hip_x, l_hip_y), (l_shldr_x, l_shldr_y), RED, 4)
        cv2.line(image, (l_hip_x, l_hip_y), (l_hip_x, l_hip_y - 100), RED, 4)

        # show time in bad
        time_string_bad = 'Bad Posture Time : ' + str(round(pose_time, 1)) + 's'
        cv2.putText(image, time_string_bad, (10, h - 20), FONT, 0.9, RED, 2)
        

def main(args):
    # grab arguements
    verbose = args.verbose
    cam_idx = args.cam_idx

    # Initialize mediapipe selfie segmentation class
    mp_pose = mp.solutions.pose
    mp_holistic = mp.solutions.holistic

    # Initialize frame counters
    good_frames = 0
    bad_frames = 0

    # Initialize mediapipe pose class
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()

    # Initialize video capture object
    cap = cv2.VideoCapture(cam_idx)

    # Meta
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_size = (w, h)

    if verbose:
        print("Status:", fps, frame_size)

    # ##### TO TYPE TO FILE!!!
    # data_file = "posture_data.txt"
    # # Initialize timer
    # last_write_time = time.time()

    #########################
    #### START MAIN LOOP ####
    #########################
    print('Processing..')
    while cap.isOpened():
        # Capture frames.
        success, image = cap.read()
        stick_figure = np.zeros_like(image)
        if not success:
            print("Null.Frames")
            break

        # Convert the BGR image to RGB.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image.
        keypoints = pose.process(image)

        # Convert the image back to BGR.
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Use lm and lmPose as representative of the following methods.
        lm = keypoints.pose_landmarks
        lmPose = mp_pose.PoseLandmark

        # make sure we captured landmarks before processing
        if lm is not None:
            
            # Acquire the landmark coordinates.
            # Once aligned properly, left or right should not be a concern.
            # Left shoulder.
            l_shldr_x = int(lm.landmark[lmPose.LEFT_SHOULDER].x * w)
            l_shldr_y = int(lm.landmark[lmPose.LEFT_SHOULDER].y * h)
            # Right shoulder
            r_shldr_x = int(lm.landmark[lmPose.RIGHT_SHOULDER].x * w)
            r_shldr_y = int(lm.landmark[lmPose.RIGHT_SHOULDER].y * h)
            # Left ear.
            l_ear_x = int(lm.landmark[lmPose.LEFT_EAR].x * w)
            l_ear_y = int(lm.landmark[lmPose.LEFT_EAR].y * h)
            # Left hip.
            l_hip_x = int(lm.landmark[lmPose.LEFT_HIP].x * w)
            l_hip_y = int(lm.landmark[lmPose.LEFT_HIP].y * h)

            # Calculate distance between left shoulder and right shoulder points.
            offset = findDistance(l_shldr_x, l_shldr_y, r_shldr_x, r_shldr_y)

            # Calculate angles.
            neck_inclination = findAngle(l_shldr_x, l_shldr_y, l_ear_x, l_ear_y)
            torso_inclination = findAngle(l_hip_x, l_hip_y, l_shldr_x, l_shldr_y)

            current_time = time.time()
            if current_time - last_write_time >= 1:
                data_string = f"{time.strftime('%Y-%m-%d %H:%M:%S')} | Neck Inclination: {neck_inclination:.2f} | Torso Inclination: {torso_inclination:.2f}"
                write_to_file(data_file, data_string)
                last_write_time = current_time
            
            
            # can adjust these to how we want to define posture
            # also could just look into measuring deviation from alignment
            if neck_inclination < 40 and torso_inclination < 10:
                bad_frames = 0
                good_frames += 1
                
            else:
                good_frames = 0
                bad_frames += 1

            draw_landmarks(stick_figure, offset, neck_inclination, torso_inclination, (good_frames if good_frames > 0 else bad_frames) / fps, 
                           l_shldr_x, l_shldr_y, r_shldr_x, r_shldr_y, 
                           l_ear_x, l_ear_y, l_hip_x, l_hip_y)
            
            draw_landmarks(image, offset, neck_inclination, torso_inclination, (good_frames if good_frames > 0 else bad_frames) / fps, 
                           l_shldr_x, l_shldr_y, r_shldr_x, r_shldr_y, 
                           l_ear_x, l_ear_y, l_hip_x, l_hip_y)

        # display image, q to quit
        cv2.imshow("Stream", image)
        cv2.imshow("Stick", stick_figure)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
# def main(args):
#     verbose = args.verbose
#     cam_idx = args.cam_idx

#     # Initialize mediapipe pose class
#     mp_pose = mp.solutions.pose
#     pose = mp_pose.Pose()

#     # Initialize video capture object
#     cap = cv2.VideoCapture(cam_idx)

#     # Meta
#     fps = int(cap.get(cv2.CAP_PROP_FPS))
#     w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     frame_size = (w, h)

#     if verbose:
#         print("Status:", fps, frame_size)

#     # File to store posture data
#     data_file = "posture_data.txt"

#     # Initialize timer
#     last_write_time = time.time()

#     print('Processing..')
#     while cap.isOpened():
#         success, image = cap.read()
#         if not success:
#             print("Null.Frames")
#             break

#         # Convert the BGR image to RGB
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#         # Process the image to find keypoints
#         keypoints = pose.process(image)

#         # Convert the image back to BGR
#         image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

#         lm = keypoints.pose_landmarks
#         lmPose = mp_pose.PoseLandmark

#         if lm is not None:
#             l_shldr_x = int(lm.landmark[lmPose.LEFT_SHOULDER].x * w)
#             l_shldr_y = int(lm.landmark[lmPose.LEFT_SHOULDER].y * h)
#             r_shldr_x = int(lm.landmark[lmPose.RIGHT_SHOULDER].x * w)
#             r_shldr_y = int(lm.landmark[lmPose.RIGHT_SHOULDER].y * h)
#             l_ear_x = int(lm.landmark[lmPose.LEFT_EAR].x * w)
#             l_ear_y = int(lm.landmark[lmPose.LEFT_EAR].y * h)
#             l_hip_x = int(lm.landmark[lmPose.LEFT_HIP].x * w)
#             l_hip_y = int(lm.landmark[lmPose.LEFT_HIP].y * h)

#             # Calculate posture metrics
#             offset = findDistance(l_shldr_x, l_shldr_y, r_shldr_x, r_shldr_y)
#             neck_inclination = findAngle(l_shldr_x, l_shldr_y, l_ear_x, l_ear_y)
#             torso_inclination = findAngle(l_hip_x, l_hip_y, l_shldr_x, l_shldr_y)

#             # Write data every second
#             current_time = time.time()
#             if current_time - last_write_time >= 1:
#                 data_string = f"{time.strftime('%Y-%m-%d %H:%M:%S')} | Neck Inclination: {neck_inclination:.2f} | Torso Inclination: {torso_inclination:.2f}"
#                 write_to_file(data_file, data_string)
#                 last_write_time = current_time

#         # Display image
#         cv2.imshow("Stream", image)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     print('Finished.')
#     cap.release()

#     print('Finished.')
#     cap.release()

if __name__ == "__main__":
    main(parser.parse_args())