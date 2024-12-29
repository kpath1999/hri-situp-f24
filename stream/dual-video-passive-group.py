# dual-video.py sync side and front camera stream and data writes

from vidgear.gears import CamGear
import cv2
import time

import os
import math
import random
import argparse
import pickle
import http.client, urllib

import numpy as np
import pandas as pd
import logging

import mediapipe as mp

parser = argparse.ArgumentParser(description='Runs mediapipe and publishes data')
parser.add_argument('-v', '--verbose', action='store_true', 
                    help='Flag to turn on printing')
parser.add_argument('-d', '--display_off', action='store_true', 
                    help='Flag to show camera feeds')
parser.add_argument('-sc', '--side_cam_idx', type=int, default=2,
                    help='index of side camera to stream from (Default: 1)')
parser.add_argument('-s', '--side_cam_pos', type=str, default="right", choices=['left', 'right'], 
                    help='specify the side of the desk the camera is on (Default: right)')
parser.add_argument('-f', '--file', type=str, default="test", 
                    help='Prefix of filename to save data (Default: test)')

# if using camera front cam
"""
parser.add_argument('-fc', '--front_cam_idx', type=int, default=0,
                    help='index of front camera to stream from (Default: 0)')
"""

# if using webcam
parser.add_argument('-fc', '--front_cam_idx', type=int, default=0,
                    help='index of logitech webcam (Default: 2)')

os.makedirs('../logs', exist_ok=True)
logging.basicConfig(filename=f'../logs/posture_data_{time.strftime("%Y-%m-%d-%H-%M-%S")}.txt', level=logging.INFO)
logger = logging.getLogger()

def write_to_pickle(file, data):
    # Open the pickle file and load existing data if any
    try:
        with open(file, 'rb') as f:
            posture_data = pickle.load(f)
    except FileNotFoundError:
        front = {
            "center.x": [],
            "center.y": [],
            "center.z": [],
            "l_shldr.x": [],
            "l_shldr.y": [],
            "l_shldr.z": [],
            "r_shlrd.x": [],
            "r_shlrd.y": [],
            "r_shlrd.z": [],
        }
        side = {
            "shlrd.x": [],
            "shlrd.y": [],
            "shlrd.z": [],
            "ear.x": [],
            "ear.y": [],
            "ear.z": [],
            "hip.x": [],
            "hip.y": [],
            "hip.z": [],
        }
        measured = {
            "neck_inclination": [],
            "torso_inclination": [],
            "level": [],
            "lean": [],
            "eval_score": [],
            "true_score": [],
            "composite_score": [],
            "spine_length": [],
        }
        posture_data = {"timeStamp":[], 
                        "raw": {"front": front, "side": side}, 
                        "measured": measured,
                        "sent_vibration": [],
                        "sent_sound": [],
                        "moved_desk": []}
    
    # Append new data
    posture_data["timeStamp"].append(data["timestamp"])
    for key in data["raw"].keys():
        for cam in data["raw"][key].keys():
            posture_data["raw"][key][cam].append(data["raw"][key][cam])

    for key in data["measured"].keys():
        posture_data["measured"][key].append(data["measured"][key])

    posture_data["sent_vibration"].append(data["sent_vibration"])
    posture_data["sent_sound"].append(data["sent_sound"])
    posture_data["moved_desk"].append(data["moved_desk"])
    
    # Write updated data back to pickle file
    with open(file, 'wb') as f:
        pickle.dump(posture_data, f)


def prepare_for_csv(file, data):
    # Try to load existing data, or initialize as a list if file not found
    try:
        with open(file, 'rb') as f:
            posture_data = pickle.load(f)
    except FileNotFoundError:
        posture_data = []  # Initialize as a list to store entries for each timestamp
    
    # Flatten the data for the current timestamp
    flattened_entry = {
        "timestamp": data["timestamp"]
    }
    
    # Flatten "raw" front and side data
    for key in data["raw"]["front"].keys():
        flattened_entry[f"raw_front_{key}"] = data["raw"]["front"][key]
    for key in data["raw"]["side"].keys():
        flattened_entry[f"raw_side_{key}"] = data["raw"]["side"][key]
    
    # Flatten "measured" data
    for key, value in data["measured"].items():
        flattened_entry[f"measured_{key}"] = value
    
    # Append "sent_vibration" and "sent_sound" data
    flattened_entry["sent_vibration"] = data["sent_vibration"]
    flattened_entry["sent_sound"] = data["sent_sound"]
    flattened_entry["moved_desk"] = data["moved_desk"]

    # Append the flattened data for the current timestamp to posture_data
    posture_data.append(flattened_entry)
    
    # Write updated data back to pickle file
    with open(file, 'wb') as f:
        pickle.dump(posture_data, f)
    

def form_datapoint(front_detect, side_detect, lmPose, side_cam_pos):
    # get front points
    front_pose = front_detect.pose_landmarks.landmark
    r_shldr = front_pose[lmPose.RIGHT_SHOULDER]
    l_shldr = front_pose[lmPose.LEFT_SHOULDER]
    r_ear = front_pose[lmPose.RIGHT_EAR]
    l_ear = front_pose[lmPose.LEFT_EAR]

    front = {
        "center.x": (r_ear.x + l_ear.x) / 2, 
        "center.y": (r_ear.y + l_ear.y) / 2, 
        "center.z": (r_ear.z + l_ear.z) / 2, 
        "l_shldr.x": l_shldr.x,
        "l_shldr.y": l_shldr.y, 
        "l_shldr.z": l_shldr.z,
        "r_shlrd.x": r_shldr.x,
        "r_shlrd.y": r_shldr.y,
        "r_shlrd.z": r_shldr.z
    }

    # get side points
    side_pose = side_detect.pose_landmarks.landmark
    # choose correct side
    if side_cam_pos == 'left':
        shldr = side_pose[lmPose.LEFT_SHOULDER]
        ear = side_pose[lmPose.LEFT_EAR]
        hip = side_pose[lmPose.LEFT_HIP]
    else:
        shldr = side_pose[lmPose.RIGHT_SHOULDER]
        ear = side_pose[lmPose.RIGHT_EAR]
        hip = side_pose[lmPose.RIGHT_HIP]
    side = {
        "shlrd.x": shldr.x,
        "shlrd.y": shldr.y,
        "shlrd.z": shldr.z, 
        "ear.x": ear.x,
        "ear.y": ear.y,
        "ear.z": ear.z, 
        "hip.x": hip.x,
        "hip.y": hip.y,
        "hip.z": hip.z
    }

    return {"front": front, "side": side}


def draw_landmarks_front(image, detection_result, lmPose, lean, level, score, composite):
    ''' Function to draw front landmarks, either overlayed on original image, or on a blank image. Returns annotated image '''
    h, w = image.shape[0:2]
    annotated_image = np.copy(image)

    # grab relevent poses
    pose = detection_result.pose_landmarks.landmark
    r_shldr = pose[lmPose.RIGHT_SHOULDER]
    l_shldr = pose[lmPose.LEFT_SHOULDER]
    r_ear = pose[lmPose.RIGHT_EAR]
    l_ear = pose[lmPose.LEFT_EAR]

    # convert to pixel space
    r_shldr_x = int(r_shldr.x * w)
    r_shldr_y = int(r_shldr.y * h)
    l_shldr_x = int(l_shldr.x * w)
    l_shldr_y = int(l_shldr.y * h)
    center_x = int((r_ear.x + l_ear.x) * w / 2)
    center_y = int((r_ear.y + l_ear.y) * h / 2)

    # choose color based on lean and level
    # lean_color = (127, 255, 0) if lean < 0.07 else (50, 50, 255)  # green if good else red
    # level_color = (127, 255, 0) if level < 0.03 else (50, 50, 255)  # green if good else red
    color = (127, 255, 0) if score >= 70 else (50, 50, 255)  # green if good else red

    # draw landmarks
    cv2.circle(annotated_image, (r_shldr_x, r_shldr_y), 7, (0, 255, 255), -1)
    cv2.circle(annotated_image, (l_shldr_x, l_shldr_y), 7, (0, 255, 255), -1)
    cv2.circle(annotated_image, (center_x, center_y), 7, (0, 255, 255), -1)

    # join landmarks
    cv2.line(annotated_image, (l_shldr_x, l_shldr_y), (r_shldr_x, r_shldr_y), color, 4)
    cv2.line(annotated_image, (l_shldr_x, l_shldr_y), (center_x, center_y), color, 4)
    cv2.line(annotated_image, (r_shldr_x, r_shldr_y), (center_x, center_y), color, 4)

    # add acore
    cv2.putText(annotated_image, f"Score: {score:.1f}%", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 255), 2)
    cv2.putText(annotated_image, f"Composite: {composite:.1f}%", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 255), 2)

    return annotated_image


def draw_landmarks_side(image, detection_result, lmPose, side_cam_pos, neck_inclination, torso_inclination, score, composite, spine_length):
    ''' Function to draw side landmarks, either overlayed on original image, or on a blank image. Returns annotated image '''
    h, w = image.shape[0:2]
    annotated_image = np.copy(image)

    pose = detection_result.pose_landmarks.landmark

    # choose correct side
    if side_cam_pos == 'left':
        shldr = pose[lmPose.LEFT_SHOULDER]
        ear = pose[lmPose.LEFT_EAR]
        hip = pose[lmPose.LEFT_HIP]
    else:
        shldr = pose[lmPose.RIGHT_SHOULDER]
        ear = pose[lmPose.RIGHT_EAR]
        hip = pose[lmPose.RIGHT_HIP]
    
    # convert to pixel space
    shldr_x = int(shldr.x * w)
    shldr_y = int(shldr.y * h)
    ear_x = int(ear.x * w)
    ear_y = int(ear.y * h)
    hip_x = int(hip.x * w)
    hip_y = int(hip.y * h)

    # choose color based on neck and torso inclination
    # posture_color = (127, 255, 0) if neck_inclination < 40 and torso_inclination < 10 else (50, 50, 255)  # green if good else red
    posture_color = (127, 255, 0) if score >= 70 else (50, 50, 255)  # green if good else red

    # draw landmarks
    cv2.circle(annotated_image, (shldr_x, shldr_y), 7, (0, 255, 255), -1)
    cv2.circle(annotated_image, (ear_x, ear_y), 7, (0, 255, 255), -1)
    cv2.circle(annotated_image, (shldr_x, shldr_y - 100), 7, (0, 255, 255), -1)
    cv2.circle(annotated_image, (hip_x, hip_y), 7, (0, 255, 255), -1)
    cv2.circle(annotated_image, (hip_x, hip_y - 100), 7, (0, 255, 255), -1)

    # Join landmarks.
    cv2.line(annotated_image, (shldr_x, shldr_y), (ear_x, ear_y), posture_color, 4)
    cv2.line(annotated_image, (shldr_x, shldr_y), (shldr_x, shldr_y - 100), posture_color, 4)
    cv2.line(annotated_image, (hip_x, hip_y), (shldr_x, shldr_y), posture_color, 4)
    cv2.line(annotated_image, (hip_x, hip_y), (hip_x, hip_y - 100), posture_color, 4)
    
    # add acore
    cv2.putText(annotated_image, f"Score: {score:.1f}%", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 255), 2)
    cv2.putText(annotated_image, f"Composite: {composite:.1f}%", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 255), 2)

    return annotated_image


def findDistance(x1, y1, z1, x2, y2, z2):
    """ Calculate Euclidean distance between two points in 3D space """
    dist = math.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
    return dist


def findAngle(x1, y1, z1, x2, y2, z2):
    """ Calculate angle between a 3D vector and the vertical axis """
    ''' i think this is angle relative to the horizon not relative to the back, fine if back is straight, but probably not an assumption we can make'''
    # Vector from point 1 to point 2
    vector = [x2-x1, y2-y1, z2-z1]
    
    # Try different vertical axes
    verticals = [[0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]
    angles = []
    
    for vertical in verticals:
        dot_product = sum([a*b for a, b in zip(vector, vertical)])
        mag_vector = math.sqrt(sum([v**2 for v in vector]))
        mag_vertical = 1    # magnitude of [0, 1, 0] is always 1
        
        cos_theta = dot_product / (mag_vector * mag_vertical)
        theta = math.acos(max(min(cos_theta, 1), -1))   # clamp to [-1, 1] to avoid domain error
        degree = math.degrees(theta)
        angles.append(degree)
    
    return min(angles)


def findRelativeAngle(x1, y1, z1, x2, y2, z2, x3, y3, z3):
    """ Calculate angle between two 3D vectors """
    vector1 = [x1-x2, y1-y2, z1-z2]
    vector2 = [x3-x2, y3-y2, z3-z2]
    dot_product = sum([a*b for a, b in zip(vector1, vector2)])
    magnitude1 = math.sqrt(sum([v**2 for v in vector1]))
    magnitude2 = math.sqrt(sum([v**2 for v in vector2]))
    
    cos_theta = dot_product / (magnitude1 * magnitude2)
    theta = math.acos(max(min(cos_theta, 1), -1))   # clamp to [-1, 1] to avoid domain error
    degree = math.degrees(theta)
    
    return degree

def process_front(detection_result, lmPose):
    ''' evaluate posture based on front camera input '''
    # grab the world coordinates we care about
    world_pose = detection_result.pose_world_landmarks.landmark
    r_shldr = world_pose[lmPose.RIGHT_SHOULDER]
    l_shldr = world_pose[lmPose.LEFT_SHOULDER]
    r_ear = world_pose[lmPose.RIGHT_EAR]
    l_ear = world_pose[lmPose.LEFT_EAR]

    level = abs(l_shldr.y - r_shldr.y)  # difference in y values

    # might need a better way to calculate this, im not convinced of the accuracy of z values when it cant see the hip
    lean = ((l_shldr.z + r_shldr.z) / 2) - ((l_ear.z + r_ear.z) / 2) # difference in mean z values

    return level, lean


def process_side(detection_results, lmPose, side_cam_pos):
    # Acquire the world position estimates.
    world_pose = detection_results.pose_world_landmarks.landmark
    
    # choose correct side
    if side_cam_pos == 'left':
        shldr = world_pose[lmPose.LEFT_SHOULDER]
        ear = world_pose[lmPose.LEFT_EAR]
        hip = world_pose[lmPose.LEFT_HIP]
    else:
        shldr = world_pose[lmPose.RIGHT_SHOULDER]
        ear = world_pose[lmPose.RIGHT_EAR]
        hip = world_pose[lmPose.RIGHT_HIP]

    # Shoulder.
    shldr_x, shldr_y, shldr_z = shldr.x, shldr.y, shldr.z
    
    # Ear.
    ear_x, ear_y, ear_z = ear.x, ear.y, ear.z
    
    # Hip.
    hip_x, hip_y, hip_z = hip.x, hip.y, hip.z

    # Calculate angles.
    # these probably need reworking, more in findAngle
    # neck_inclination = findAngle(shldr_x, shldr_y, ear_x, ear_y)
    torso_inclination = findAngle(hip_x, hip_y, hip_z, shldr_x, shldr_y, shldr_z)
    neck_inclination = findRelativeAngle(ear_x, ear_y, ear_z, shldr_x, shldr_y, shldr_z, hip_x, hip_y, hip_z)

    #print(f"Debug - Torso inclination: {torso_inclination}, Neck inclination: {neck_inclination}")
    
    # calculate spine length
    spine_length = findDistance(hip_x, hip_y, hip_z, shldr_x, shldr_y, shldr_z)
    
    return neck_inclination, torso_inclination, spine_length


def main(args):
    ''' dual stream front and side cameras with logging '''

    # start both camera streams
    stream2 = CamGear(source=args.side_cam_idx, logging=True).start() 
    stream1 = CamGear(source=args.front_cam_idx, logging=True).start() 
    side_cam_pos = args.side_cam_pos

    # Initialize mediapipe pose class
    mp_pose = mp.solutions.pose
    lmPose = mp_pose.PoseLandmark
    # need seperate classes for each detection
    front_pose = mp_pose.Pose()
    side_pose = mp_pose.Pose()
    
    last_write_time = time.time()
    
    # Get the current directory of this script
    current_dir = os.path.dirname(__file__)

    # File to store the posture pickle data
    file_name = f"{args.file}_posture_data_{time.strftime('%Y-%m-%d-%H-%M-%S')}.pkl"
    
    # Construct the path to the video file
    pickle_file = os.path.join(current_dir, '..', 'logs', file_name)

    # define posture evaluation 
    level_min = 0.02
    level_max = 0.07

    lean_min = 0.03
    lean_max = 0.1

    neck_min = 110  # severe forward head posture
    neck_max = 160  # slightly extended neck 

    torso_min = 0   # bent backwards??
    torso_max = 30  # significant forward lean
    
    #print(f"Debug - Neck range: {neck_min} - {neck_max}, Torso range: {torso_min} - {torso_max}")
    
    # initialize variables for notification control
    last_notification_time = 0
    notification_cooldown = 30      # cooldown period in seconds
    bad_posture_duration = 0
    bad_posture_threshold = 10      # seconds of continuous bad posture to trigger notification
    start_time = time.time()
    study_time = 35 * 60  # 35 minutes

    # initialize notification trackers
    moved_desk = 0
    sent_vibration = 0
    sent_sound = 0
    first_mod = random.choice(["vibrate", "ding"])
    continue_study = True

    # will go on for 35 mins, the duration of the study
    try: 
        while continue_study:
            # get current frame
            frameA = stream1.read()
            frameB = stream2.read()
            # frameB = np.copy(frameA)  # use this to test with only one camera, comment out stream2 and stream2.read and close at eof

            # check if any of two frame is None
            if frameA is None or frameB is None:
                break
            
            # Detect pose landmarks from the input image.
            front_detection_result = front_pose.process(frameA)
            side_detection_result = side_pose.process(frameB)

            # skipping detection if either cam doesn't find landmarks
            if side_detection_result.pose_landmarks is None or front_detection_result.pose_world_landmarks is None:
                if args.verbose:
                    print("ERROR: Missed frame")
                continue

            # process the landmarks accordingly
            level, lean = process_front(front_detection_result, lmPose)
            neck_inclination, torso_inclination, spine_length = process_side(side_detection_result, lmPose, side_cam_pos)

            # predict a posture score and compare to side cam score
            level_ = np.clip((level - level_min) / (level_max - level_min), 0, 1)
            lean_ = np.clip((lean - lean_min) / (lean_max - lean_min), 0, 1)

            # this will make torso_ highest (1.0) when torso_inclination is in the middle of the range
            neck_ = np.clip((neck_inclination - neck_min) / (neck_max - neck_min), 0, 1)
            torso_ = 1 - abs(2 * ((torso_inclination - torso_min) / (torso_max - torso_min) - 0.5))

            eval_posture = 40 * (1 - level_) + 60 * (1 - lean_)
            true_posture = 70 * neck_ + 30 * torso_
            
            # Metric that integrates both scores
            composite_score = 55 * neck_ + 40 * torso_ + 3 * (1 - lean_) + 2 * (1 - level_)

            # desk move logic will go here
            # if eval_posture mostly bad for the past t minutes:
            #     send move desk
            #     reset time counter
            
            # phone vibration/sound logic
            current_time = time.time()
            continue_study = current_time - start_time < study_time
            if current_time - last_write_time >= 0.5:
                sent_vibration = 0
                sent_sound = 0
                if current_time - start_time < 5 * 60:
                    modality = "none"
                elif current_time - start_time < 20 * 60:
                    modality = first_mod
                elif current_time - start_time < 35 * 60:
                    second_mod = "vibrate" if first_mod == "ding" else "ding"
                    modality = second_mod
                
                # every second minute, print the time and composite score
                if math.floor(((current_time - start_time) / 60)) % 2 == 0:
                    logger.info(f"time: {((current_time - start_time)/60):.2f} mins, modality: {modality}, composite: {composite_score:.2f}%")
                    print(f"time: {((current_time - start_time)/60):.2f} mins, modality: {modality}, composite: {composite_score:.2f}%")
                
                # don't ask me why this is the threshold, it just is. making it sensitive
                if composite_score <= 77:
                    bad_posture_duration += 0.5   # since loop runs every 0.5 seconds
                else:
                    bad_posture_duration = 0
                
                if (modality != "none" and bad_posture_duration >= bad_posture_threshold and current_time - last_notification_time >= notification_cooldown):
                    conn = http.client.HTTPSConnection("api.pushover.net:443")
                    
                    """ for Kausar's primary phone """
                    # User key: upc7noc9xcf417q66qk8stmo2akvc7
                    # Token: air5p7k39oc8ja6hdoqwyok53dtg3v
                    """ for Kausar's secondary phone """
                    # User key: ujtxo272bevjmysbrx2e85jm9y69ge
                    # Token key: abu1zinzovb48dc55foh6pn9ymqpe5
                    ''' Justins phone '''
                    # User key: aaztrycaxb92c4fstp9zf4x8upx66d
                    # Token key: uvkaj921wej8istc6h46m6gmamkkd8
                    
                    # Setting destination user and token based on modality
                    if modality == "vibrate":
                        token = "abu1zinzovb48dc55foh6pn9ymqpe5"
                        user = "ujtxo272bevjmysbrx2e85jm9y69ge"
                        sound = "vibrate"
                    else:   # "ding"
                        token = "air5p7k39oc8ja6hdoqwyok53dtg3v"
                        user = "upc7noc9xcf417q66qk8stmo2akvc7"
                        sound = "default"
                    
                    # make POST request to send message (vibration/ding)
                    conn.request("POST", "/1/messages.json",
                    urllib.parse.urlencode({
                        "token": token,
                        "user": user,
                        "title": "Posture Alert",
                        "message": f"Phone will {modality} now",
                        "url": "",
                        "priority": "1",
                        "sound": sound
                    }), { "Content-type": "application/x-www-form-urlencoded" })
                    # get response
                    conn.getresponse()
                    
                    # reset notification time and bad posture detection
                    last_notification_time = current_time
                    bad_posture_duration = 0
                    if modality == "vibrate":
                        sent_vibration = 1
                    else:
                        sent_sound = 1

                # Append data as a new entry.
                timestamp = time.strftime('%Y-%m-%d %H:%M:%S')

                measured_data = {
                    "neck_inclination": neck_inclination,
                    "torso_inclination": torso_inclination,
                    "level": level,
                    "lean": lean,
                    "eval_score": eval_posture, 
                    "true_score": true_posture,
                    "composite_score": composite_score,
                    "spine_length": spine_length, 
                }
                raw = form_datapoint(front_detection_result, side_detection_result, lmPose, side_cam_pos)
                data_dict = {"timestamp": timestamp, "raw": raw, "measured": measured_data, "moved_desk": moved_desk, "sent_vibration": sent_vibration, "sent_sound": sent_sound}
                #write_to_pickle(pickle_file, data_dict)
                prepare_for_csv(pickle_file, data_dict)
                last_write_time = current_time

            # turn off display from command line with -d
            if not args.display_off:
                # draw landmarks on front image
                front_stick_image = draw_landmarks_front(np.zeros_like(frameA), front_detection_result, lmPose, lean, level, eval_posture, composite_score)
                frameA = draw_landmarks_front(frameA, front_detection_result, lmPose, lean, level, eval_posture, composite_score)  # uncomment to overlay landmarks on image
                
                # Draw landmarks on side image
                side_stick_image = draw_landmarks_side(np.zeros_like(frameB), side_detection_result, lmPose, side_cam_pos, neck_inclination, torso_inclination, true_posture, composite_score, spine_length)
                frameB = draw_landmarks_side(frameB, side_detection_result, lmPose, side_cam_pos, neck_inclination, torso_inclination, true_posture, composite_score, spine_length)  # uncomment to overlay landmarks on image

                # Display video streams
                cv2.imshow("Front Cam", np.hstack((frameA, front_stick_image)))
                cv2.imshow("Side Cam", np.hstack((frameB, side_stick_image)))

                key = cv2.waitKey(1) & 0xFF
                # check for 'q' key-press
                if key == ord("q"):
                    break   # if q is pressed
            
    finally:
        # close output window
        cv2.destroyAllWindows()

        # safely close both video streams
        stream1.stop()
        stream2.stop()
        
        with open(pickle_file, 'rb') as f:
            data = pickle.load(f)
        
        # convert data to a pandas dataframe
        df = pd.DataFrame(data)
        
        # write to a csv file
        csv_output_file = os.path.join(current_dir, '..', 'logs/csv', file_name.replace('.pkl', '.csv'))
        df.to_csv(csv_output_file, index=False)
        print(f"Data has been dumped into {csv_output_file}.")

if __name__ == "__main__":
    main(parser.parse_args())