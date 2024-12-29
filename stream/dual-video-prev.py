# dual-video.py sync side and front camera stream and data writes

from vidgear.gears import CamGear
import cv2
import time

import os
import math
import argparse
import pickle
import http.client, urllib

import numpy as np

import mediapipe as mp
import pandas as pd

parser = argparse.ArgumentParser(description='Runs mediapipe and publishes data')
parser.add_argument('-v', '--verbose', action='store_true', 
                    help='Flag to turn on printing')
parser.add_argument('-d', '--display_off', action='store_true', 
                    help='Flag to show camera feeds')
parser.add_argument('-fc', '--front_cam_idx', type=int, default=0,
                    help='index of front camera to stream from (Default: 0)')
parser.add_argument('-sc', '--side_cam_idx', type=int, default=1,
                    help='index of side camera to stream from (Default: 1)')
parser.add_argument('-s', '--side_cam_pos', type=str, default="right", choices=['left', 'right'], 
                    help='specify the side of the desk the camera is on (Default: right)')
parser.add_argument('-f', '--file', type=str, default="test", 
                    help='Prefix of filename to save data (Default: test)')

def write_to_pickle(file, data):
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


def draw_landmarks_front(image, detection_result, lmPose, lean, level, score):
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
    color = (127, 255, 0) if score >= 75 else (50, 50, 255)  # green if good else red

    # draw landmarks
    cv2.circle(annotated_image, (r_shldr_x, r_shldr_y), 7, (0, 255, 255), -1)
    cv2.circle(annotated_image, (l_shldr_x, l_shldr_y), 7, (0, 255, 255), -1)
    cv2.circle(annotated_image, (center_x, center_y), 7, (0, 255, 255), -1)

    # join landmarks
    cv2.line(annotated_image, (l_shldr_x, l_shldr_y), (r_shldr_x, r_shldr_y), color, 4)
    cv2.line(annotated_image, (l_shldr_x, l_shldr_y), (center_x, center_y), color, 4)
    cv2.line(annotated_image, (r_shldr_x, r_shldr_y), (center_x, center_y), color, 4)

    # add acore
    cv2.putText(annotated_image, f"Score: {score:.1f}%", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 255))

    return annotated_image


def draw_landmarks_side(image, detection_result, lmPose, side_cam_pos, neck_inclination, torso_inclination, score, spine_length):
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
    posture_color = (127, 255, 0) if score > 75 else (50, 50, 255)  # green if good else red

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
    cv2.putText(annotated_image, f"Score: {score:.1f}%", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 255))

    return annotated_image


def findDistance(x1, y1, x2, y2):
    dist = math.sqrt((x2-x1)**2+(y2-y1)**2)
    return dist


def findAngle(x1, y1, x2, y2):
    ''' i think this is angle relative to the horizon not relative to the back, fine if back is straight, but probably not an assumption we can make'''
    theta = math.acos((y2-y1)*(-y1) / (math.sqrt((x2-x1)**2 + (y2-y1)**2)*y1))
    degree = int(180/math.pi)*theta
    return degree


def findRelativeAngle(x1, y1, x2, y2, x3, y3):
    if (y2 != y1):
        slope1 = (x2 - x1) / (y2 - y1)
    else:
        slope1 = 0
    
    if (y2 - y3) :
        slope2 = (x2 - x3) / (y2 - y3)
    else:
        slope2 = 0
    
    theta =  abs(math.atan((slope1 - slope2) / (1 + slope1*slope2)))
    degree = int(180/math.pi)*theta

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


def process_side(detection_results, lmPose, input_shape, side_cam_pos):
    ''' this function is using pixel location should we be using world position esimates instead?'''
    h, w, _ = input_shape
    
    # Acquire the landmark coordinates.
    pose = detection_results.pose_landmarks.landmark
    # choose correct side
    if side_cam_pos == 'left':
        shldr = pose[lmPose.LEFT_SHOULDER]
        ear = pose[lmPose.LEFT_EAR]
        hip = pose[lmPose.LEFT_HIP]
    else:
        shldr = pose[lmPose.RIGHT_SHOULDER]
        ear = pose[lmPose.RIGHT_EAR]
        hip = pose[lmPose.RIGHT_HIP]

    # Left shoulder.
    shldr_x = int(shldr.x * w)
    shldr_y = int(shldr.y * h)
    # Right shoulder
    # r_shldr_x = int(pose[lmPose.RIGHT_SHOULDER].x * w)
    # r_shldr_y = int(pose[lmPose.RIGHT_SHOULDER].y * h)
    # Left ear.
    ear_x = int(ear.x * w)
    ear_y = int(ear.y * h)
    # Left hip.
    hip_x = int(hip.x * w)
    hip_y = int(hip.y * h)

    # Calculate distance between left shoulder and right shoulder points. 
    # Why?
    # offset = findDistance(l_shldr_x, l_shldr_y, r_shldr_x, r_shldr_y)
    offset = 0

    # Calculate angles.
    # these probably need reworking, more in findAngle
    # neck_inclination = findAngle(shldr_x, shldr_y, ear_x, ear_y)
    torso_inclination = findAngle(hip_x, hip_y, shldr_x, shldr_y)

    neck_inclination = findRelativeAngle(ear_x, ear_y, shldr_x, shldr_y, hip_x, hip_y)

    # calculate spine length
    spine_length = findDistance(hip_x, hip_y, shldr_x, shldr_y)
    
    return neck_inclination, torso_inclination, offset, offset < 100, spine_length


def main(args):
    ''' dual stream front and side cameras with logging '''

    # start both camera streams
    stream1 = CamGear(source=args.front_cam_idx, logging=True).start() 
    stream2 = CamGear(source=args.side_cam_idx, logging=True).start() 
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

    neck_min = 8
    neck_max = 45

    torso_min = 0
    torso_max = 20
    
    # initialize variables for notification control
    last_notification_time = 0
    notification_cooldown = 30      # cooldown period in seconds
    bad_posture_duration = 0
    bad_posture_threshold = 10      # seconds of continuous bad posture to trigger notification

    # infinite loop
    try: 
        while True:
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
            print(lean)
            neck_inclination, torso_inclination, offset, aligned, spine_length = process_side(side_detection_result, lmPose, frameA.shape, side_cam_pos)

            # predict a posture score and compare to side cam score
            level_ = np.clip((level - level_min) / (level_max - level_min), 0, 1)
            lean_ = np.clip((lean - lean_min) / (lean_max - lean_min), 0, 1)

            neck_ = np.clip((neck_inclination - neck_min) / (neck_max - neck_min), 0, 1)
            torso_ = np.clip((torso_inclination - torso_min) / (torso_max - torso_min), 0, 1)

            eval_posture = 40 * (1 - level_) + 60 * (1 - lean_)
            true_posture = 70 * (1 - neck_) + 30 * (1 - torso_)

            # desk move logic will go here
            # if eval_posture mostly bad for the past t minutes:
            #     send move desk
            #     reset time counter
            
            # phone vibration/sound logic
            current_time = time.time()
            if true_posture < 60:
                bad_posture_duration += 0.5   # since loop runs every 0.5 seconds
            else:
                bad_posture_duration = 0
            if (bad_posture_duration >= bad_posture_threshold and current_time - last_notification_time >= notification_cooldown):
                conn = http.client.HTTPSConnection("api.pushover.net:443")
                """ for Kausar's phone, wink-wink """
                # User key: upc7noc9xcf417q66qk8stmo2akvc7
                # Token: air5p7k39oc8ja6hdoqwyok53dtg3v
                # make POST request to send message
                conn.request("POST", "/1/messages.json",
                urllib.parse.urlencode({
                    "token": "air5p7k39oc8ja6hdoqwyok53dtg3v",
                    "user": "upc7noc9xcf417q66qk8stmo2akvc7",
                    "title": "Posture Alert!",
                    "message": "Yo sit up straight!",
                    "url": "",
                    "priority": "0" 
                }), { "Content-type": "application/x-www-form-urlencoded" })
                # get response
                conn.getresponse()
                # reset notification time and bad posture detection
                last_notification_time = current_time
                bad_posture_duration = 0

            # turn off display from command line with -d
            if not args.display_off:
                # draw landmarks on front image
                front_stick_image = draw_landmarks_front(np.zeros_like(frameA), front_detection_result, lmPose, lean, level, eval_posture)
                frameA = draw_landmarks_front(frameA, front_detection_result, lmPose, lean, level, eval_posture)  # uncomment to overlay landmarks on image
                
                # Draw landmarks on side image
                side_stick_image = draw_landmarks_side(np.zeros_like(frameB), side_detection_result, lmPose, side_cam_pos, neck_inclination, torso_inclination, true_posture, spine_length)
                frameB = draw_landmarks_side(frameB, side_detection_result, lmPose, side_cam_pos, neck_inclination, torso_inclination, true_posture, spine_length)  # uncomment to overlay landmarks on image

                # Display video streams
                cv2.imshow("Front Cam", np.hstack((frameA, front_stick_image)))
                cv2.imshow("Side Cam", np.hstack((frameB, side_stick_image)))

                key = cv2.waitKey(1) & 0xFF
                # check for 'q' key-press
                if key == ord("q"):
                    break   # if q is pressed

            # Append data as a new entry.
            current_time = time.time()
            if current_time - last_write_time >= 0.5:
                timestamp = time.strftime('%Y-%m-%d %H:%M:%S')

                measured_data = {
                    "neck_inclination": neck_inclination,
                    "torso_inclination": torso_inclination,
                    "level": level,
                    "lean": lean,
                    "offset": offset,
                    "aligned": aligned, 
                    "eval_score": eval_posture, 
                    "true_score": true_posture, 
                    "spine_length": spine_length, 
                }
                raw = form_datapoint(front_detection_result, side_detection_result, lmPose, side_cam_pos)
                data_dict = {"timestamp": timestamp, "raw": raw, "measured": measured_data}
                write_to_pickle(pickle_file, data_dict)
                last_write_time = current_time
            
    finally:
        # close output window
        cv2.destroyAllWindows()

        # safely close both video streams
        stream1.stop()
        stream2.stop()
        
        with open(pickle_file, 'rb') as f:
            data = pickle.load(f)
            
        # Convert the data to a pandas DataFrame
        df = pd.DataFrame(data)

        # Write the data to a CSV file
        csv_output_file = os.path.join(current_dir, '..', 'logs/csv', file_name.replace('.pkl', '.csv'))
        df.to_csv(csv_output_file, index=False)
        print(f"Data has been dumped into {csv_output_file}.")

if __name__ == "__main__":
    main(parser.parse_args())