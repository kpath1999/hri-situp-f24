import cv2
import time
import pandas as pd
import math as m
import mediapipe as mp
import os

# Initialize mediapipe selfie segmentation class
mp_pose = mp.solutions.pose
mp_holistic = mp.solutions.holistic

def findDistance(x1, y1, x2, y2):
    dist = m.sqrt((x2-x1)**2+(y2-y1)**2)
    return dist

def findAngle(x1, y1, x2, y2):
    theta = m.acos((y2-y1)*(-y1) / (m.sqrt((x2-x1)**2 + (y2-y1)**2)*y1))
    degree = int(180/m.pi)*theta
    return degree

def sendWarning(x):
    pass

# Initialize frame counters
good_frames = 0
bad_frames = 0

# Font type
font = cv2.FONT_HERSHEY_SIMPLEX

# Colors
blue = (255, 127, 0)
red = (50, 50, 255)
green = (127, 255, 0)
dark_blue = (127, 20, 0)
light_green = (127, 233, 100)
yellow = (0, 255, 255)
pink = (255, 0, 255)

# Initialize mediapipe pose class
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Get the current directory of this script
current_dir = os.path.dirname(__file__)

# Construct the path to the video file
file_name = os.path.join(current_dir, '..', 'videos', 'kausar-posture-input.mp4')

# Normalize the path
file_name = os.path.normpath(file_name)
print(f"Input video file location: {file_name}")
cap = cv2.VideoCapture(file_name)

# Define the output file path
output_file_path = os.path.join(current_dir, '..', 'videos', 'kausar-posture-output.mp4')

# Check if the output file already exists and delete it if it does
if os.path.exists(output_file_path):
    os.remove(output_file_path)
    print(f'Existing file {output_file_path} deleted.')
else:
    print('No existing file to delete.')

# Meta
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_size = (width, height)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# Initalize video writer
video_output = cv2.VideoWriter('videos\kausar-posture-output.mp4', fourcc, fps, frame_size)
print('Video writer initialized')

print("Status:", fps, width, height, frame_size, fourcc)

# Prepare to store the data
data = []

print('Processing..')

while cap.isOpened():
    # Capture frames.
    success, image = cap.read()
    
    if not success:
        print("Null.Frames")
        break
    
    # Get fps.
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Get height and width.
    h, w = image.shape[:2]

    # Get the current timestamp based on frame count
    current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
    timestamp = current_frame / fps
    
    # Convert the BGR image to RGB.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image.
    keypoints = pose.process(image)

    # Convert the image back to BGR.
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Use lm and lmPose as representative of the following methods.
    lm = keypoints.pose_landmarks
    lmPose = mp_pose.PoseLandmark

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
    aligned = False

    # Assist to align the camera to point at the side view of the person.
    # Offset threshold 30 is based on results obtained from analysis over 100 samples.
    if offset < 100:
        cv2.putText(image, str(int(offset)) + ' Aligned', (w - 150, 30), font, 0.9, green, 2)
        aligned = True
    else:
        cv2.putText(image, str(int(offset)) + ' Not Aligned', (w - 150, 30), font, 0.9, red, 2)

    # Calculate angles.
    neck_inclination = findAngle(l_shldr_x, l_shldr_y, l_ear_x, l_ear_y)
    torso_inclination = findAngle(l_hip_x, l_hip_y, l_shldr_x, l_shldr_y)

    # Draw landmarks.
    cv2.circle(image, (l_shldr_x, l_shldr_y), 7, yellow, -1)
    cv2.circle(image, (l_ear_x, l_ear_y), 7, yellow, -1)

    # Let's take y - coordinate of P3 100px above x1,  for display elegance.
    # Although we are taking y = 0 while calculating angle between P1,P2,P3.
    cv2.circle(image, (l_shldr_x, l_shldr_y - 100), 7, yellow, -1)
    cv2.circle(image, (r_shldr_x, r_shldr_y), 7, pink, -1)
    cv2.circle(image, (l_hip_x, l_hip_y), 7, yellow, -1)

    # Similarly, here we are taking y - coordinate 100px above x1. Note that
    # you can take any value for y, not necessarily 100 or 200 pixels.
    cv2.circle(image, (l_hip_x, l_hip_y - 100), 7, yellow, -1)

    # Put text, Posture and angle inclination.
    # Text string for display.
    angle_text_string = 'Neck : ' + str(int(neck_inclination)) + '  Torso : ' + str(int(torso_inclination))

    # Determine whether good posture or bad posture.
    # The threshold angles have been set based on intuition.
    if neck_inclination < 40 and torso_inclination < 10:
        bad_frames = 0
        good_frames += 1

        cv2.putText(image, angle_text_string, (10, 30), font, 0.9, light_green, 2)
        cv2.putText(image, str(int(neck_inclination)), (l_shldr_x + 10, l_shldr_y), font, 0.9, light_green, 2)
        cv2.putText(image, str(int(torso_inclination)), (l_hip_x + 10, l_hip_y), font, 0.9, light_green, 2)

        # Join landmarks.
        cv2.line(image, (l_shldr_x, l_shldr_y), (l_ear_x, l_ear_y), green, 4)
        cv2.line(image, (l_shldr_x, l_shldr_y), (l_shldr_x, l_shldr_y - 100), green, 4)
        cv2.line(image, (l_hip_x, l_hip_y), (l_shldr_x, l_shldr_y), green, 4)
        cv2.line(image, (l_hip_x, l_hip_y), (l_hip_x, l_hip_y - 100), green, 4)

    else:
        good_frames = 0
        bad_frames += 1

        cv2.putText(image, angle_text_string, (10, 30), font, 0.9, red, 2)
        cv2.putText(image, str(int(neck_inclination)), (l_shldr_x + 10, l_shldr_y), font, 0.9, red, 2)
        cv2.putText(image, str(int(torso_inclination)), (l_hip_x + 10, l_hip_y), font, 0.9, red, 2)

        # Join landmarks.
        cv2.line(image, (l_shldr_x, l_shldr_y), (l_ear_x, l_ear_y), red, 4)
        cv2.line(image, (l_shldr_x, l_shldr_y), (l_shldr_x, l_shldr_y - 100), red, 4)
        cv2.line(image, (l_hip_x, l_hip_y), (l_shldr_x, l_shldr_y), red, 4)
        cv2.line(image, (l_hip_x, l_hip_y), (l_hip_x, l_hip_y - 100), red, 4)

    # Calculate the time of remaining in a particular posture.
    good_time = (1 / fps) * good_frames
    bad_time =  (1 / fps) * bad_frames

    # Pose time.
    if good_time > 0:
        time_string_good = 'Good Posture Time : ' + str(round(good_time, 1)) + 's'
        cv2.putText(image, time_string_good, (10, h - 20), font, 0.9, green, 2)
    else:
        time_string_bad = 'Bad Posture Time : ' + str(round(bad_time, 1)) + 's'
        cv2.putText(image, time_string_bad, (10, h - 20), font, 0.9, red, 2)

    # If you stay in bad posture for more than 3 minutes (180s) send an alert.
    if bad_time > 180:
        sendWarning()
    
    # Write frames.
    video_output.write(image)
    
    # Append data as a new entry.
    data.append([timestamp, offset, aligned, neck_inclination, torso_inclination, good_frames, bad_frames, good_time, bad_time])    

print('Finished.')
cap.release()

# Convert to DataFrame
columns = ['Timestamp', 'Offset', 'Aligned?', 'Neck Inclination', 'Torso Inclination', 'Good Frames', 'Bad Frames', 'Good Time (s)', 'Bad Time (s)']
df = pd.DataFrame(data, columns=columns)

# Save DataFrame to CSV
csv_file_path = os.path.join(current_dir, '..', 'logs', 'kausar-posture-log.csv')
df.to_csv(csv_file_path, index=False)

print(f"Data saved to {csv_file_path}")

video_output.release()