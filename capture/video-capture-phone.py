import cv2
import matplotlib.pyplot as plt

# Replace the URL below with the one provided by your IP camera app
url = "http://10.8.15.10:4747/video"

# Open the IP camera stream
cam = cv2.VideoCapture(url)

# Check if the stream was opened successfully
if not cam.isOpened():
    print("Error: Unable to open video stream")
    exit()

# Get a frame from the IP camera
ret, frame = cam.read()
print("Frame captured successfully:", ret)

if ret:
    # Convert BGR to RGB for displaying
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Display the frame
    plt.figure(figsize=(10, 8))
    plt.imshow(frame_rgb)
    plt.axis('off')
    plt.title('Captured Frame from Phone Camera')
    plt.show()
else:
    print("Failed to capture frame")

# Release the camera
cam.release()