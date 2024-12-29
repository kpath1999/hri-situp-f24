import cv2
import matplotlib.pyplot as plt

# Open the default camera
cam = cv2.VideoCapture(0)

# Get the default frame width and height
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

print('Frame width:', frame_width)
print('Frame height:', frame_height)

# Get a frame from the capture device
ret, frame = cam.read()
print("Frame captured successfully:", ret)

if ret:
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Display the frame
    plt.figure(figsize=(10, 8))
    plt.imshow(frame_rgb)
    plt.axis('off')
    plt.title('Captured Frame')
    plt.show()
else:
    print("Failed to capture frame")

# Release the camera
cam.release()