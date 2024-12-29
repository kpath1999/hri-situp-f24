import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the data from CSV file
data = pd.read_csv('logs/analysis/passive_group/abivishaq_posture_data.csv')

# Convert timestamp to datetime
data['timestamp'] = pd.to_datetime(data['timestamp'])

# Calculate rolling averages (window size of 50 data points)
window_size = 500
metrics = ['measured_neck_inclination', 'measured_torso_inclination', 
           'measured_eval_score', 'measured_true_score', 'measured_composite_score']

smoothed_data = pd.DataFrame()
for metric in metrics:
    smoothed_data[metric] = data[metric].rolling(window=window_size, center=True).mean()

# Create the plot
plt.figure(figsize=(15, 10))

# Plot the smoothed metrics
plt.plot(data.index, smoothed_data['measured_neck_inclination'], 
         label='Neck Inclination', linewidth=2)
plt.plot(data.index, smoothed_data['measured_torso_inclination'], 
         label='Torso Inclination', linewidth=2)
plt.plot(data.index, smoothed_data['measured_eval_score'], 
         label='Evaluation Score', linewidth=2)
plt.plot(data.index, smoothed_data['measured_true_score'], 
         label='True Score', linewidth=2)
plt.plot(data.index, smoothed_data['measured_composite_score'], 
         label='Composite Score', linewidth=2)

# Add horizontal lines for events
y_max = smoothed_data.max().max()
y_min = smoothed_data.min().min()
y_range = y_max - y_min

# Add event lines with different y-positions to avoid overlap
vibration_indices = data.index[data['sent_vibration'] == 1]
sound_indices = data.index[data['sent_sound'] == 1]
desk_indices = data.index[data['moved_desk'] == 1]

# print("vibrations:", vibration_indices, "sounds", sound_indices)
for idx in vibration_indices:
    plt.axvline(x=idx, color='red', alpha=0.3, linestyle='--', label='Vibration' if idx == vibration_indices[0] else '')

for idx in sound_indices:
    plt.axvline(x=idx, color='blue', alpha=0.3, linestyle='--', label='Sound' if idx == sound_indices[0] else '')

for idx in desk_indices:
    plt.axvline(x=idx, color='green', alpha=0.3, linestyle='--', label='Desk Movement' if idx == desk_indices[0] else '')

# Customize the plot
plt.title(f'Posture Measurements Over Time with Events\n({window_size}-point Moving Average)', fontsize=14, pad=20)
plt.xlabel('Time Points', fontsize=12)
plt.ylabel('Values', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# Print event counts
print("\nEvent Counts:")
print(f"Vibrations: {len(vibration_indices)}")
print(f"Sounds: {len(sound_indices)}")
print(f"Desk Movements: {len(desk_indices)}")

# Adjust layout and display
plt.tight_layout()
# plt.show()

plt.savefig('analysis/viz/abivishaq_posture_data.png')
print("Image saved to analysis/viz/abivishaq_posture_data.png")


# do the same for kausar
window_size = 500
data = pd.read_csv('logs/analysis/passive_group/kausar_posture_data.csv')

# Convert timestamp to datetime
data['timestamp'] = pd.to_datetime(data['timestamp'])

# Calculate rolling averages (window size of 50 data points)
metrics = ['measured_neck_inclination', 'measured_torso_inclination', 
           'measured_eval_score', 'measured_true_score', 'measured_composite_score']

smoothed_data = pd.DataFrame()
for metric in metrics:
    smoothed_data[metric] = data[metric].rolling(window=window_size, center=True).mean()

# Create the plot
plt.figure(figsize=(15, 10))

# Plot the smoothed metrics
plt.plot(data.index, smoothed_data['measured_neck_inclination'], 
         label='Neck Inclination', linewidth=2)
plt.plot(data.index, smoothed_data['measured_torso_inclination'], 
         label='Torso Inclination', linewidth=2)
plt.plot(data.index, smoothed_data['measured_eval_score'], 
         label='Evaluation Score', linewidth=2)
plt.plot(data.index, smoothed_data['measured_true_score'], 
         label='True Score', linewidth=2)
plt.plot(data.index, smoothed_data['measured_composite_score'], 
         label='Composite Score', linewidth=2)

# Add horizontal lines for events
y_max = smoothed_data.max().max()
y_min = smoothed_data.min().min()
y_range = y_max - y_min

# Add event lines with different y-positions to avoid overlap
vibration_indices = data.index[data['sent_vibration'] == 1]
sound_indices = data.index[data['sent_sound'] == 1]
desk_indices = data.index[data['moved_desk'] == 1]

# print("vibrations:", vibration_indices, "sounds", sound_indices)
for idx in vibration_indices:
    plt.axvline(x=idx, color='red', alpha=0.3, linestyle='--', label='Vibration' if idx == vibration_indices[0] else '')

for idx in sound_indices:
    plt.axvline(x=idx, color='blue', alpha=0.3, linestyle='--', label='Sound' if idx == sound_indices[0] else '')

for idx in desk_indices:
    plt.axvline(x=idx, color='green', alpha=0.3, linestyle='--', label='Desk Movement' if idx == desk_indices[0] else '')

# Customize the plot
plt.title(f'Posture Measurements Over Time with Events\n({window_size}-point Moving Average)', fontsize=14, pad=20)
plt.xlabel('Time Points', fontsize=12)
plt.ylabel('Values', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# Print event counts
print("\nEvent Counts:")
print(f"Vibrations: {len(vibration_indices)}")
print(f"Sounds: {len(sound_indices)}")
print(f"Desk Movements: {len(desk_indices)}")

# Adjust layout and display
plt.tight_layout()
# plt.show()

plt.savefig('analysis/viz/kausar_passiveposture_data.png')
print("Image saved to analysis/viz/kausar_passiveposture_data.png")

# [ 600,  660,  728,  788,  848,  942, 1002, 1062, 1156, 1264, 1324, 1384, 1457, 1519, 1581, 1643, 1737, 1799, 1887, 2011, 2079, 2146, 2209, 
#  2222,2273, 2323, 2375, 2425, 2477, 2526, 2577, 2723, 2773]
# [ 839,  890, 2400, 2460, 2520, 2580, 2654, 2714, 2774, 2834, 2894, 2961,3020, 3086, 3146, 3212, 3272, 3352, 3447, 3538, 3598, 3658, 3748, 
#  3830,3894, 3994, 4157]