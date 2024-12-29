import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the data from CSV file
# data = pd.read_csv('logs/analysis/active_group/rushil_posture_data_2024-12-06-15-47-56.csv')
data = pd.read_csv('logs/analysis/active_group/su_posture_data_2024-12-06-18-30-28.csv')

# Convert timestamp to datetime
data['timestamp'] = pd.to_datetime(data['timestamp'])

# Calculate rolling averages (window size of 500 data points)
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

# Add markers for moved_desk events
moved_desk_positive = data.index[data['moved_desk'] == 1]
moved_desk_negative = data.index[data['moved_desk'] == -1]

for idx in moved_desk_positive:
    plt.axvline(x=idx, color='green', alpha=0.5, linestyle='--', label='Desk Moved Up' if idx == moved_desk_positive[0] else '')

for idx in moved_desk_negative:
    plt.axvline(x=idx, color='purple', alpha=0.5, linestyle='--', label='Desk Moved Down' if idx == moved_desk_negative[0] else '')

# Customize the plot
plt.title(f'Posture Measurements Over Time with Desk Movements\n({window_size}-point Moving Average)', fontsize=14, pad=20)
plt.xlabel('Time Points', fontsize=12)
plt.ylabel('Values', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# Adjust layout and display
plt.tight_layout()

# Save the figure
plt.savefig('analysis/viz/Su_desk_movement.png')
print("Image saved to analysis/viz/Su_desk_movement.png")
