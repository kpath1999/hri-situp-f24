import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def process_data(file_path, window_size):
    data = pd.read_csv(file_path)
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    
    metrics = ['measured_neck_inclination', 'measured_torso_inclination', 
               'measured_eval_score', 'measured_true_score', 'measured_composite_score']
    
    smoothed_data = pd.DataFrame()
    for metric in metrics:
        smoothed_data[metric] = data[metric].rolling(window=window_size, center=True).mean()
    
    return data, smoothed_data

def plot_data(ax, data, smoothed_data, title):
    ax.plot(data.index, smoothed_data['measured_neck_inclination'], label='Neck Inclination', linewidth=2)
    ax.plot(data.index, smoothed_data['measured_torso_inclination'], label='Torso Inclination', linewidth=2)
    ax.plot(data.index, smoothed_data['measured_eval_score'], label='Evaluation Score', linewidth=2)
    ax.plot(data.index, smoothed_data['measured_true_score'], label='True Score', linewidth=2)
    ax.plot(data.index, smoothed_data['measured_composite_score'], label='Composite Score', linewidth=2)

    vibration_indices = data.index[data['sent_vibration'] == 1]
    sound_indices = data.index[data['sent_sound'] == 1]
    desk_indices = data.index[data['moved_desk'] == 1]

    for idx in vibration_indices:
        ax.axvline(x=idx, color='red', alpha=0.3, linestyle='--', label='Vibration' if idx == vibration_indices[0] else '')
    for idx in sound_indices:
        ax.axvline(x=idx, color='blue', alpha=0.3, linestyle='--', label='Sound' if idx == sound_indices[0] else '')
    for idx in desk_indices:
        ax.axvline(x=idx, color='green', alpha=0.3, linestyle='--', label='Desk Movement' if idx == desk_indices[0] else '')

    ax.set_title(title, fontsize=28, fontweight='bold')
    ax.set_xlabel('Time Points', fontsize=26, fontweight='bold')
    ax.set_ylabel('Values', fontsize=26, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.grid(True, linestyle='--', alpha=0.7)

    return len(vibration_indices), len(sound_indices), len(desk_indices)

# Set up the figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
fig.suptitle('Posture Measurements Over Time with Events (500-point Moving Average)', fontsize=32, fontweight='bold')

window_size = 500

# Process and plot data for abivishaq
data1, smoothed_data1 = process_data('logs/analysis/passive_group/abivishaq_posture_data.csv', window_size)
v1, s1, d1 = plot_data(ax1, data1, smoothed_data1, '')

# Process and plot data for kausar
data2, smoothed_data2 = process_data('logs/analysis/passive_group/kausar_posture_data.csv', window_size)
v2, s2, d2 = plot_data(ax2, data2, smoothed_data2, '')

# data1, smoothed_data1 = process_data('/Users/anweshagorantla/Desktop/HRI/hri-posture-detection-f24/logs/analysis/active_group/rushil_posture_data_2024-12-06-15-47-56.csv', window_size)
# v1, s1, d1 = plot_data(ax1, data1, smoothed_data1, '')

# Create a single legend for both plots
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.05), ncol=4, fontsize=25)

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(bottom=0.3)

# Save the figure
plt.savefig('analysis/viz/combined_posture_data.png', dpi=300, bbox_inches='tight')
print("Image saved to analysis/viz/combined_posture_data.png")

# Print event counts
print("\nEvent Counts:")
print(f"Abivishaq - Vibrations: {v1}, Sounds: {s1}, Desk Movements: {d1}")
print(f"Kausar - Vibrations: {v2}, Sounds: {s2}, Desk Movements: {d2}")