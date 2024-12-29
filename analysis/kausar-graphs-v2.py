import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta, datetime
import seaborn as sns

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
passive_path = os.path.join(root, "logs", "analysis", "passive_group")

# Initialize lists to store data from all files
all_timestamps = []
all_composite_scores = []

# Go through all CSV files in the folder
for filename in os.listdir(passive_path):
    if filename.endswith('.csv'):
        # Read the CSV file
        df = pd.read_csv(os.path.join(passive_path, filename))

        # Calculate time elapsed in minutes from the start
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['minutes_elapsed'] = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds() / 60
        
        # Append data to the lists
        all_timestamps.extend(df['minutes_elapsed'])
        all_composite_scores.extend(df['measured_composite_score'])

# Convert lists to numpy arrays for easier calculations
timestamps = np.array(all_timestamps)
composite_scores = np.array(all_composite_scores)

# Create bins for every 30 seconds (0.5 minutes)
bins = np.arange(0, 35.5, 0.5)

# Calculate mean and standard deviation for each bin
mean_scores = []
std_scores = []

for i in range(len(bins) - 1):
    mask = (timestamps >= bins[i]) & (timestamps < bins[i+1])
    bin_scores = composite_scores[mask]
    mean_scores.append(np.mean(bin_scores) if len(bin_scores) > 0 else np.nan)
    std_scores.append(np.std(bin_scores) if len(bin_scores) > 0 else np.nan)

# Convert to numpy arrays
mean_scores = np.array(mean_scores)
std_scores = np.array(std_scores)

# For passive group composite score
plt.figure(figsize=(12, 8))
plt.plot(bins[:-1], mean_scores, 'b-', linewidth=2.5, label='Mean Composite Score')
plt.fill_between(bins[:-1], mean_scores - std_scores, mean_scores + std_scores, alpha=0.2, color='b', label='Standard Deviation')
plt.xlabel('Time (minutes)', fontsize=16, fontweight='bold')
plt.ylabel('Composite Score', fontsize=16, fontweight='bold')
plt.title('Passive Group - Mean Composite Score Over Time', fontsize=20, fontweight='bold')
plt.legend(fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()
plt.savefig('analysis/viz/passive_composite_score_over_time.png', dpi=300, bbox_inches='tight')
plt.close()

# Print some statistics
print(f"Overall mean composite score: {np.mean(composite_scores):.2f}")
print(f"Overall standard deviation of composite score: {np.std(composite_scores):.2f}")
print(f"Total number of data points: {len(composite_scores)}")
print(f"Data saved to 'passive_composite_score_over_time.png'")

control_path = os.path.join(root, "logs", "analysis", "control_group")

# Initialize lists to store data from all files
all_timestamps = []
all_composite_scores = []

# Go through all CSV files in the folder
for filename in os.listdir(control_path):
    if filename.endswith('.csv'):
        # Read the CSV file
        df = pd.read_csv(os.path.join(control_path, filename))

        # Calculate time elapsed in minutes from the start
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['minutes_elapsed'] = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds() / 60
        
        # Append data to the lists
        all_timestamps.extend(df['minutes_elapsed'])
        all_composite_scores.extend(df['measured_composite_score'])

# Convert lists to numpy arrays for easier calculations
timestamps = np.array(all_timestamps)
composite_scores = np.array(all_composite_scores)

# Create bins for every 30 seconds (0.5 minutes)
bins = np.arange(0, 35.5, 0.5)

# Calculate mean and standard deviation for each bin
mean_scores = []
std_scores = []

for i in range(len(bins) - 1):
    mask = (timestamps >= bins[i]) & (timestamps < bins[i+1])
    bin_scores = composite_scores[mask]
    mean_scores.append(np.mean(bin_scores) if len(bin_scores) > 0 else np.nan)
    std_scores.append(np.std(bin_scores) if len(bin_scores) > 0 else np.nan)
    
# Convert to numpy arrays
mean_scores = np.array(mean_scores)
std_scores = np.array(std_scores)

# For control group composite score
plt.figure(figsize=(12, 8))
plt.plot(bins[:-1], mean_scores, 'r-', linewidth=2.5, label='Mean Composite Score')
plt.fill_between(bins[:-1], mean_scores - std_scores, mean_scores + std_scores, alpha=0.2, color='r', label='Standard Deviation')
plt.xlabel('Time (minutes)', fontsize=16, fontweight='bold')
plt.ylabel('Composite Score', fontsize=16, fontweight='bold')
plt.title('Control Group - Mean Composite Score Over Time', fontsize=20, fontweight='bold')
plt.legend(fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()
plt.savefig('analysis/viz/control_composite_score_over_time.png', dpi=300, bbox_inches='tight')
plt.close()

# Print some statistics
print(f"Overall mean composite score: {np.mean(composite_scores):.2f}")
print(f"Overall standard deviation of composite score: {np.std(composite_scores):.2f}")
print(f"Total number of data points: {len(composite_scores)}")
print(f"Data saved to 'control_composite_score_over_time.png'")

all_timestamps = []
all_neck_inclinations = []
all_torso_inclinations = []
all_spine_lengths = []

for filename in os.listdir(passive_path):
    if filename.endswith('.csv'):
        df = pd.read_csv(os.path.join(passive_path, filename))
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['minutes_elapsed'] = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds() / 60
        
        all_timestamps.extend(df['minutes_elapsed'])
        all_neck_inclinations.extend(df['measured_neck_inclination'])
        all_torso_inclinations.extend(df['measured_torso_inclination'])
        all_spine_lengths.extend(df['measured_spine_length'])

timestamps = np.array(all_timestamps)
neck_inclinations = np.array(all_neck_inclinations)
torso_inclinations = np.array(all_torso_inclinations)
spine_lengths = np.array(all_spine_lengths)

# Calculate means for each bin
mean_neck = []
mean_torso = []
mean_spine = []

for i in range(len(bins) - 1):
    mask = (timestamps >= bins[i]) & (timestamps < bins[i+1])
    mean_neck.append(np.mean(neck_inclinations[mask]) if np.any(mask) else np.nan)
    mean_torso.append(np.mean(torso_inclinations[mask]) if np.any(mask) else np.nan)
    mean_spine.append(np.mean(spine_lengths[mask]) if np.any(mask) else np.nan)

# For posture metrics plot
plt.figure(figsize=(12, 8))
plt.plot(bins[:-1], mean_neck, label='Neck Inclination', color='blue', linewidth=2.5)
plt.plot(bins[:-1], mean_torso, label='Torso Inclination', color='red', linewidth=2.5)
plt.plot(bins[:-1], mean_spine, label='Spine Length', color='green', linewidth=2.5)
plt.xlabel('Time (minutes)', fontsize=16, fontweight='bold')
plt.ylabel('Measurement Value', fontsize=16, fontweight='bold')
plt.title('Passive Group - Mean Posture Metrics Over Time', fontsize=20, fontweight='bold')
plt.legend(fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.ylim(bottom=0)
plt.tight_layout()
plt.savefig('analysis/viz/passive_group_posture_metrics.png', dpi=300, bbox_inches='tight')
plt.close()

print("Graph saved as 'passive_group_posture_metrics.png'")

passive_path_with_correct_interventions = os.path.join(root, "logs", "archive", "csv")

all_timestamps = []
all_neck_inclinations = []
all_torso_inclinations = []
all_spine_lengths = []
all_composite_scores = []
all_vibrations = []
all_sounds = []

for filename in os.listdir(passive_path_with_correct_interventions):
    if filename.endswith('.csv'):
        df = pd.read_csv(os.path.join(passive_path_with_correct_interventions, filename))
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['minutes_elapsed'] = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds() / 60
        
        all_timestamps.extend(df['minutes_elapsed'])
        all_neck_inclinations.extend(df['measured_neck_inclination'])
        all_torso_inclinations.extend(df['measured_torso_inclination'])
        all_spine_lengths.extend(df['measured_spine_length'])
        all_composite_scores.extend(df['measured_composite_score'])
        all_vibrations.extend(df['sent_vibration'])
        all_sounds.extend(df['sent_sound'])

timestamps = np.array(all_timestamps)
neck_inclinations = np.array(all_neck_inclinations)
torso_inclinations = np.array(all_torso_inclinations)
spine_lengths = np.array(all_spine_lengths)
composite_scores = np.array(all_composite_scores)
vibrations = np.array(all_vibrations)
sounds = np.array(all_sounds)

# Function to calculate improvement after intervention
def calculate_improvement(scores, interventions, window=15):
    improvements = []
    for i in range(len(scores)):
        if interventions[i] == 1:
            before = np.mean(scores[max(0, i-window):i])
            after = np.mean(scores[i:min(len(scores), i+window)])
            improvements.append(after - before)
    return improvements

# Calculate improvements
vibration_improvements = calculate_improvement(composite_scores, vibrations)
sound_improvements = calculate_improvement(composite_scores, sounds)

# Assuming vibration_improvements and sound_improvements are lists
# Create a DataFrame from your data
df = pd.DataFrame({
    'Improvement': vibration_improvements + sound_improvements,
    'Intervention': ['Vibration'] * len(vibration_improvements) + ['Sound'] * len(sound_improvements)
})

# For intervention effectiveness boxplot
plt.figure(figsize=(12, 8))
sns.set_style("whitegrid")
sns.set_palette(['lightblue', 'lightgreen'])
ax = sns.boxplot(x='Intervention', y='Improvement', data=df, width=0.6)
plt.title('Posture Improvement After Interventions', fontsize=26, fontweight='bold')
plt.xlabel('Intervention Type', fontsize=22, fontweight='bold')
plt.ylabel('Change in Composite Score', fontsize=22, fontweight='bold')
plt.xticks(fontsize=18, fontweight='bold')
plt.yticks(fontsize=18, fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('analysis/viz/intervention_effectiveness.png', dpi=300, bbox_inches='tight')
plt.close()

print("Graph saved as 'intervention_effectiveness.png'")