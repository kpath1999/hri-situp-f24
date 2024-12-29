# import pandas as pd
# import matplotlib.pyplot as plt

# # Load the CSV file into a DataFrame
# file_path = 'logs/analysis/active_group/yibo_posture_data_2024-12-06-16-41-42.csv'  # Replace with the path to your CSV file
# data = pd.read_csv(file_path)

# # Convert timestamp column to datetime
# data['timestamp'] = pd.to_datetime(data['timestamp'])

# # Compute rolling mean and standard deviation over a window of 50 timestamps
# rolling_window = 50  # Define the smoothing window size
# rolling_mean = data['measured_composite_score'].rolling(window=rolling_window, min_periods=1).mean()
# rolling_std = data['measured_composite_score'].rolling(window=rolling_window, min_periods=1).std()

# # Plot the smoothed mean composite score with shaded standard deviation
# plt.figure(figsize=(10, 6))
# plt.plot(data['timestamp'], rolling_mean, label='Smoothed Mean Composite Score', color='blue', linewidth=2)
# plt.fill_between(
#     data['timestamp'], 
#     rolling_mean - rolling_std, 
#     rolling_mean + rolling_std, 
#     color='blue', 
#     alpha=0.2, 
#     label='Standard Deviation'
# )

# # Formatting the plot
# plt.title('Mean Composite Score Over Time', fontsize=14, fontweight='bold')
# plt.xlabel('Time', fontsize=12)
# plt.ylabel('Composite Score', fontsize=12)
# plt.legend(fontsize=12)
# plt.grid(True, linestyle='--', alpha=0.6)
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.savefig('analysis/viz/active_yibo.png')
# # Show the plot
# plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

# Define the directory containing the CSV files
csv_directory = 'logs/analysis/active_group'  # Replace with the path to your CSV folder
csv_files = glob.glob(os.path.join(csv_directory, '*.csv'))

# Initialize a list to store dataframes
dataframes = []

# Load all CSVs into dataframes
for file in csv_files:
    df = pd.read_csv(file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])  # Ensure consistent timestamp format
    dataframes.append(df)

# Merge all dataframes on the 'timestamp' column
merged_data = pd.concat(dataframes)

# Group data by timestamp and compute the mean and standard deviation
grouped = merged_data.groupby('timestamp').agg(
    mean_composite_score=('measured_composite_score', 'mean'),
    std_composite_score=('measured_composite_score', 'std')
).reset_index()

# Apply a rolling window for smoothing
rolling_window = 50
grouped['smoothed_mean'] = grouped['mean_composite_score'].rolling(window=rolling_window, min_periods=1).mean()
grouped['smoothed_std'] = grouped['std_composite_score'].rolling(window=rolling_window, min_periods=1).mean()

# Plot the smoothed mean composite score with shaded standard deviation
plt.figure(figsize=(10, 6))
plt.plot(grouped['timestamp'], grouped['smoothed_mean'], label='Smoothed Mean Composite Score', color='blue', linewidth=2)
plt.fill_between(
    grouped['timestamp'], 
    grouped['smoothed_mean'] - grouped['smoothed_std'], 
    grouped['smoothed_mean'] + grouped['smoothed_std'], 
    color='blue', 
    alpha=0.2, 
    label='Standard Deviation'
)

# Formatting the plot
plt.title('Smoothed Mean Composite Score Over Time (Averaged Across Files)', fontsize=14, fontweight='bold')
plt.xlabel('Time', fontsize=12)
plt.ylabel('Composite Score', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks(rotation=45)
plt.tight_layout()

# Show the plot
plt.show()

