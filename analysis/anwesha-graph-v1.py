import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file into a DataFrame
file_path = 'logs/analysis/active_group/rushil_posture_data_2024-12-06-15-47-56.csv'  # Replace with the path to your CSV file
data = pd.read_csv(file_path)

# Convert timestamp column to datetime
data['timestamp'] = pd.to_datetime(data['timestamp'])

# Calculate the mean composite score over time
data['mean_composite_score'] = data['measured_composite_score'].expanding().mean()

# Plot the mean composite score over time
plt.figure(figsize=(12, 6))
plt.plot(data['timestamp'], data['mean_composite_score'], label='Mean Composite Score', linewidth=2)

# Add markers for desk movements
desk_up = data[data['moved_desk'] == 1]
desk_down = data[data['moved_desk'] == -1]

plt.scatter(desk_up['timestamp'], desk_up['mean_composite_score'], color='green', label='Desk Moved Up', zorder=5)
plt.scatter(desk_down['timestamp'], desk_down['mean_composite_score'], color='red', label='Desk Moved Down', zorder=5)

# Formatting the plot
plt.xlabel('Timestamp')
plt.ylabel('Mean Composite Score')
plt.title('Mean Composite Score Over Time with Desk Movements (Active)')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('analysis/viz/active_mean_composite_score_over_time_rushil.png')

# Show the plot
plt.show()
