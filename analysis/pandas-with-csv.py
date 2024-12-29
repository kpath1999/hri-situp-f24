import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np
import os
import sys
import logging

# Create a 'plot' subfolder if it doesn't exist
plot_folder = os.path.join(os.path.dirname(__file__), 'plot')
os.makedirs(plot_folder, exist_ok=True)

# Create a log file
log_file = os.path.join(plot_folder, 'test_analysis_log.txt')

# Redirect stdout to the log file
sys.stdout = open(log_file, 'w')

# Load the data
csv_file_path = os.path.join(os.path.dirname(__file__), '..', 'logs/csv/test_posture_data_2024-11-02-21-46-48.csv')
df = pd.read_csv(csv_file_path)

# Convert timestamp to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Basic EDA functions
def plot_time_series(df, column, title):
    plt.figure(figsize=(12, 6))
    plt.plot(df['timestamp'], df[column])
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel(column)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_folder, f"test_{title.replace(' ', '_').lower()}.png"))
    plt.close()
    print(f"Saved plot: test_{title.replace(' ', '_').lower()}.png")

def plot_histogram(df, column, title):
    plt.figure(figsize=(10, 6))
    sns.histplot(df[column], kde=True)
    plt.title(title)
    plt.xlabel(column)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_folder, f"test_{title.replace(' ', '_').lower()}.png"))
    plt.close()
    print(f"Saved plot: test_{title.replace(' ', '_').lower()}.png")

def plot_correlation_heatmap(df, title):
    plt.figure(figsize=(12, 10))
    sns.heatmap(df.corr(), annot=False, cmap='coolwarm')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_folder, f"test_{title.replace(' ', '_').lower()}.png"))
    plt.close()
    print(f"Saved plot: test_{title.replace(' ', '_').lower()}.png")

def plot_scatter(df, x, y, title):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x=x, y=y)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_folder, f"test_{title.replace(' ', '_').lower()}.png"))
    plt.close()
    print(f"Saved plot: test_{title.replace(' ', '_').lower()}.png")

# Perform EDA
print("Basic Statistics:")
print(df.describe())
logging.info("Basic Statistics:")
logging.info(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())
logging.info("\nMissing Values:")
logging.info(df.isnull().sum())

# Time series plots
plot_time_series(df, 'measured_neck_inclination', 'Neck Inclination Over Time')
plot_time_series(df, 'measured_torso_inclination', 'Torso Inclination Over Time')
plot_time_series(df, 'measured_eval_score', 'Evaluation Score Over Time')
plot_time_series(df, 'measured_true_score', 'True Score Over Time')

# Histograms
plot_histogram(df, 'measured_neck_inclination', 'Distribution of Neck Inclination')
plot_histogram(df, 'measured_torso_inclination', 'Distribution of Torso Inclination')
plot_histogram(df, 'measured_eval_score', 'Distribution of Evaluation Score')
plot_histogram(df, 'measured_true_score', 'Distribution of True Score')

# Correlation heatmap
measured_columns = [col for col in df.columns if col.startswith('measured_')]
plot_correlation_heatmap(df[measured_columns], 'Correlation Heatmap of Measured Variables')

# Scatter plots
plot_scatter(df, 'measured_neck_inclination', 'measured_torso_inclination', 'Neck vs Torso Inclination')
plot_scatter(df, 'measured_eval_score', 'measured_true_score', 'Evaluation Score vs True Score')

# Additional analyses
print("\nCorrelation between Evaluation Score and True Score:")
print(df['measured_eval_score'].corr(df['measured_true_score']))
logging.info("\nCorrelation between Evaluation Score and True Score:")
logging.info(df['measured_eval_score'].corr(df['measured_true_score']))

print("\nAverage Neck and Torso Inclination:")
print(f"Average Neck Inclination: {df['measured_neck_inclination'].mean():.2f}")
print(f"Average Torso Inclination: {df['measured_torso_inclination'].mean():.2f}")
logging.info("\nAverage Neck and Torso Inclination:")
logging.info(f"Average Neck Inclination: {df['measured_neck_inclination'].mean():.2f}")
logging.info(f"Average Torso Inclination: {df['measured_torso_inclination'].mean():.2f}")

# Detect potential outliers
z_scores = np.abs(stats.zscore(df[['measured_neck_inclination', 'measured_torso_inclination']]))
outliers = np.where(z_scores > 3)
print("\nPotential Outliers (Z-score > 3):")
print(df.iloc[outliers[0]][['timestamp', 'measured_neck_inclination', 'measured_torso_inclination']])
logging.info("\nPotential Outliers (Z-score > 3):")
logging.info(df.iloc[outliers[0]][['timestamp', 'measured_neck_inclination', 'measured_torso_inclination']])

# Time-based analysis
df['hour'] = df['timestamp'].dt.hour
hourly_avg = df.groupby('hour')[['measured_neck_inclination', 'measured_torso_inclination']].mean()
print("\nHourly Averages:")
print(hourly_avg)
logging.info("\nHourly Averages:")
logging.info(hourly_avg)

plt.figure(figsize=(12, 6))
hourly_avg.plot()
plt.title('Average Inclinations by Hour')
plt.xlabel('Hour of Day')
plt.ylabel('Inclination')
plt.legend(['Neck Inclination', 'Torso Inclination'])
plt.tight_layout()
plt.savefig(os.path.join(plot_folder, "test_average_inclinations_by_hour.png"))
plt.close()
print("Saved plot: test_average_inclinations_by_hour.png")

print(f"\nAll plots and this log have been saved in the '{plot_folder}' directory.")

# Close the log file
sys.stdout.close()

# Reset stdout to its default value
sys.stdout = sys.__stdout__

print(f"Analysis complete. Log file saved as {log_file}")