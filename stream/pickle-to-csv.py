import pickle
import os
import json
import pandas as pd
from datetime import datetime

# Specify the pickle file path and the output files
current_dir = os.path.dirname(__file__)
pick_file_name = "test_posture_data_2024-11-02-21-33-40.pkl"
pickle_file = os.path.join(current_dir, '..', 'logs', pick_file_name)

txt_file_name = "test_posture_data_2024-11-02-21-33-40.txt"
txt_output_file = os.path.join(current_dir, '..', 'logs', txt_file_name)

csv_file_name = "test_posture_data_2024-11-02-21-33-40.csv"
csv_output_file = os.path.join(current_dir, '..', 'logs/csv', csv_file_name)

# Load data from the pickle file
with open(pickle_file, 'rb') as f:
    data = pickle.load(f)

# Write the data to a text file
with open(txt_output_file, 'w') as f:
    # Attempt to write as JSON if possible for better readability
    try:
        json.dump(data, f, indent=4)
    except (TypeError, OverflowError):
        # Fallback to string representation if JSON serialization fails
        f.write(str(data))

print(f"Data has been dumped into {txt_output_file}.")

with open(pickle_file, 'rb') as f:
    data = pickle.load(f)
    
# Convert the data to a pandas DataFrame
df = pd.DataFrame(data)

# Write the data to a CSV file
df.to_csv(csv_output_file, index=False)
print(f"Data has been dumped into {csv_output_file}.")